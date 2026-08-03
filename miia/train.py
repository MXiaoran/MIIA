from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from miia.checkpoint import load_checkpoint, save_checkpoint
from miia.config import load_config, resolve_path
from miia.data.dataset import (
    RetrievalDataset,
    build_dataloader,
    build_dvae_transform,
    build_train_transform,
)
from miia.data.manifest import DatasetRecord, load_manifest
from miia.evaluation import evaluate_records
from miia.models.miia import MIIA
from miia.scheduler import WarmupCosineScheduler
from miia.utils import set_seed


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def build_optimizer(model: MIIA, config: dict[str, Any]) -> torch.optim.Optimizer:
    training = config["training"]
    pretrained = list(model.pretrained_parameters())
    new_modules = list(model.new_module_parameters())
    if not pretrained or not new_modules:
        raise RuntimeError("Expected both pretrained CLIP and newly initialized MIIA parameter groups")
    return torch.optim.Adam(
        [
            {"params": pretrained, "lr": float(training["pretrained_lr"]), "name": "pretrained"},
            {"params": new_modules, "lr": float(training["new_module_lr"]), "name": "new_modules"},
        ],
        weight_decay=float(training["weight_decay"]),
    )


def combined_validation_mr(
    model: MIIA,
    tokenizer,
    records: list[DatasetRecord],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[float, dict[str, Any]]:
    results: dict[str, Any] = {}
    for dataset in config["evaluation"]["datasets"]:
        subset = [record for record in records if record.dataset == dataset and record.split == "val"]
        if not subset:
            continue
        results[dataset] = evaluate_records(
            model,
            tokenizer,
            subset,
            device=device,
            image_size=int(config["data"]["image_size"]),
            batch_size=int(config["evaluation"]["batch_size"]),
        )
    score = float(sum(value["mR"] for value in results.values()) / len(results)) if results else float("-inf")
    return score, results


def train(config: dict[str, Any], max_steps: int | None = None, skip_validation: bool = False) -> Path:
    experiment = config["experiment"]
    seed = int(experiment["seed"])
    set_seed(seed, deterministic=bool(experiment.get("deterministic", False)))
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        print("WARNING: CUDA is unavailable. A full MIIA run is intended for a GPU; CPU is suitable only for diagnostics.")
    else:
        device = torch.device("cuda")

    full_records, _ = load_manifest(resolve_path(config, config["data"]["manifest"]))
    train_records, train_metadata = load_manifest(resolve_path(config, config["data"]["train_manifest"]))
    if config.get("debug", {}).get("tiny_model", False):
        from miia.models.builders import create_tiny_miia

        tiny_size = int(config["debug"].get("tiny_image_size", 32))
        config["data"]["image_size"] = tiny_size
        config["data"]["dvae_target_size"] = tiny_size // 2
        model, tokenizer = create_tiny_miia(tiny_size)
    else:
        model, tokenizer = MIIA.from_config(config)
    model.to(device)
    model.initialize_momentum()

    micro_batch = int(config["data"]["micro_batch_size"])
    logical_batch = int(config["data"]["logical_batch_size"])
    if logical_batch % micro_batch:
        raise ValueError("logical_batch_size must be divisible by micro_batch_size")
    accumulation_steps = logical_batch // micro_batch
    optimizer = build_optimizer(model, config)
    counts_by_dataset = {
        dataset_name: sum(record.dataset == dataset_name for record in train_records)
        for dataset_name in {record.dataset for record in train_records}
    }
    micro_batches_per_epoch = sum(
        ((count // logical_batch) * logical_batch) // micro_batch
        for count in counts_by_dataset.values()
    )
    if micro_batches_per_epoch == 0:
        raise ValueError(
            "No complete logical batch can be formed. Lower data.logical_batch_size "
            "or prepare the complete training datasets."
        )
    optimizer_steps_per_epoch = max(1, micro_batches_per_epoch // accumulation_steps)
    total_steps = int(config["training"]["epochs"]) * optimizer_steps_per_epoch
    warmup_steps = int(config["training"]["warmup_epochs"]) * optimizer_steps_per_epoch
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        warmup_start_lr=float(config["training"]["warmup_start_lr"]),
        min_lr=float(config["training"]["min_lr"]),
    )
    amp_enabled = bool(config["training"]["amp"]) and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

    output_dir = resolve_path(config, experiment["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    log_path = output_dir / "train.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    resume_batch = 0
    global_step = 0
    best_metric = float("-inf")
    resume = config["training"].get("resume")
    if resume:
        payload = load_checkpoint(resolve_path(config, resume), model, optimizer, scheduler, scaler)
        saved_epoch = int(payload["epoch"])
        resume_batch = int(payload.get("batch_in_epoch", 0))
        epoch_complete = bool(payload.get("epoch_complete", True))
        start_epoch = saved_epoch + 1 if epoch_complete else saved_epoch
        global_step = int(payload["global_step"])
        best_metric = float(payload["best_metric"])
        expected_hash = train_metadata.get("manifest_hash", "")
        if payload.get("manifest_hash") != expected_hash:
            raise RuntimeError("Training manifest hash differs from the checkpoint; refusing an inexact resume")

    dataset = RetrievalDataset(
        train_records,
        tokenizer,
        image_transform=build_train_transform(int(config["data"]["image_size"])),
        dvae_transform=build_dvae_transform(int(config["data"]["dvae_target_size"])),
        training=True,
        seed=seed,
    )
    loader = build_dataloader(
        dataset,
        batch_size=micro_batch,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=bool(config["data"]["pin_memory"]),
        training=True,
        seed=seed,
        logical_batch_size=logical_batch,
    )

    optimizer.zero_grad(set_to_none=True)
    pending_keys: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
    accumulation_dataset: str | None = None
    accumulation_count = 0
    stop = False
    for epoch in range(start_epoch, int(config["training"]["epochs"])):
        model.train()
        dataset.set_epoch(epoch)
        if hasattr(loader.batch_sampler, "set_epoch"):
            loader.batch_sampler.set_epoch(epoch)
        if hasattr(loader.batch_sampler, "start_offset"):
            loader.batch_sampler.start_offset = resume_batch if epoch == start_epoch else 0
        running: dict[str, float] = defaultdict(float)
        samples = 0
        progress = tqdm(loader, desc=f"epoch {epoch + 1}")
        for batch_index, batch in enumerate(progress):
            batch = move_batch(batch, device)
            current_dataset = batch["dataset"][0]
            # A logical batch must be source-homogeneous. If the sampler changes
            # datasets, close the previous accumulation window first.
            if accumulation_count and accumulation_dataset != current_dataset:
                scaler.unscale_(optimizer)
                if config["training"].get("grad_clip_norm"):
                    nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["grad_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                model.update_momentum()
                for dataset_name, chunks in pending_keys.items():
                    model.enqueue(dataset_name, torch.cat([chunk[0] for chunk in chunks]), torch.cat([chunk[1] for chunk in chunks]))
                pending_keys.clear()
                global_step += 1
                accumulation_count = 0
            accumulation_dataset = current_dataset
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                output = model(batch)
                loss = output.loss / accumulation_steps
            scaler.scale(loss).backward()
            accumulation_count += 1
            pending_keys[output.dataset].append((output.queued_image_keys.detach(), output.queued_text_keys.detach()))
            batch_size = batch["image"].shape[0]
            samples += batch_size
            running["loss"] += float(output.loss.detach()) * batch_size
            for name, value in output.losses.items():
                running[name] += float(value.detach()) * batch_size

            boundary = accumulation_count == accumulation_steps
            if boundary:
                scaler.unscale_(optimizer)
                if config["training"].get("grad_clip_norm"):
                    nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["grad_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                model.update_momentum()
                for dataset_name, chunks in pending_keys.items():
                    model.enqueue(
                        dataset_name,
                        torch.cat([chunk[0] for chunk in chunks]),
                        torch.cat([chunk[1] for chunk in chunks]),
                    )
                pending_keys.clear()
                global_step += 1
                accumulation_count = 0
                if max_steps is not None and global_step >= max_steps:
                    stop = True
            progress.set_postfix(loss=running["loss"] / max(1, samples), step=global_step)
            if stop:
                break

        validation_results: dict[str, Any] = {}
        validation_metric = float("nan")
        should_validate = not skip_validation and (epoch + 1) % int(config["training"]["validate_every_epochs"]) == 0
        if should_validate:
            validation_metric, validation_results = combined_validation_mr(model, tokenizer, full_records, config, device)
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "samples": samples,
            "losses": {key: value / max(1, samples) for key, value in running.items()},
            "validation_mR": validation_metric,
            "validation": validation_results,
            "lr": {group.get("name", str(index)): group["lr"] for index, group in enumerate(optimizer.param_groups)},
            "cuda_peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0,
            "time": time.time(),
        }
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

        checkpoint_path = checkpoint_dir / "last.pth"
        completed_epoch = not stop
        batch_in_epoch = 0 if completed_epoch else resume_batch + batch_index + 1
        if should_validate and validation_metric > best_metric:
            best_metric = validation_metric
            save_checkpoint(
                checkpoint_dir / "best.pth",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                global_step,
                best_metric,
                config,
                train_metadata.get("manifest_hash", ""),
                batch_in_epoch=0,
                epoch_complete=True,
            )
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            global_step,
            best_metric,
            config,
            train_metadata.get("manifest_hash", ""),
            batch_in_epoch=batch_in_epoch,
            epoch_complete=completed_epoch,
        )
        resume_batch = 0
        if stop:
            break
    return checkpoint_dir / "last.pth"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train MIIA on the RET-3 union")
    parser.add_argument("--config", default="configs/ret3_single_gpu.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="stop after N optimizer steps (for smoke tests)")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.resume:
        config["training"]["resume"] = args.resume
    checkpoint = train(config, max_steps=args.max_steps, skip_validation=args.skip_validation)
    print(f"checkpoint: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
