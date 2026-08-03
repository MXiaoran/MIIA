from __future__ import annotations

import argparse
from pathlib import Path

import torch

from miia.checkpoint import load_checkpoint, load_online_backbone_checkpoint
from miia.config import load_config, resolve_path
from miia.data.manifest import load_manifest
from miia.evaluation import evaluate_records
from miia.reporting import write_reports


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a MIIA checkpoint on RET-3")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/ret3_single_gpu.yaml")
    parser.add_argument("--datasets", nargs="+", choices=["rsicd", "rsitmd", "ucm"], default=["rsicd", "rsitmd", "ucm"])
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device)
    records, _ = load_manifest(resolve_path(config, config["data"]["manifest"]))
    if config.get("debug", {}).get("tiny_model", False):
        from miia.models.builders import create_tiny_miia

        image_size = int(config["debug"].get("tiny_image_size", 32))
        config["data"]["image_size"] = image_size
        model, tokenizer = create_tiny_miia(image_size)
    else:
        from miia.models.clip_backbone import create_clip_backbone

        model, tokenizer = create_clip_backbone(
            config["model"]["clip_model"],
            pretrained=None,
            image_size=int(config["data"]["image_size"]),
        )
    if config.get("debug", {}).get("tiny_model", False):
        load_checkpoint(args.checkpoint, model, restore_rng=False)
    else:
        load_online_backbone_checkpoint(args.checkpoint, model)
    model.to(device)
    results = {}
    for dataset in args.datasets:
        subset = [record for record in records if record.dataset == dataset and record.split == args.split]
        if not subset:
            raise ValueError(f"No {args.split} records found for {dataset}")
        results[dataset] = evaluate_records(
            model,
            tokenizer,
            subset,
            device=device,
            image_size=int(config["data"]["image_size"]),
            batch_size=int(config["evaluation"]["batch_size"]),
        )
    output = Path(args.report_dir) if args.report_dir else resolve_path(config, config["evaluation"]["report_dir"])
    paths = write_reports(results, output)
    for kind, path in paths.items():
        print(f"{kind}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
