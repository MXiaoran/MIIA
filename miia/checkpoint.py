from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from miia.utils import capture_rng_state, restore_rng_state


def _checkpoint_model_state(model: torch.nn.Module) -> dict[str, Any]:
    # The frozen DALL-E encoder is a downloaded dependency, not a learned MIIA
    # component. Omitting it keeps every checkpoint hundreds of MB smaller.
    return {
        name: value
        for name, value in model.state_dict().items()
        if not name.startswith("visual_tokenizer.")
    }


def _torch_load(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    global_step: int,
    best_metric: float,
    config: dict,
    manifest_hash: str,
    batch_in_epoch: int = 0,
    epoch_complete: bool = True,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "model": _checkpoint_model_state(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "batch_in_epoch": batch_in_epoch,
        "epoch_complete": epoch_complete,
        "best_metric": best_metric,
        "config": config,
        "manifest_hash": manifest_hash,
        "rng_state": capture_rng_state(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    payload = _torch_load(path)
    incompatible = model.load_state_dict(payload["model"], strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = [name for name in incompatible.missing_keys if not name.startswith("visual_tokenizer.")]
    if unexpected or missing:
        raise RuntimeError(
            "Checkpoint/model mismatch. "
            f"Missing keys: {missing}; unexpected keys: {unexpected}"
        )
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if restore_rng and payload.get("rng_state") is not None:
        restore_rng_state(payload["rng_state"])
    return payload


def load_online_backbone_checkpoint(path: str | Path, backbone: torch.nn.Module) -> dict[str, Any]:
    """Load only the learned online CLIP state for retrieval inference."""
    payload = _torch_load(path)
    state = {
        name.removeprefix("backbone."): value
        for name, value in payload["model"].items()
        if name.startswith("backbone.")
    }
    if not state:
        raise RuntimeError("Checkpoint does not contain an online MIIA backbone")
    backbone.load_state_dict(state, strict=True)
    return payload
