from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def tensor_summary(state: dict[str, Any]) -> dict[str, Any]:
    tensor_items = {key: value for key, value in state.items() if torch.is_tensor(value)}
    prefixes: dict[str, int] = {}
    for key in tensor_items:
        prefix = key.split(".", 1)[0]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
    visual_conv = tensor_items.get("clip_model.visual.conv1.weight")
    return {
        "tensor_count": len(tensor_items),
        "parameter_count": sum(value.numel() for value in tensor_items.values()),
        "prefixes": prefixes,
        "visual_patch_size": list(visual_conv.shape[-2:]) if visual_conv is not None else None,
        "has_mii": any("mii" in key.lower() or "cmii" in key.lower() or "cmli" in key.lower() for key in tensor_items),
        "has_momentum_encoder": any("momentum" in key.lower() or "encoder_m" in key.lower() for key in tensor_items),
        "has_feature_queue": any("queue" in key.lower() for key in tensor_items),
        "non_clip_keys": [key for key in tensor_items if not key.startswith("clip_model.")],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely inspect a legacy CUSA/CLIP checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    try:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(args.checkpoint, map_location="cpu")
    state = payload.get("model", payload)
    summary = tensor_summary(state)
    summary.update({
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "epoch": payload.get("epoch") if isinstance(payload, dict) else None,
        "best": payload.get("best") if isinstance(payload, dict) else None,
        "config": payload.get("config") if isinstance(payload, dict) else None,
        "classification": "CUSA/CLIP baseline, not a MIIA checkpoint" if not summary["has_mii"] else "contains MII-like keys",
    })
    text = json.dumps(summary, ensure_ascii=False, indent=2, default=str)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
