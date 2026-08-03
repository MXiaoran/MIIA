from __future__ import annotations

import importlib
import json
import platform
import sys
from pathlib import Path


def main() -> int:
    modules = ["torch", "torchvision", "open_clip", "yaml", "PIL", "imagehash"]
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "modules": {},
    }
    failed = False
    for name in modules:
        try:
            module = importlib.import_module(name)
            report["modules"][name] = getattr(module, "__version__", "installed")
        except Exception as exc:
            report["modules"][name] = f"ERROR: {exc}"
            failed = True
    try:
        import torch

        report["cuda_available"] = torch.cuda.is_available()
        report["torch_cuda"] = torch.version.cuda
        report["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        if not torch.cuda.is_available():
            failed = True
        if torch.cuda.is_available():
            total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
            report["gpu_memory_gib"] = round(total_gib, 2)
            report["gpu_memory_suitable_for_default_micro_batch"] = total_gib >= 15.0
            if total_gib < 15.0:
                failed = True
    except Exception:
        pass
    try:
        import numpy as np

        numpy_major = int(np.__version__.split(".")[0])
        report["numpy"] = np.__version__
        if numpy_major >= 2:
            report["numpy_warning"] = "Install numpy<2 as pinned by this project"
            failed = True
    except Exception as exc:
        report["numpy"] = f"ERROR: {exc}"
        failed = True
    project_root = Path(__file__).resolve().parents[1]
    manifest = project_root / "data" / "processed" / "ret3_manifest.json"
    report["ret3_manifest"] = str(manifest) if manifest.exists() else "missing (run python -m miia.data prepare)"
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
