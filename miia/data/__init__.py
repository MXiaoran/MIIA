"""Dataset preparation utilities.

Training transforms live in :mod:`miia.data.dataset`; keeping this package
initializer lightweight lets ``python -m miia.data prepare`` run before the
GPU/torchvision environment is installed.
"""

from .manifest import DatasetRecord, load_manifest

__all__ = ["DatasetRecord", "load_manifest"]
