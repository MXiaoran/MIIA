"""Minimal OpenAI DALL-E dVAE encoder loader.

The encoder implementation is adapted from https://github.com/openai/DALL-E
under the MIT license. See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .encoder import Encoder
from .utils import map_pixels


def load_model(path: str | Path, device: torch.device | str | None = None) -> nn.Module:
    with Path(path).open("rb") as stream:
        try:
            return torch.load(stream, map_location=device, weights_only=False)
        except TypeError:
            return torch.load(stream, map_location=device)


__all__ = ["Encoder", "load_model", "map_pixels"]

