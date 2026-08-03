from __future__ import annotations

from pathlib import Path

import requests
import torch
from torch import nn
from torch.nn import functional as F


DALL_E_ENCODER_URL = "https://cdn.openai.com/dall-e/encoder.pkl"


def download_dvae_encoder(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with requests.get(DALL_E_ENCODER_URL, stream=True, timeout=60, headers={"User-Agent": "miia-repro/0.1"}) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    stream.write(chunk)
    temporary.replace(path)


class DalleVisualTokenizer(nn.Module):
    def __init__(self, path: str | Path, auto_download: bool = True) -> None:
        super().__init__()
        path = Path(path)
        if not path.exists() and auto_download:
            download_dvae_encoder(path)
        if not path.exists():
            raise FileNotFoundError(f"DALL-E dVAE encoder not found: {path}")
        from dall_e import load_model

        self.encoder = load_model(path, device="cpu").eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def forward(self, image: torch.Tensor, target_grid: tuple[int, int]) -> torch.Tensor:
        from dall_e import map_pixels

        logits = self.encoder(map_pixels(image.float()))
        if logits.shape[-2:] != target_grid:
            logits = F.interpolate(logits.float(), size=target_grid, mode="bilinear", align_corners=False)
        return logits.argmax(dim=1)


class DeterministicVisualTokenizer(nn.Module):
    """Dependency-free smoke-test target generator; never use for the paper run."""

    def __init__(self, vocab_size: int = 8192) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    @torch.no_grad()
    def forward(self, image: torch.Tensor, target_grid: tuple[int, int]) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(image.float(), target_grid)
        red = (pooled[:, 0].clamp(0, 1) * 31).long()
        green = (pooled[:, 1].clamp(0, 1) * 31).long()
        blue = (pooled[:, 2].clamp(0, 1) * 7).long()
        return ((red * 32 + green) * 8 + blue) % self.vocab_size


def create_visual_tokenizer(config: dict, project_root: Path) -> nn.Module:
    backend = config.get("backend", "dall_e")
    if backend == "deterministic":
        return DeterministicVisualTokenizer()
    path = Path(config.get("encoder_path", "data/cache/dvae/encoder.pkl"))
    if not path.is_absolute():
        path = project_root / path
    try:
        return DalleVisualTokenizer(path, auto_download=bool(config.get("auto_download", True)))
    except Exception:
        if config.get("allow_deterministic_fallback", False):
            return DeterministicVisualTokenizer()
        raise
