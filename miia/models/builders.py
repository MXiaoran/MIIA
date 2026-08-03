from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from .clip_backbone import ClipBackbone
from .miia import MIIA


class TinyResidualBlock(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))

    def forward(self, value: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        normal = self.ln_1(value)
        mask = attn_mask.to(value.device, value.dtype) if attn_mask is not None else None
        attended, _ = self.attn(normal, normal, normal, attn_mask=mask, need_weights=False)
        value = value + attended
        return value + self.mlp(self.ln_2(value))


class TinyTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int) -> None:
        super().__init__()
        self.resblocks = nn.ModuleList([TinyResidualBlock(width, heads) for _ in range(layers)])

    def forward(self, value: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.resblocks:
            value = block(value, attn_mask=attn_mask)
        return value


class TinyVisual(nn.Module):
    def __init__(self, image_size: int, patch_size: int, width: int, embed_dim: int, layers: int, heads: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, patch_size, stride=patch_size, bias=False)
        grid = image_size // patch_size
        self.class_embedding = nn.Parameter(torch.randn(width) / math.sqrt(width))
        self.positional_embedding = nn.Parameter(torch.randn(grid * grid + 1, width) / math.sqrt(width))
        self.patch_dropout = nn.Identity()
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = TinyTransformer(width, layers, heads)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(torch.randn(width, embed_dim) / math.sqrt(width))


class TinyClip(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 8,
        width: int = 32,
        embed_dim: int = 16,
        context_length: int = 12,
        vocab_size: int = 128,
    ) -> None:
        super().__init__()
        self.visual = TinyVisual(image_size, patch_size, width, embed_dim, layers=1, heads=4)
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.transformer = TinyTransformer(width, layers=1, heads=4)
        self.positional_embedding = nn.Parameter(torch.randn(context_length, width) * 0.01)
        self.ln_final = nn.LayerNorm(width)
        self.text_projection = nn.Parameter(torch.randn(width, embed_dim) / math.sqrt(width))
        mask = torch.empty(context_length, context_length).fill_(float("-inf")).triu_(1)
        self.register_buffer("attn_mask", mask, persistent=False)


class TinyTokenizer:
    def __init__(self, context_length: int = 12, vocab_size: int = 128) -> None:
        self.context_length = context_length
        self.vocab_size = vocab_size

    def __call__(self, texts: list[str]) -> torch.Tensor:
        output = torch.zeros(len(texts), self.context_length, dtype=torch.long)
        output[:, 0] = self.vocab_size - 2
        for row, text in enumerate(texts):
            words = text.lower().split()[: self.context_length - 2]
            for column, word in enumerate(words, start=1):
                output[row, column] = 1 + sum(word.encode("utf-8")) % (self.vocab_size - 3)
            output[row, len(words) + 1] = self.vocab_size - 1
        return output


def create_tiny_miia(image_size: int = 32) -> tuple[MIIA, TinyTokenizer]:
    clip = TinyClip(image_size=image_size)
    backbone = ClipBackbone(clip, image_size=image_size)
    config = {
        "image_mask_ratio": 0.4,
        "text_mask_ratio": 0.15,
        "mii": {
            "enabled": True,
            "hidden_dim": 32,
            "heads": 4,
            "layers": 1,
            "visual_vocab_size": 8192,
            "text_vocab_size": 128,
            "gradient_checkpointing": False,
        },
        "dvae": {"backend": "deterministic", "allow_deterministic_fallback": True},
        "dcl": {
            "enabled": True,
            "momentum": 0.99,
            "temperature": 0.02,
            "queue_sizes": {"rsicd": 8, "rsitmd": 8, "ucm": 8},
        },
        "bdm": {"enabled": True},
        "loss_weights": {"mii": 1.0, "dcl": 1.0, "bdm": 1.0},
    }
    return MIIA(backbone, config, project_root=Path(".")), TinyTokenizer()
