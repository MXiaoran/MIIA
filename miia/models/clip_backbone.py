from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def interpolate_visual_positional_embedding(positional: torch.Tensor, grid_size: tuple[int, int]) -> torch.Tensor:
    """Bicubically resize CLIP's learned patch positions while keeping CLS fixed."""
    target_height, target_width = grid_size
    patch_positions = positional[1:]
    source = int(math.sqrt(patch_positions.shape[0]))
    if source * source != patch_positions.shape[0]:
        raise ValueError("CLIP visual positional embedding is not a square grid")
    if (source, source) == grid_size:
        return positional
    patch_positions = patch_positions.reshape(1, source, source, -1).permute(0, 3, 1, 2)
    patch_positions = F.interpolate(patch_positions.float(), size=grid_size, mode="bicubic", align_corners=False)
    patch_positions = patch_positions.permute(0, 2, 3, 1).reshape(target_height * target_width, -1)
    return torch.cat([positional[:1].float(), patch_positions], dim=0).to(positional.dtype)


class ClipBackbone(nn.Module):
    """OpenCLIP wrapper exposing local tokens and learnable patch/text mask embeddings."""

    def __init__(self, clip_model: nn.Module, image_size: int = 256) -> None:
        super().__init__()
        # Register the actual CLIP components once. Keeping both the parent
        # model and its children would duplicate state_dict key paths.
        self.visual = clip_model.visual
        self.token_embedding = clip_model.token_embedding
        self.text_transformer = clip_model.transformer
        self.text_positional_embedding = clip_model.positional_embedding
        self.text_ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.register_buffer("text_attn_mask", clip_model.attn_mask, persistent=False)
        self.image_size = image_size
        visual_width = int(self.visual.conv1.out_channels)
        text_width = int(self.token_embedding.embedding_dim)
        self.image_mask_embedding = nn.Parameter(torch.zeros(visual_width))
        self.text_mask_embedding = nn.Parameter(torch.zeros(text_width))
        nn.init.normal_(self.image_mask_embedding, std=visual_width**-0.5)
        nn.init.normal_(self.text_mask_embedding, std=0.02)

    @property
    def image_width(self) -> int:
        return int(self.visual.conv1.out_channels)

    @property
    def text_width(self) -> int:
        return int(self.token_embedding.embedding_dim)

    @property
    def embed_dim(self) -> int:
        projection = self.text_projection
        return int(projection.shape[-1] if isinstance(projection, torch.Tensor) else projection.out_features)

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        if hasattr(self.visual, "set_grad_checkpointing"):
            self.visual.set_grad_checkpointing(enabled)
        elif hasattr(self.visual, "transformer") and hasattr(self.visual.transformer, "grad_checkpointing"):
            self.visual.transformer.grad_checkpointing = enabled
        if hasattr(self.text_transformer, "grad_checkpointing"):
            self.text_transformer.grad_checkpointing = enabled

    def encode_image(self, image: torch.Tensor, patch_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        visual = self.visual
        value = visual.conv1(image).reshape(image.shape[0], self.image_width, -1).permute(0, 2, 1)
        if patch_mask is not None:
            if patch_mask.shape != value.shape[:2]:
                raise ValueError(f"Patch mask {patch_mask.shape} does not match tokens {value.shape[:2]}")
            value = torch.where(patch_mask.unsqueeze(-1), self.image_mask_embedding.to(value.dtype), value)
        class_token = visual.class_embedding.to(value.dtype).reshape(1, 1, -1).expand(value.shape[0], -1, -1)
        value = torch.cat([class_token, value], dim=1)
        patch_height = image.shape[-2] // visual.conv1.kernel_size[0]
        patch_width = image.shape[-1] // visual.conv1.kernel_size[1]
        positions = interpolate_visual_positional_embedding(
            visual.positional_embedding,
            (patch_height, patch_width),
        )
        value = value + positions.to(value.dtype)
        value = visual.patch_dropout(value)
        value = visual.ln_pre(value).permute(1, 0, 2)
        value = visual.transformer(value).permute(1, 0, 2)
        value = visual.ln_post(value)
        pooled = value[:, 0]
        if visual.proj is not None:
            pooled = pooled @ visual.proj
        return F.normalize(pooled, dim=-1), value[:, 1:]

    def encode_text(self, token_ids: torch.Tensor, token_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.token_embedding(token_ids)
        if token_mask is not None:
            if token_mask.shape != token_ids.shape:
                raise ValueError("Text mask must match token id shape")
            value = torch.where(token_mask.unsqueeze(-1), self.text_mask_embedding.to(value.dtype), value)
        value = value + self.text_positional_embedding[: value.shape[1]].to(value.dtype)
        value = value.permute(1, 0, 2)
        try:
            value = self.text_transformer(value, attn_mask=self.text_attn_mask)
        except TypeError:
            value = self.text_transformer(value)
        value = self.text_ln_final(value.permute(1, 0, 2))
        pooled = value[torch.arange(value.shape[0], device=value.device), token_ids.argmax(dim=-1)]
        projection = self.text_projection
        pooled = projection(pooled) if isinstance(projection, nn.Linear) else pooled @ projection
        return F.normalize(pooled, dim=-1), value


def create_clip_backbone(model_name: str, pretrained: str | None, image_size: int = 256) -> tuple[ClipBackbone, Any]:
    try:
        import open_clip
    except ImportError as exc:
        raise RuntimeError("Install open_clip_torch==2.24.0 to construct MIIA") from exc
    # Keep the original 224-position table; encode_image interpolates it to the
    # requested grid explicitly.
    model = open_clip.create_model(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    return ClipBackbone(model, image_size=image_size), tokenizer
