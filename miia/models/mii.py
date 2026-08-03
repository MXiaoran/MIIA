from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class CrossModalInferrer(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, layers: int, output_size: int, checkpointing: bool = False) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.ModuleList([layer if index == 0 else type(layer)(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        ) for index in range(layers)])
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_size)
        self.checkpointing = checkpointing

    def forward(self, query: torch.Tensor, context: torch.Tensor, context_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        cross, _ = self.cross_attention(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            key_padding_mask=context_padding_mask,
            need_weights=False,
        )
        value = self.cross_norm(query + cross)
        for layer in self.layers:
            if self.checkpointing and self.training:
                value = checkpoint(layer, value, use_reentrant=False)
            else:
                value = layer(value)
        return self.head(self.output_norm(value))


class MaskedInteractionInferring(nn.Module):
    def __init__(
        self,
        image_width: int,
        text_width: int,
        hidden_dim: int = 512,
        heads: int = 8,
        layers: int = 4,
        visual_vocab_size: int = 8192,
        text_vocab_size: int = 49408,
        checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.image_projection = nn.Linear(image_width, hidden_dim)
        self.text_projection = nn.Linear(text_width, hidden_dim)
        self.cmii = CrossModalInferrer(hidden_dim, heads, layers, visual_vocab_size, checkpointing)
        self.cmli = CrossModalInferrer(hidden_dim, heads, layers, text_vocab_size, checkpointing)

    def image_logits(self, masked_image_tokens: torch.Tensor, text_tokens: torch.Tensor, text_padding: torch.Tensor) -> torch.Tensor:
        return self.cmii(
            self.image_projection(masked_image_tokens),
            self.text_projection(text_tokens),
            context_padding_mask=text_padding,
        )

    def text_logits(self, masked_text_tokens: torch.Tensor, image_tokens: torch.Tensor) -> torch.Tensor:
        return self.cmli(self.text_projection(masked_text_tokens), self.image_projection(image_tokens))

    @staticmethod
    def masked_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not mask.any():
            return logits.sum() * 0
        return F.cross_entropy(logits[mask].float(), labels[mask])
