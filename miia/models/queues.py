from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PairedFeatureQueue(nn.Module):
    def __init__(self, size: int, dim: int) -> None:
        super().__init__()
        if size <= 0:
            raise ValueError("Queue size must be positive")
        image = F.normalize(torch.randn(size, dim), dim=-1)
        text = F.normalize(torch.randn(size, dim), dim=-1)
        self.register_buffer("image", image)
        self.register_buffer("text", text)
        self.register_buffer("pointer", torch.zeros((), dtype=torch.long))

    @property
    def size(self) -> int:
        return self.image.shape[0]

    @torch.no_grad()
    def enqueue(self, image: torch.Tensor, text: torch.Tensor) -> None:
        image = F.normalize(image.detach(), dim=-1)
        text = F.normalize(text.detach(), dim=-1)
        if image.shape != text.shape:
            raise ValueError("Paired queues require matching image/text feature shapes")
        if image.shape[0] >= self.size:
            image, text = image[-self.size :], text[-self.size :]
        count = image.shape[0]
        start = int(self.pointer.item())
        first = min(count, self.size - start)
        self.image[start : start + first] = image[:first]
        self.text[start : start + first] = text[:first]
        remaining = count - first
        if remaining:
            self.image[:remaining] = image[first:]
            self.text[:remaining] = text[first:]
        self.pointer.fill_((start + count) % self.size)

