from __future__ import annotations

import math

import attr
import torch
from torch import nn
from torch.nn import functional as F


LOGIT_LAPLACE_EPS = 0.1


@attr.s(eq=False)
class Conv2d(nn.Module):
    n_in: int = attr.ib()
    n_out: int = attr.ib()
    kw: int = attr.ib()
    use_float16: bool = attr.ib(default=True)
    device: torch.device = attr.ib(default=torch.device("cpu"))
    requires_grad: bool = attr.ib(default=False)

    def __attrs_post_init__(self) -> None:
        super().__init__()
        weight = torch.empty(
            (self.n_out, self.n_in, self.kw, self.kw),
            dtype=torch.float32,
            device=self.device,
            requires_grad=self.requires_grad,
        )
        weight.normal_(std=1 / math.sqrt(self.n_in * self.kw**2))
        bias = torch.zeros((self.n_out,), dtype=torch.float32, device=self.device, requires_grad=self.requires_grad)
        self.w = nn.Parameter(weight)
        self.b = nn.Parameter(bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.use_float16 and self.w.device.type == "cuda":
            value, weight, bias = value.half(), self.w.half(), self.b.half()
        else:
            value, weight, bias = value.float(), self.w, self.b
        return F.conv2d(value, weight, bias, padding=(self.kw - 1) // 2)


def map_pixels(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 4 or value.dtype != torch.float32:
        raise ValueError("dVAE input must be a 4D float32 tensor")
    return (1 - 2 * LOGIT_LAPLACE_EPS) * value + LOGIT_LAPLACE_EPS

