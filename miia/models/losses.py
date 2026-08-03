from __future__ import annotations

import torch
from torch.nn import functional as F


def paired_logits(
    image_query: torch.Tensor,
    text_query: torch.Tensor,
    image_key: torch.Tensor,
    text_key: torch.Tensor,
    image_queue: torch.Tensor,
    text_queue: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    positive_i2t = (image_query * text_key).sum(dim=-1, keepdim=True)
    positive_t2i = (text_query * image_key).sum(dim=-1, keepdim=True)
    negative_i2t = image_query @ text_queue.T
    negative_t2i = text_query @ image_queue.T
    return (
        torch.cat([positive_i2t, negative_i2t], dim=1) / temperature,
        torch.cat([positive_t2i, negative_t2i], dim=1) / temperature,
    )


def dcl_loss(logits_i2t: torch.Tensor, logits_t2i: torch.Tensor) -> torch.Tensor:
    labels = torch.zeros(logits_i2t.shape[0], dtype=torch.long, device=logits_i2t.device)
    return 0.5 * (F.cross_entropy(logits_i2t.float(), labels) + F.cross_entropy(logits_t2i.float(), labels))


def bdm_loss(logits_i2t: torch.Tensor, logits_t2i: torch.Tensor) -> torch.Tensor:
    log_i2t = F.log_softmax(logits_i2t.float(), dim=-1)
    log_t2i = F.log_softmax(logits_t2i.float(), dim=-1)
    probability_i2t = log_i2t.exp()
    probability_t2i = log_t2i.exp()
    forward = F.kl_div(log_i2t, probability_t2i.detach(), reduction="batchmean")
    reverse = F.kl_div(log_t2i, probability_i2t.detach(), reduction="batchmean")
    return 0.5 * (forward + reverse)
