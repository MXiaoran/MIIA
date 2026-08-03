from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class RetrievalMetrics:
    image_to_text: dict[str, float]
    text_to_image: dict[str, float]
    mean_recall: float

    def to_dict(self) -> dict:
        return {
            "image_to_text": self.image_to_text,
            "text_to_image": self.text_to_image,
            "mR": self.mean_recall,
            "paper_column_mapping": {
                "text_retrieval": "image_to_text",
                "image_retrieval": "text_to_image",
            },
        }


def _recall_at_k(ranks: np.ndarray, k: int) -> float:
    return float(100.0 * np.mean(ranks < k))


def _retrieval_metrics_numpy(
    image_features: np.ndarray,
    text_features: np.ndarray,
    text_to_image: np.ndarray,
    ks: Iterable[int] = (1, 5, 10),
) -> RetrievalMetrics:
    if image_features.ndim != 2 or text_features.ndim != 2:
        raise ValueError("Image and text features must be 2D matrices")
    if text_features.shape[0] != len(text_to_image):
        raise ValueError("text_to_image mapping must have one entry per caption")
    similarity = image_features @ text_features.T
    image_ranks = np.empty(image_features.shape[0], dtype=np.int64)
    for image_index in range(image_features.shape[0]):
        order = np.argsort(-similarity[image_index])
        positives = np.flatnonzero(text_to_image == image_index)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        image_ranks[image_index] = inverse[positives].min()

    text_ranks = np.empty(text_features.shape[0], dtype=np.int64)
    image_order = np.argsort(-similarity, axis=0)
    for text_index, positive_image in enumerate(text_to_image):
        text_ranks[text_index] = int(np.flatnonzero(image_order[:, text_index] == positive_image)[0])

    i2t = {f"R@{k}": _recall_at_k(image_ranks, k) for k in ks}
    t2i = {f"R@{k}": _recall_at_k(text_ranks, k) for k in ks}
    mean_recall = float(np.mean([*i2t.values(), *t2i.values()]))
    return RetrievalMetrics(i2t, t2i, mean_recall)


def _retrieval_metrics_torch(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    text_to_image: torch.Tensor,
    ks: Iterable[int],
) -> RetrievalMetrics:
    if image_features.ndim != 2 or text_features.ndim != 2:
        raise ValueError("Image and text features must be 2D matrices")
    mapping = text_to_image.to(device=image_features.device, dtype=torch.long)
    text_features = text_features.to(image_features.device)
    if text_features.shape[0] != mapping.numel():
        raise ValueError("text_to_image mapping must have one entry per caption")
    similarity = image_features @ text_features.T
    image_ranks = torch.empty(image_features.shape[0], dtype=torch.long, device=similarity.device)
    for image_index in range(image_features.shape[0]):
        order = torch.argsort(similarity[image_index], descending=True)
        positives = torch.nonzero(mapping == image_index, as_tuple=False).flatten()
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=order.device)
        image_ranks[image_index] = inverse[positives].min()

    image_order = torch.argsort(similarity, dim=0, descending=True)
    text_ranks = torch.empty(text_features.shape[0], dtype=torch.long, device=similarity.device)
    for text_index, positive_image in enumerate(mapping.tolist()):
        text_ranks[text_index] = torch.nonzero(
            image_order[:, text_index] == positive_image,
            as_tuple=False,
        )[0, 0]

    def recall(ranks: torch.Tensor, k: int) -> float:
        return float((ranks < k).float().mean().item() * 100.0)

    i2t = {f"R@{k}": recall(image_ranks, k) for k in ks}
    t2i = {f"R@{k}": recall(text_ranks, k) for k in ks}
    mean_recall = float(sum([*i2t.values(), *t2i.values()]) / (len(i2t) + len(t2i)))
    return RetrievalMetrics(i2t, t2i, mean_recall)


def retrieval_metrics(
    image_features: np.ndarray | torch.Tensor,
    text_features: np.ndarray | torch.Tensor,
    text_to_image: np.ndarray | torch.Tensor,
    ks: Iterable[int] = (1, 5, 10),
) -> RetrievalMetrics:
    """Compute multi-positive recall using the caller's native array backend."""
    ks = tuple(ks)
    if isinstance(image_features, torch.Tensor):
        if not isinstance(text_features, torch.Tensor):
            raise TypeError("image_features and text_features must use the same backend")
        mapping = text_to_image if isinstance(text_to_image, torch.Tensor) else torch.tensor(text_to_image)
        return _retrieval_metrics_torch(image_features, text_features, mapping, ks)
    return _retrieval_metrics_numpy(
        np.asarray(image_features),
        np.asarray(text_features),
        np.asarray(text_to_image),
        ks,
    )
