import numpy as np
import torch

from miia.metrics import retrieval_metrics


def test_mult_positive_retrieval_metrics_are_perfect():
    images = np.eye(3, dtype=np.float32)
    texts = np.repeat(np.eye(3, dtype=np.float32), 5, axis=0)
    mapping = np.repeat(np.arange(3), 5)
    metrics = retrieval_metrics(images, texts, mapping)
    assert metrics.mean_recall == 100.0
    assert metrics.image_to_text["R@1"] == 100.0
    assert metrics.text_to_image["R@1"] == 100.0


def test_torch_metrics_match_numpy_metrics():
    images = np.eye(3, dtype=np.float32)
    texts = np.repeat(images, 5, axis=0)
    mapping = np.repeat(np.arange(3), 5)
    numpy_metrics = retrieval_metrics(images, texts, mapping)
    torch_metrics = retrieval_metrics(
        torch.eye(3),
        torch.eye(3).repeat_interleave(5, dim=0),
        torch.arange(3).repeat_interleave(5),
    )
    assert torch_metrics == numpy_metrics
