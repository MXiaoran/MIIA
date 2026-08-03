from __future__ import annotations

from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

from miia.data.dataset import build_eval_transform
from miia.data.manifest import DatasetRecord
from miia.metrics import retrieval_metrics


@torch.inference_mode()
def evaluate_records(
    model,
    tokenizer,
    records: list[DatasetRecord],
    device: torch.device,
    image_size: int,
    batch_size: int = 128,
) -> dict[str, Any]:
    model.eval()
    transform = build_eval_transform(image_size)
    image_features: list[torch.Tensor] = []
    for start in tqdm(range(0, len(records), batch_size), desc="images", leave=False):
        chunk = records[start : start + batch_size]
        tensors = []
        for record in chunk:
            with Image.open(record.image_path) as image:
                tensors.append(transform(image.convert("RGB")))
        batch = torch.stack(tensors).to(device, non_blocking=True)
        encoded = model.encode_image(batch)
        image_features.append((encoded[0] if isinstance(encoded, tuple) else encoded).cpu())

    captions: list[str] = []
    text_to_image: list[int] = []
    for image_index, record in enumerate(records):
        captions.extend(record.captions)
        text_to_image.extend([image_index] * len(record.captions))
    text_features: list[torch.Tensor] = []
    for start in tqdm(range(0, len(captions), batch_size), desc="texts", leave=False):
        tokens = tokenizer(captions[start : start + batch_size]).to(device)
        encoded = model.encode_text(tokens)
        text_features.append((encoded[0] if isinstance(encoded, tuple) else encoded).cpu())

    metrics = retrieval_metrics(
        torch.cat(image_features).float(),
        torch.cat(text_features).float(),
        torch.tensor(text_to_image, dtype=torch.long),
    )
    return metrics.to_dict()
