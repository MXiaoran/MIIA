from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Callable

import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:  # torchvision 0.8 compatibility for legacy environment inspection
    from PIL import Image as PILImage

    BICUBIC = PILImage.BICUBIC

from miia.data.manifest import DatasetRecord


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    buffer = bytearray(image.tobytes())
    tensor = torch.frombuffer(buffer, dtype=torch.uint8).clone()
    tensor = tensor.view(image.height, image.width, len(image.getbands())).permute(2, 0, 1)
    return tensor.float().div_(255.0)


def build_train_transform(image_size: int = 256) -> Callable[[Image.Image], torch.Tensor]:
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=BICUBIC),
        transforms.RandomCrop(image_size, padding=10, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(pil_to_tensor),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.20), value="random"),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def build_eval_transform(image_size: int = 256) -> Callable[[Image.Image], torch.Tensor]:
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=BICUBIC),
        transforms.Lambda(pil_to_tensor),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])


class DvaeFromClipTransform:
    """Recover aligned RGB pixels from the already augmented CLIP tensor."""

    def __init__(self, target_size: int = 128) -> None:
        self.target_size = target_size

    def __call__(self, normalized_image: torch.Tensor) -> torch.Tensor:
        mean = normalized_image.new_tensor(CLIP_MEAN).view(3, 1, 1)
        std = normalized_image.new_tensor(CLIP_STD).view(3, 1, 1)
        rgb = (normalized_image * std + mean).clamp(0, 1)
        return F.interpolate(
            rgb.unsqueeze(0),
            size=(self.target_size, self.target_size),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0).clamp(0, 1)


def build_dvae_transform(target_size: int = 128) -> DvaeFromClipTransform:
    return DvaeFromClipTransform(target_size)


class RetrievalDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: list[DatasetRecord],
        tokenizer: Callable[[list[str]], torch.Tensor],
        image_transform: Callable[[Image.Image], torch.Tensor],
        dvae_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        training: bool = False,
        seed: int = 23,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.dvae_transform = dvae_transform
        self.training = training
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        if self.training:
            item_seed = self.seed + self.epoch * len(self.records) + index
            generator = random.Random(item_seed)
            caption = record.captions[generator.randrange(len(record.captions))]
        else:
            caption = record.captions[0]
        with Image.open(record.image_path) as opened:
            image = opened.convert("RGB")
            if self.training:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(item_seed)
                    image_tensor = self.image_transform(image)
            else:
                image_tensor = self.image_transform(image)
            dvae_tensor = self.dvae_transform(image_tensor) if self.dvae_transform else torch.empty(0)
        text = self.tokenizer([caption])[0]
        return {
            "image": image_tensor,
            "dvae_image": dvae_tensor,
            "text": text,
            "caption": caption,
            "captions": record.captions,
            "dataset": record.dataset,
            "image_id": record.image_id,
            "image_path": record.image_path,
        }


class HomogeneousLogicalBatchSampler(Sampler[list[int]]):
    """Yield dataset-homogeneous batches so each batch has one DCL queue."""

    def __init__(
        self,
        records: list[DatasetRecord],
        batch_size: int,
        seed: int = 23,
        start_offset: int = 0,
        drop_last: bool = True,
        shuffle: bool = True,
        logical_batch_size: int | None = None,
    ) -> None:
        self.groups: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            self.groups[record.dataset].append(index)
        self.batch_size = batch_size
        self.seed = seed
        self.start_offset = start_offset
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.logical_batch_size = logical_batch_size or batch_size
        if self.logical_batch_size % self.batch_size:
            raise ValueError("logical_batch_size must be divisible by sampler batch_size")
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = random.Random(self.seed + self.epoch)
        grouped_batches: dict[str, list[list[int]]] = {}
        for name, indices in self.groups.items():
            shuffled = list(indices)
            if self.shuffle:
                generator.shuffle(shuffled)
            if self.drop_last:
                usable = (len(shuffled) // self.logical_batch_size) * self.logical_batch_size
                shuffled = shuffled[:usable]
            batches: list[list[int]] = []
            for start in range(0, len(shuffled), self.batch_size):
                batch = shuffled[start:start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
            grouped_batches[name] = batches
        if self.shuffle:
            names = list(grouped_batches)
            generator.shuffle(names)
        else:
            names = list(grouped_batches)
        if self.start_offset:
            flattened = [batch for name in names for batch in grouped_batches[name]]
            yield from flattened[self.start_offset:]
            return
        # Keep runs within one dataset long enough to form complete logical
        # batches while rotating their order between epochs.
        for name in names:
            yield from grouped_batches[name]

    def __len__(self) -> int:
        if self.drop_last:
            total = sum(
                ((len(indices) // self.logical_batch_size) * self.logical_batch_size) // self.batch_size
                for indices in self.groups.values()
            )
        else:
            total = sum((len(indices) + self.batch_size - 1) // self.batch_size for indices in self.groups.values())
        return max(0, total - self.start_offset)


def build_dataloader(
    dataset: RetrievalDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    training: bool,
    seed: int = 23,
    homogeneous: bool = False,
    shuffle: bool | None = None,
    logical_batch_size: int | None = None,
    start_offset: int = 0,
) -> DataLoader:
    worker_generator = torch.Generator()
    worker_generator.manual_seed(seed)
    if training or homogeneous:
        sampler = HomogeneousLogicalBatchSampler(
            dataset.records,
            batch_size=batch_size,
            seed=seed,
            drop_last=training,
            shuffle=training if shuffle is None else shuffle,
            logical_batch_size=logical_batch_size,
            start_offset=start_offset,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=worker_generator,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False if shuffle is None else shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=worker_generator,
    )
