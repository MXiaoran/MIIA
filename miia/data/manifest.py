from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from miia.utils import atomic_json_dump, sha256_file, stable_hash

try:
    import imagehash
except ImportError:  # allows lightweight manifest unit tests without ImageHash
    imagehash = None


def perceptual_hash(image: Image.Image) -> str:
    if imagehash is not None:
        return str(imagehash.phash(image.convert("RGB")))
    # Lightweight pHash-compatible fallback: DCT over a 32x32 grayscale image.
    import numpy as np

    values = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    size = values.shape[0]
    positions = np.arange(size, dtype=np.float32)
    frequencies = positions.reshape(-1, 1)
    transform = np.cos((2 * positions + 1) * frequencies * np.pi / (2 * size))
    coefficients = transform @ values @ transform.T
    low = coefficients[:8, :8]
    median = np.median(low.flatten()[1:])
    bits = low > median
    return f"{int(''.join('1' if value else '0' for value in bits.flatten()), 2):016x}"


EXPECTED_COUNTS = {"rsicd": 10921, "rsitmd": 4743, "ucm": 2100}
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validate": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
    "restval": "train",
}


@dataclass
class DatasetRecord:
    dataset: str
    image_id: str
    image_path: str
    captions: list[str]
    split: str
    sha256: str = ""
    phash: str = ""
    sources: list[str] = field(default_factory=list)
    excluded_from_train: bool = False
    exclusion_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DatasetRecord":
        fields = {item.name for item in dataclasses.fields(cls)}
        return cls(**{key: value for key, value in value.items() if key in fields})


def _normalise_split(value: Any) -> str:
    key = str(value or "train").strip().lower()
    if key not in SPLIT_ALIASES:
        raise ValueError(f"Unsupported dataset split: {value!r}")
    return SPLIT_ALIASES[key]


def _extract_captions(entry: dict[str, Any]) -> list[str]:
    raw = entry.get("captions") or entry.get("sentences") or entry.get("caption") or entry.get("texts")
    if isinstance(raw, str):
        candidates = [raw]
    elif isinstance(raw, list):
        candidates = []
        for item in raw:
            if isinstance(item, str):
                candidates.append(item)
            elif isinstance(item, dict):
                text = item.get("raw") or item.get("caption") or item.get("text")
                if text is None and isinstance(item.get("tokens"), list):
                    text = " ".join(str(token) for token in item["tokens"])
                if text:
                    candidates.append(str(text))
    else:
        candidates = []
    captions: list[str] = []
    for caption in candidates:
        caption = " ".join(caption.strip().split())
        if caption:
            captions.append(caption)
    return captions


def _entries_from_json(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, dict)]
    if isinstance(payload, dict):
        for key in ("images", "data", "annotations", "records"):
            value = payload.get(key)
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return value
        # filename -> captions mapping
        entries: list[dict[str, Any]] = []
        for key, value in payload.items():
            if isinstance(value, (list, str)):
                entries.append({"filename": key, "captions": value})
        if entries:
            return entries
    raise ValueError("Unsupported annotation JSON structure")


def _build_image_index(images_root: Path) -> tuple[dict[str, Path], list[Path]]:
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    files = [path for path in images_root.rglob("*") if path.is_file() and path.suffix.lower() in extensions]
    index: dict[str, Path] = {}
    for path in files:
        index.setdefault(path.name.lower(), path)
        index.setdefault(path.stem.lower(), path)
    return index, files


def _resolve_image(entry: dict[str, Any], dataset: str, index: dict[str, Path], files: list[Path]) -> Path:
    name = (
        entry.get("filename")
        or entry.get("file_name")
        or entry.get("image")
        or entry.get("image_path")
        or entry.get("img_path")
    )
    if name is None:
        image_id = entry.get("imgid") or entry.get("image_id") or entry.get("id")
        if image_id is not None:
            name = str(image_id)
    if name is None:
        raise KeyError(f"Annotation does not identify an image: {entry}")
    name = Path(str(name)).name
    candidates = [name.lower(), Path(name).stem.lower()]
    prefixes = (f"{dataset}_", "train_", "val_", "test_")
    for candidate in list(candidates):
        for prefix in prefixes:
            candidates.append((prefix + candidate).lower())
    for candidate in candidates:
        if candidate in index:
            return index[candidate]
    stem = Path(name).stem.lower()
    suffix_matches = [path for path in files if path.stem.lower().endswith(stem)]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    raise FileNotFoundError(f"Could not resolve image {name!r} below the provided image root")


def parse_dataset(
    dataset: str,
    annotation_path: str | Path,
    images_root: str | Path,
    compute_hashes: bool = True,
) -> list[DatasetRecord]:
    annotation_path = Path(annotation_path).resolve()
    images_root = Path(images_root).resolve()
    with annotation_path.open("r", encoding="utf-8-sig") as stream:
        entries = _entries_from_json(json.load(stream))
    index, image_files = _build_image_index(images_root)
    records: list[DatasetRecord] = []
    for position, entry in enumerate(entries):
        captions = _extract_captions(entry)
        if not captions:
            raise ValueError(f"No captions for {dataset} annotation entry {position}")
        image_path = _resolve_image(entry, dataset, index, image_files)
        image_id = str(
            entry.get("imgid")
            if entry.get("imgid") is not None
            else entry.get("image_id")
            if entry.get("image_id") is not None
            else entry.get("id")
            if entry.get("id") is not None
            else image_path.stem
        )
        record = DatasetRecord(
            dataset=dataset,
            image_id=image_id,
            image_path=str(image_path),
            captions=captions[:5],
            split=_normalise_split(entry.get("split", "train")),
            sources=[f"{dataset}:{image_id}"],
        )
        if compute_hashes:
            record.sha256 = sha256_file(image_path)
            with Image.open(image_path) as image:
                record.phash = perceptual_hash(image)
        records.append(record)
    return records


def validate_records(records: list[DatasetRecord], strict_counts: bool = True) -> dict[str, Any]:
    by_dataset: dict[str, list[DatasetRecord]] = {}
    for record in records:
        by_dataset.setdefault(record.dataset, []).append(record)
        if len(record.captions) != 5:
            raise ValueError(
                f"{record.dataset}:{record.image_id} has {len(record.captions)} unique captions; expected 5"
            )
        if not Path(record.image_path).is_file():
            raise FileNotFoundError(record.image_path)
    summary: dict[str, Any] = {}
    for dataset, items in sorted(by_dataset.items()):
        split_counts = {split: sum(item.split == split for item in items) for split in ("train", "val", "test")}
        summary[dataset] = {"images": len(items), "captions": sum(len(item.captions) for item in items), "splits": split_counts}
        expected = EXPECTED_COUNTS.get(dataset)
        if strict_counts and expected is not None and len(items) != expected:
            raise ValueError(f"{dataset} has {len(items)} images; expected {expected}")
    return summary


class _BKNode:
    def __init__(self, value: int, record: DatasetRecord):
        self.value = value
        self.records = [record]
        self.children: dict[int, _BKNode] = {}


class PhashIndex:
    """BK-tree over 64-bit perceptual hashes for audit-friendly near-duplicate lookup."""

    def __init__(self) -> None:
        self.root: _BKNode | None = None

    @staticmethod
    def distance(left: int, right: int) -> int:
        return (left ^ right).bit_count()

    def add(self, record: DatasetRecord) -> None:
        value = int(record.phash, 16)
        if self.root is None:
            self.root = _BKNode(value, record)
            return
        node = self.root
        while True:
            distance = self.distance(value, node.value)
            if distance == 0:
                node.records.append(record)
                return
            if distance not in node.children:
                node.children[distance] = _BKNode(value, record)
                return
            node = node.children[distance]

    def search(self, phash: str, threshold: int = 2) -> list[DatasetRecord]:
        if self.root is None:
            return []
        target = int(phash, 16)
        matches: list[DatasetRecord] = []
        pending = [self.root]
        while pending:
            node = pending.pop()
            distance = self.distance(target, node.value)
            if distance <= threshold:
                matches.extend(node.records)
            lower, upper = distance - threshold, distance + threshold
            pending.extend(child for edge, child in node.children.items() if lower <= edge <= upper)
        return matches


def build_ret3_train(records: list[DatasetRecord], threshold: int = 2) -> tuple[list[DatasetRecord], dict[str, Any]]:
    # The union of all official test splits has strict priority.  This must be
    # global rather than dataset-local: RSITMD reuses imagery from RSICD and
    # UCM, often re-encoded under a different filename, so a source train image
    # can otherwise reveal a target dataset's test image.
    held_out = [record for record in records if record.split == "test"]
    held_out_index = PhashIndex()
    for record in held_out:
        held_out_index.add(record)
    train_index = PhashIndex()
    kept: list[DatasetRecord] = []
    audit: list[dict[str, Any]] = []
    for record in sorted((item for item in records if item.split == "train"), key=lambda item: (item.dataset, item.image_id)):
        held_out_matches = held_out_index.search(record.phash, threshold)
        if held_out_matches:
            record.excluded_from_train = True
            record.exclusion_reason = "near_duplicate_of_held_out"
            audit.append({
                "record": record.sources,
                "reason": record.exclusion_reason,
                "matches": [item.sources for item in held_out_matches],
            })
            continue
        train_matches = [match for match in train_index.search(record.phash, threshold) if match.dataset != record.dataset]
        if train_matches:
            canonical = train_matches[0]
            record.excluded_from_train = True
            record.exclusion_reason = "near_duplicate_in_train"
            canonical.sources.extend(source for source in record.sources if source not in canonical.sources)
            audit.append({
                "record": record.sources,
                "reason": record.exclusion_reason,
                "matches": [canonical.sources],
            })
            continue
        kept.append(record)
        train_index.add(record)
    return kept, {
        "phash_threshold": threshold,
        "protected_test_records": len(held_out),
        "candidate_train_records": sum(record.split == "train" for record in records),
        "kept_train_records": len(kept),
        "excluded_records": sum(record.excluded_from_train for record in records if record.split == "train"),
        "audited_overlaps": len(audit),
        "exclusions": audit,
    }


def write_manifests(records: list[DatasetRecord], output_dir: str | Path, strict_counts: bool = True) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = validate_records(records, strict_counts=strict_counts)
    train_records, audit = build_ret3_train(records)
    payload = {
        "schema_version": 1,
        "summary": summary,
        "records": [record.to_dict() for record in records],
    }
    train_payload = {
        "schema_version": 1,
        "audit": audit,
        "records": [record.to_dict() for record in train_records],
    }
    payload["manifest_hash"] = stable_hash(payload["records"])
    train_payload["manifest_hash"] = stable_hash(train_payload["records"])
    atomic_json_dump(payload, output_dir / "ret3_manifest.json")
    atomic_json_dump(train_payload, output_dir / "ret3_train.json")
    atomic_json_dump(audit, output_dir / "ret3_dedup_audit.json")
    return {"summary": summary, "audit": audit, "manifest_hash": payload["manifest_hash"], "train_hash": train_payload["manifest_hash"]}


def load_manifest(path: str | Path) -> tuple[list[DatasetRecord], dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    records = [DatasetRecord.from_dict(item) for item in payload.get("records", payload)]
    metadata = {key: value for key, value in payload.items() if key != "records"} if isinstance(payload, dict) else {}
    return records, metadata
