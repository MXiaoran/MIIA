"""Build a fallback RET-3 manifest from RemoteCLIP's official release.

RemoteCLIP publishes the de-duplicated RET-3 train archive and the three
benchmark test splits, but omits the original RSITMD/UCM validation archives.
This converter combines those files with the complete official RSICD manifest.
It intentionally does not manufacture validation examples from training data:
model selection therefore uses the available official RSICD validation split,
while all three final evaluations use RemoteCLIP's untouched test splits.

Prefer ``python -m miia.data prepare`` with the complete original datasets.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from miia.data.manifest import DatasetRecord, load_manifest, perceptual_hash, write_manifests
from miia.utils import sha256_file


TEST_CSVS = {
    "rsitmd": "rsitmd_test.csv",
    "ucm": "ucm_test.csv",
}


def read_caption_groups(path: Path, dataset: str) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            filename = Path(row["filename"]).name
            if not filename.lower().startswith(f"{dataset}_"):
                continue
            grouped[filename].append(" ".join(row["title"].strip().split()))
    invalid = {name: len(captions) for name, captions in grouped.items() if len(captions) != 5}
    if invalid:
        example = next(iter(invalid.items()))
        raise ValueError(f"{path}: expected five captions per image; found {example}")
    return dict(grouped)


def records_from_release(
    dataset: str,
    split: str,
    csv_path: Path,
    images_root: Path,
) -> list[DatasetRecord]:
    groups = read_caption_groups(csv_path, dataset)
    records: list[DatasetRecord] = []
    for filename, captions in tqdm(sorted(groups.items()), desc=f"hash {dataset}/{split}"):
        image_path = (images_root / filename).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image:
            phash = perceptual_hash(image)
        records.append(
            DatasetRecord(
                dataset=dataset,
                image_id=Path(filename).stem.removeprefix(f"{dataset}_"),
                image_path=str(image_path),
                captions=captions,
                split=split,
                sha256=sha256_file(image_path),
                phash=phash,
                sources=[f"remoteclip_ret3:{split}:{filename}"],
            )
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project_root.resolve()
    processed = root / "data" / "processed"
    release = root / "data" / "raw" / "remoteclip_ret3"
    csv_root = root / "data" / "cache" / "remoteclip_ret3" / "csv"

    rsicd_records, rsicd_metadata = load_manifest(processed / "ret3_manifest.json")
    rsicd_records = [record for record in rsicd_records if record.dataset == "rsicd"]
    if len(rsicd_records) != 10921:
        raise ValueError(
            "The current manifest must contain the complete 10,921-image RSICD set "
            f"before conversion; found {len(rsicd_records)}"
        )

    records = list(rsicd_records)
    train_csv = csv_root / "Ret-3_train.csv"
    for dataset in ("rsitmd", "ucm"):
        records.extend(records_from_release(dataset, "train", train_csv, release / "Ret-3_train"))
        records.extend(
            records_from_release(dataset, "test", csv_root / TEST_CSVS[dataset], release / "Ret-3_test")
        )

    result = write_manifests(records, processed, strict_counts=False)
    result["protocol"] = {
        "rsicd": "complete official train/val/test",
        "rsitmd": "RemoteCLIP de-duplicated train + untouched official test; validation unavailable",
        "ucm": "RemoteCLIP de-duplicated train + untouched official test; validation unavailable",
        "upstream_rsicd_manifest_hash": rsicd_metadata.get("manifest_hash", ""),
    }
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
