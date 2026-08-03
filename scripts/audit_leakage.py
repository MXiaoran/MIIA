"""Fail when a prepared RET-3 training manifest overlaps any test split."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from miia.data.manifest import PhashIndex, load_manifest


def audit(full_manifest: Path, train_manifest: Path, threshold: int) -> dict:
    full_records, _ = load_manifest(full_manifest)
    train_records, _ = load_manifest(train_manifest)
    train_index = PhashIndex()
    for record in train_records:
        train_index.add(record)

    result = {
        "phash_threshold": threshold,
        "train_records": len(train_records),
        "test_records": 0,
        "leaked_test_records": 0,
        "by_dataset": {},
        "examples": [],
    }
    for dataset in sorted({record.dataset for record in full_records}):
        test_records = [
            record for record in full_records
            if record.dataset == dataset and record.split == "test"
        ]
        leaked = []
        sources: Counter[str] = Counter()
        for record in test_records:
            matches = train_index.search(record.phash, threshold)
            if not matches:
                continue
            leaked.append(record)
            sources.update(match.dataset for match in matches)
            if len(result["examples"]) < 20:
                result["examples"].append({
                    "test": record.sources,
                    "matches": [match.sources for match in matches[:5]],
                })
        result["by_dataset"][dataset] = {
            "test_records": len(test_records),
            "leaked_test_records": len(leaked),
            "source_datasets": dict(sources),
        }
        result["test_records"] += len(test_records)
        result["leaked_test_records"] += len(leaked)
    result["passed"] = result["leaked_test_records"] == 0
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/ret3_manifest.json"))
    parser.add_argument("--train-manifest", type=Path, default=Path("data/processed/ret3_train.json"))
    parser.add_argument("--threshold", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.manifest, args.train_manifest, args.threshold)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
