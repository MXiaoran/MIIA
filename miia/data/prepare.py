from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable

from PIL import Image

from miia.data.download import DOWNLOADS, DownloadUnavailable, ensure_download, safe_extract_archive
from miia.data.manifest import DatasetRecord, parse_dataset, perceptual_hash, write_manifests
from miia.utils import atomic_json_dump, sha256_file, stable_hash

ANNOTATION_NAMES = {
    "rsicd": ("dataset_rsicd.json",),
    "rsitmd": ("dataset_RSI_TMD.json", "dataset_RSITMD.json", "dataset_rsitmd.json"),
    "ucm": ("dataset_ucm.json", "dataset_UCM.json", "dataset_UCM_captions.json"),
}


def _find_one(root: Path, names: Iterable[str]) -> Path | None:
    lower_names = {name.lower() for name in names}
    matches = [path for path in root.rglob("*") if path.is_file() and path.name.lower() in lower_names]
    return sorted(matches, key=lambda path: len(path.parts))[0] if matches else None


def _find_images_root(root: Path, annotation: Path) -> Path:
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    candidate_dirs = [annotation.parent, root]
    candidate_dirs.extend(path for path in root.rglob("*") if path.is_dir() and path.name.lower() in {"images", "pics", "rsicd_images", "ucm_images"})
    scored: list[tuple[int, Path]] = []
    for candidate in candidate_dirs:
        count = sum(1 for path in candidate.rglob("*") if path.is_file() and path.suffix.lower() in extensions)
        if count:
            scored.append((count, candidate))
    if not scored:
        raise FileNotFoundError(f"No images found below {root}")
    return max(scored, key=lambda item: item[0])[1]


def _prepare_downloaded_dataset(dataset: str, raw_root: Path, cache_root: Path, force: bool) -> tuple[Path, Path]:
    dataset_root = raw_root / dataset
    dataset_root.mkdir(parents=True, exist_ok=True)
    annotation = _find_one(dataset_root, ANNOTATION_NAMES[dataset])
    try:
        images_root = _find_images_root(dataset_root, annotation) if annotation is not None else None
    except FileNotFoundError:
        images_root = None

    failures: list[str] = []
    if annotation is None or images_root is None or force:
        for spec in DOWNLOADS[dataset]:
            try:
                downloaded = ensure_download(spec, cache_root / dataset, force=force)
                if downloaded.suffix.lower() == ".zip" or "".join(downloaded.suffixes).lower().endswith((".tar.gz", ".tgz")):
                    safe_extract_archive(downloaded, dataset_root)
                elif downloaded.suffix.lower() == ".json":
                    shutil.copy2(downloaded, dataset_root / downloaded.name)
            except Exception as exc:
                failures.append(str(exc))
        annotation = _find_one(dataset_root, ANNOTATION_NAMES[dataset])
    if annotation is not None:
        try:
            images_root = _find_images_root(dataset_root, annotation)
        except FileNotFoundError:
            images_root = None
    if annotation is None or images_root is None:
        expected = [f"  - {cache_root / dataset / spec.filename}" for spec in DOWNLOADS[dataset]]
        reason = "annotation JSON" if annotation is None else "image files"
        message = [
            f"Could not locate the required {reason} for {dataset}.",
            "Download the official files to one of these cache paths, then rerun the command:",
            *expected,
        ]
        if failures:
            message.extend(["", *failures])
        raise DownloadUnavailable("\n".join(message))
    return annotation, images_root


def prepare(args: argparse.Namespace) -> int:
    if args.skip_hashes:
        raise ValueError("--skip-hashes is incompatible with RET-3 leakage and duplicate auditing")
    project_root = Path(args.project_root).resolve()
    raw_root = project_root / args.raw_dir
    cache_root = project_root / args.cache_dir
    processed_root = project_root / args.processed_dir
    for dataset in args.datasets:
        dataset_root = raw_root / dataset
        cache_dataset = cache_root / dataset
        if _find_one(dataset_root, ANNOTATION_NAMES[dataset]) is None and cache_dataset.exists():
            for archive in cache_dataset.iterdir():
                suffixes = "".join(archive.suffixes).lower()
                if archive.is_file() and suffixes.endswith((".zip", ".tar", ".tar.gz", ".tgz")):
                    safe_extract_archive(archive, dataset_root)
    records: list[DatasetRecord] = []
    for dataset in args.datasets:
        annotation, images_root = _prepare_downloaded_dataset(dataset, raw_root, cache_root, args.force)
        dataset_records = parse_dataset(dataset, annotation, images_root, compute_hashes=not args.skip_hashes)
        records.extend(dataset_records)
        print(f"Prepared {dataset}: {len(dataset_records)} images from {images_root}")
    result = write_manifests(
        records,
        processed_root,
        strict_counts=not args.allow_incomplete and set(args.datasets) == set(DOWNLOADS),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def smoke(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    source_root = Path(args.source_root).resolve() if args.source_root else None
    patterns = {}
    if source_root is not None:
        patterns = {
            "rsicd": source_root / "rsicd",
            "rsitmd": source_root / "rsitmd",
            "ucm": source_root / "ucm",
        }
    smoke_root = project_root / "data" / "smoke"
    smoke_root.mkdir(parents=True, exist_ok=True)
    records: list[DatasetRecord] = []
    for dataset in ("rsicd", "rsitmd", "ucm"):
        root = patterns.get(dataset)
        images = [] if root is None else [
            path for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        ][:4]
        if len(images) < 4:
            dataset_root = smoke_root / dataset
            dataset_root.mkdir(parents=True, exist_ok=True)
            images = []
            for index in range(4):
                path = dataset_root / f"sample_{index}.png"
                # Deterministic synthetic aerial-like textures keep the public
                # smoke test self-contained without redistributing benchmarks.
                color = ((index * 47 + len(dataset) * 13) % 256, (index * 83 + 61) % 256, (index * 29 + 137) % 256)
                image = Image.new("RGB", (64, 64), color)
                for offset in range(0, 64, 8):
                    for x in range(64):
                        image.putpixel((x, (offset + x // 4) % 64), tuple(min(255, value + 24) for value in color))
                image.save(path)
                images.append(path)
        for index, path in enumerate(images):
            with Image.open(path) as image:
                phash = perceptual_hash(image)
            records.append(DatasetRecord(
                dataset=dataset,
                image_id=path.stem,
                image_path=str(path.resolve()),
                captions=[
                    f"A remote sensing image from the {dataset} smoke set number {index}.",
                    f"This aerial scene is a smoke-test sample for {dataset}.",
                    f"Satellite style imagery used to verify the {dataset} data loader.",
                    f"A small example image for cross-modal retrieval in {dataset}.",
                    f"Remote sensing test image {index} from dataset {dataset}.",
                ],
                split="train" if index < 2 else "val" if index == 2 else "test",
                sha256=sha256_file(path),
                phash=phash,
                sources=[f"smoke:{dataset}:{path.name}"],
            ))
    output = project_root / args.processed_dir
    output.mkdir(parents=True, exist_ok=True)
    payload_records = [record.to_dict() for record in records]
    train_records = [record.to_dict() for record in records if record.split == "train"]
    payload = {"schema_version": 1, "manifest_hash": stable_hash(payload_records), "records": payload_records}
    train_payload = {"schema_version": 1, "manifest_hash": stable_hash(train_records), "records": train_records}
    atomic_json_dump(payload, output / "smoke_manifest.json")
    atomic_json_dump(train_payload, output / "smoke_train.json")
    print(f"Wrote {len(records)} smoke records to {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the RET-3 datasets for MIIA")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[2])
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare", help="download, validate, hash and de-duplicate RET-3")
    prepare_parser.add_argument("--datasets", nargs="+", choices=sorted(DOWNLOADS), default=sorted(DOWNLOADS))
    prepare_parser.add_argument("--raw-dir", default="data/raw")
    prepare_parser.add_argument("--cache-dir", default="data/cache")
    prepare_parser.add_argument("--processed-dir", default="data/processed")
    prepare_parser.add_argument("--force", action="store_true")
    prepare_parser.add_argument("--allow-incomplete", action="store_true")
    prepare_parser.add_argument(
        "--skip-hashes",
        action="store_true",
        help="diagnostic-only flag; rejected for RET-3 because pHash auditing is required",
    )
    prepare_parser.set_defaults(func=prepare)
    smoke_parser = subparsers.add_parser("smoke", help="build a tiny manifest from images already in this workspace")
    smoke_parser.add_argument(
        "--source-root",
        default=None,
        help="optional directory with rsicd/rsitmd/ucm image subdirectories; synthetic images are used otherwise",
    )
    smoke_parser.add_argument("--processed-dir", default="data/processed")
    smoke_parser.set_defaults(func=smoke)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except DownloadUnavailable as exc:
        print(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
