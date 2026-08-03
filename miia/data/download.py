from __future__ import annotations

import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tqdm import tqdm


@dataclass(frozen=True)
class DownloadSpec:
    dataset: str
    filename: str
    urls: tuple[str, ...] = ()
    google_drive_id: str | None = None
    manual_urls: tuple[str, ...] = ()
    note: str = ""


DOWNLOADS: dict[str, tuple[DownloadSpec, ...]] = {
    "rsicd": (
        DownloadSpec(
            dataset="rsicd",
            filename="dataset_rsicd.json",
            urls=(
                "https://raw.githubusercontent.com/201528014227051/RSICD_optimal/master/dataset_rsicd.json",
            ),
        ),
        DownloadSpec(
            dataset="rsicd",
            filename="RSICD_images.zip",
            urls=(
                "https://media.githubusercontent.com/media/201528014227051/RSICD_optimal/master/RSICD_images.zip",
            ),
            manual_urls=(
                "https://mega.nz/folder/EOpjTAwL#LWdHVjKAJbd3NbLsCvzDGA",
                "https://drive.google.com/open?id=0B1jt7lJDEXy3aE90cG9YSl9ScUk",
                "http://pan.baidu.com/s/1bp71tE3",
            ),
            note="The GitHub file is stored through Git LFS; use an official mirror if it is unavailable.",
        ),
    ),
    "rsitmd": (
        DownloadSpec(
            dataset="rsitmd",
            filename="RSITMD.zip",
            google_drive_id="1NJY86TAAUd8BVs7hyteImv8I2_Lh95W6",
            manual_urls=(
                "https://drive.google.com/file/d/1NJY86TAAUd8BVs7hyteImv8I2_Lh95W6/view?usp=sharing",
                "https://pan.baidu.com/s/1gDj38mzUL-LmQX32PYxr0Q (password: NIST)",
            ),
        ),
    ),
    "ucm": (
        DownloadSpec(
            dataset="ucm",
            filename="UCM_captions.zip",
            manual_urls=(
                "https://mega.nz/folder/wCpSzSoS#RXzIlrv--TDt3ENZdKN8JA",
                "https://pan.baidu.com/s/1mjPToHq",
            ),
            note="The official UCM-Captions mirrors are folder-based and may require a browser login.",
        ),
    ),
}


class DownloadUnavailable(RuntimeError):
    pass


def _download_http(url: str, target: Path, timeout: int = 30) -> None:
    try:
        import requests
    except ImportError as exc:
        raise DownloadUnavailable("Install requests to use automatic HTTP downloads") from exc
    temporary = target.with_suffix(target.suffix + ".part")
    headers = {"User-Agent": "miia-repro/0.1"}
    with requests.get(url, stream=True, timeout=timeout, headers=headers) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with temporary.open("wb") as stream, tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            desc=target.name,
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    stream.write(chunk)
                    progress.update(len(chunk))
    if temporary.stat().st_size < 128:
        temporary.unlink(missing_ok=True)
        raise DownloadUnavailable(f"Downloaded file is unexpectedly small: {url}")
    temporary.replace(target)


def _download_gdrive(file_id: str, target: Path) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise DownloadUnavailable("Install gdown to download the RSITMD Google Drive archive") from exc
    result = gdown.download(id=file_id, output=str(target), quiet=False, fuzzy=True)
    if not result or not target.exists():
        raise DownloadUnavailable(f"Google Drive download failed for id={file_id}")


def ensure_download(spec: DownloadSpec, cache_dir: Path, force: bool = False) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / spec.filename
    if target.exists() and target.stat().st_size >= 128 and not force:
        return target
    target.unlink(missing_ok=True)
    errors: list[str] = []
    for url in spec.urls:
        try:
            _download_http(url, target)
            return target
        except Exception as exc:  # each official mirror gets a chance
            errors.append(f"{url}: {exc}")
    if spec.google_drive_id:
        try:
            _download_gdrive(spec.google_drive_id, target)
            return target
        except Exception as exc:
            errors.append(f"Google Drive {spec.google_drive_id}: {exc}")
    instructions = [
        f"Automatic download for {spec.dataset}/{spec.filename} was unavailable.",
        f"Place the official file at: {target}",
    ]
    if spec.manual_urls:
        instructions.append("Official/manual mirrors:")
        instructions.extend(f"  - {url}" for url in spec.manual_urls)
    if spec.note:
        instructions.append(spec.note)
    if errors:
        instructions.append("Automatic attempts:")
        instructions.extend(f"  - {error}" for error in errors)
    raise DownloadUnavailable("\n".join(instructions))


def safe_extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            output = (destination / member.filename).resolve()
            if destination_root not in output.parents and output != destination_root:
                raise ValueError(f"Unsafe path in archive: {member.filename}")
        bundle.extractall(destination)


def safe_extract_archive(archive: Path, destination: Path) -> None:
    suffixes = "".join(archive.suffixes).lower()
    if suffixes.endswith(".zip"):
        safe_extract_zip(archive, destination)
        return
    if suffixes.endswith((".tar", ".tar.gz", ".tgz")):
        destination.mkdir(parents=True, exist_ok=True)
        root = destination.resolve()
        with tarfile.open(archive) as bundle:
            for member in bundle.getmembers():
                if member.issym() or member.islnk():
                    raise ValueError(f"Archive links are not permitted: {member.name}")
                output = (destination / member.name).resolve()
                if root not in output.parents and output != root:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
            bundle.extractall(destination)
        return
    raise ValueError(f"Unsupported archive type: {archive}")


def copy_files(files: Iterable[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source in files:
        shutil.copy2(source, destination / source.name)
