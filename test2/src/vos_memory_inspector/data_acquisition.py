"""Resumable, checksummed acquisition for the official LVOS v2 and VOST releases."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


# ``sha256``/``bytes`` pin the archives used for the recorded experiments. The
# upstream projects do not publish checksums, so these values were recorded at
# first acquisition; a mismatch means the hosted file changed or is corrupt.
SOURCES = {
    "lvos_v2_train": {
        "dataset": "LVOS v2",
        "release": "v2",
        "url": "https://drive.google.com/uc?id=1-ehpl5s0Fd14WwtT-GmWtIWa_BxZl9D6",
        "provider": "google_drive",
        "license": "non-commercial research; annotations CC BY 4.0",
        "sha256": "e4c0cfcb400dbd103ddea6eac3b5ffb5cdd3d50cb87013bf457d963a0b248c0b",
        "bytes": 22414546249,
    },
    "lvos_v2_val": {
        "dataset": "LVOS v2",
        "release": "v2",
        "url": "https://drive.google.com/uc?id=17Hwc__6i2rpF5e2s5OPqoywNxG5bzlcO",
        "provider": "google_drive",
        "license": "non-commercial research; annotations CC BY 4.0",
        "sha256": "beb488046f74e0cb4154a0cb2bcc2c79cae858da0693e5adabcf966a5712d4e2",
        "bytes": 11833302254,
    },
    "lvos_v2_metadata": {
        "dataset": "LVOS v2",
        "release": "v2 metadata",
        "url": "https://drive.google.com/drive/folders/1EtTW57QfSkUK3Jl1A_m9D120muA7NG4L",
        "provider": "google_drive_folder",
        "license": "non-commercial research; annotations CC BY 4.0",
    },
    "vost": {
        "dataset": "VOST",
        "release": "official train+validation bundle",
        "url": "https://tri-ml-public.s3.amazonaws.com/datasets/VOST.zip",
        "provider": "http",
        "license": "CC BY-NC-SA 4.0",
        "sha256": "fb17075ab3afab0fe30f264d8adce2e29ce6249a73cd44d2a9cf4936cc8de978",
        "bytes": 54012104924,
    },
}


class ChecksumMismatchError(ValueError):
    """The acquired archive differs from the pinned release."""


def verify_archive(archive: Path, source: dict[str, Any], digest: str) -> bool | None:
    """Compare an archive with its pinned size and SHA-256.

    Returns ``True`` when it matches, ``None`` when no checksum is pinned, and
    raises :class:`ChecksumMismatchError` otherwise.
    """

    expected_bytes = source.get("bytes")
    actual_bytes = archive.stat().st_size
    if expected_bytes is not None and actual_bytes != int(expected_bytes):
        raise ChecksumMismatchError(
            f"{archive.name}: expected {expected_bytes} bytes, found {actual_bytes}"
        )
    expected = source.get("sha256")
    if expected is None:
        return None
    if digest != expected:
        raise ChecksumMismatchError(
            f"{archive.name}: expected SHA-256 {expected}, found {digest}"
        )
    return True


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _http_download(url: str, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "cmmt-research/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response:
        status = getattr(response, "status", 200)
        if offset and status != 206:
            raise RuntimeError(
                "server ignored the resume range; preserve the partial file and retry explicitly"
            )
        mode = "ab" if offset else "wb"
        with partial.open(mode) as stream:
            shutil.copyfileobj(response, stream, length=4 * 1024 * 1024)
    partial.replace(destination)


def _google_drive_download(url: str, destination: Path) -> None:
    if importlib.util.find_spec("gdown") is None:
        raise RuntimeError(
            "LVOS v2 is hosted on Google Drive. Install the optional `gdown` package "
            "in the research environment or download the official archive manually; "
            "the acquisition status records this as blocked without touching other data."
        )
    import gdown

    partial = destination.with_suffix(destination.suffix + ".partial")
    result = gdown.download(url=url, output=str(partial), quiet=False, resume=True)
    if result is None or not partial.is_file():
        raise RuntimeError("gdown did not produce the requested LVOS archive")
    partial.replace(destination)


def _google_drive_folder_download(url: str, destination: Path) -> list[Path]:
    if importlib.util.find_spec("gdown") is None:
        raise RuntimeError("LVOS v2 metadata download requires the optional `gdown` package")
    import gdown

    destination.mkdir(parents=True, exist_ok=True)
    outputs = gdown.download_folder(url=url, output=str(destination), quiet=False, resume=True)
    if not outputs:
        raise RuntimeError("gdown did not produce the requested LVOS metadata folder")
    return [Path(path).resolve() for path in outputs if Path(path).is_file()]


def _safe_extract_zip(archive: Path, destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    count = 0
    with zipfile.ZipFile(archive) as bundle:
        bad = bundle.testzip()
        if bad is not None:
            raise ValueError(f"archive CRC check failed at {bad}")
        for member in bundle.infolist():
            output = (destination / member.filename).resolve()
            if destination_resolved not in output.parents and output != destination_resolved:
                raise ValueError(f"archive contains unsafe path: {member.filename}")
        bundle.extractall(destination)
        count = sum(not member.is_dir() for member in bundle.infolist())
    return count


def acquire_dataset(
    source_name: str,
    *,
    download_dir: str | Path,
    extract_dir: str | Path,
    archive_name: str | None = None,
    verify_checksum: bool = True,
) -> dict[str, Any]:
    """Download one official archive without deleting or overwriting existing data.

    When ``verify_checksum`` is true (default) an archive that differs from the
    pinned size/SHA-256 is rejected before extraction. Otherwise the mismatch is
    recorded in the status file and extraction proceeds.
    """

    if source_name not in SOURCES:
        raise ValueError(f"unknown source {source_name!r}; choose from {sorted(SOURCES)}")
    source = SOURCES[source_name]
    download_dir = Path(download_dir).resolve()
    extract_dir = Path(extract_dir).resolve()
    download_dir.mkdir(parents=True, exist_ok=True)
    archive = download_dir / (archive_name or f"{source_name}.zip")
    status_path = download_dir / f"{source_name}.acquisition.json"
    status: dict[str, Any] = {
        "schema_version": "cmmt.dataset_acquisition.v1",
        "source_name": source_name,
        **source,
        "archive": None if source["provider"] == "google_drive_folder" else str(archive),
        "extract_dir": str(extract_dir),
        "state": "downloading",
    }

    def save() -> None:
        partial = status_path.with_suffix(".json.partial")
        partial.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
        partial.replace(status_path)

    save()
    try:
        if source["provider"] == "google_drive_folder":
            files = _google_drive_folder_download(str(source["url"]), extract_dir)
            status["files"] = [
                {
                    "path": str(path.relative_to(extract_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in sorted(files)
            ]
            status["extracted_file_count"] = len(files)
            status["state"] = "complete"
            save()
            return status
        if not archive.is_file():
            if source["provider"] == "google_drive":
                _google_drive_download(str(source["url"]), archive)
            else:
                _http_download(str(source["url"]), archive)
        status["archive_bytes"] = archive.stat().st_size
        status["archive_sha256"] = sha256_file(archive)
        try:
            status["checksum_verified"] = verify_archive(
                archive, source, status["archive_sha256"]
            )
        except ChecksumMismatchError as exc:
            if verify_checksum:
                raise
            status["checksum_verified"] = False
            status["checksum_warning"] = str(exc)
        status["state"] = "extracting"
        save()
        status["extracted_file_count"] = _safe_extract_zip(archive, extract_dir)
        status["state"] = "complete"
    except Exception as exc:
        status["state"] = "blocked"
        status["error_type"] = type(exc).__name__
        status["error"] = str(exc)
        save()
        raise
    save()
    return status
