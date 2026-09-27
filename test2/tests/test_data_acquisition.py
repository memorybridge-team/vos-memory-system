from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from vos_memory_inspector import data_acquisition
from vos_memory_inspector.data_acquisition import (
    SOURCES,
    ChecksumMismatchError,
    acquire_dataset,
)


def _archive(download_dir: Path, name: str) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    archive = download_dir / f"{name}.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("dataset/file.txt", "payload")
    return archive


def _register(monkeypatch, name: str, archive: Path, *, sha256: str | None) -> None:
    source = {
        "dataset": "fixture",
        "release": "test",
        "url": "http://127.0.0.1/unused.zip",
        "provider": "http",
        "license": "test",
        "bytes": archive.stat().st_size,
    }
    if sha256 is not None:
        source["sha256"] = sha256
    monkeypatch.setitem(data_acquisition.SOURCES, name, source)


def test_pinned_archives_have_sha256_and_size() -> None:
    for name, source in SOURCES.items():
        if source["provider"] == "google_drive_folder":
            continue
        assert len(source["sha256"]) == 64, name
        assert int(source["bytes"]) > 0, name


def test_matching_archive_is_verified_and_extracted(tmp_path: Path, monkeypatch) -> None:
    archive = _archive(tmp_path / "downloads", "fixture")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    _register(monkeypatch, "fixture", archive, sha256=digest)
    status = acquire_dataset(
        "fixture", download_dir=tmp_path / "downloads", extract_dir=tmp_path / "data"
    )
    assert status["checksum_verified"] is True
    assert status["state"] == "complete"
    assert (tmp_path / "data" / "dataset" / "file.txt").read_text() == "payload"


def test_mismatched_archive_is_rejected_before_extraction(tmp_path: Path, monkeypatch) -> None:
    archive = _archive(tmp_path / "downloads", "fixture")
    _register(monkeypatch, "fixture", archive, sha256="0" * 64)
    with pytest.raises(ChecksumMismatchError, match="expected SHA-256"):
        acquire_dataset(
            "fixture", download_dir=tmp_path / "downloads", extract_dir=tmp_path / "data"
        )
    assert not (tmp_path / "data").exists()
    status = json.loads((tmp_path / "downloads" / "fixture.acquisition.json").read_text())
    assert status["state"] == "blocked"
    assert status["error_type"] == "ChecksumMismatchError"

    allowed = acquire_dataset(
        "fixture",
        download_dir=tmp_path / "downloads",
        extract_dir=tmp_path / "data",
        verify_checksum=False,
    )
    assert allowed["checksum_verified"] is False
    assert "expected SHA-256" in allowed["checksum_warning"]
    assert allowed["state"] == "complete"


def test_unpinned_archive_is_recorded_as_unverified(tmp_path: Path, monkeypatch) -> None:
    archive = _archive(tmp_path / "downloads", "fixture")
    _register(monkeypatch, "fixture", archive, sha256=None)
    status = acquire_dataset(
        "fixture", download_dir=tmp_path / "downloads", extract_dir=tmp_path / "data"
    )
    assert status["checksum_verified"] is None
    assert len(status["archive_sha256"]) == 64


def test_unsafe_archive_member_is_rejected(tmp_path: Path, monkeypatch) -> None:
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    archive = downloads / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.txt", "x")
    _register(monkeypatch, "fixture", archive, sha256=None)
    with pytest.raises(ValueError, match="unsafe path"):
        acquire_dataset("fixture", download_dir=downloads, extract_dir=tmp_path / "data")
    assert not (tmp_path / "escape.txt").exists()
