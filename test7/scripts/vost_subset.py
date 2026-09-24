#!/usr/bin/env python3
"""Inspect/extract a small subset from the official VOST ZIP using HTTP ranges.

The public archive is about 50 GB. This script reads its ZIP directory remotely
and downloads only requested entries; it never fetches the archive as a whole.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from collections import OrderedDict
from pathlib import Path, PurePosixPath


ARCHIVE_URL = "https://tri-ml-public.s3.amazonaws.com/datasets/VOST.zip"
BLOCK_SIZE = 4 * 1024 * 1024


class RemoteZip(io.RawIOBase):
    """Seekable ZIP reader backed by strict HTTP 206 ranges and a small LRU."""

    def __init__(self, url: str = ARCHIVE_URL) -> None:
        self.url = url
        self.position = 0
        self.blocks: OrderedDict[int, bytes] = OrderedDict()
        data, content_range = self._request_range(0, 0)
        if data != b"P":
            raise RuntimeError("remote object is not a ZIP file")
        self.length = int(content_range.split("/")[-1])

    def _request_range(self, start: int, end: int) -> tuple[bytes, str]:
        for attempt in range(3):
            request = urllib.request.Request(
                self.url,
                headers={"Range": f"bytes={start}-{end}", "User-Agent": "test7-vost-subset/1"},
            )
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    if response.status != 206:
                        raise RuntimeError(f"HTTP range expected 206, got {response.status}")
                    content_range = response.headers.get("Content-Range", "")
                    data = response.read()
                    if not content_range.startswith(f"bytes {start}-{end}/"):
                        raise RuntimeError(f"unexpected Content-Range: {content_range}")
                    if len(data) != end - start + 1:
                        raise RuntimeError(f"incomplete HTTP range {start}-{end}")
                    return data, content_range
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
                if attempt == 2:
                    raise
                time.sleep(1 + attempt)
        raise AssertionError("unreachable")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self.position + offset
        elif whence == os.SEEK_END:
            position = self.length + offset
        else:
            raise ValueError(f"invalid seek mode {whence}")
        if position < 0:
            raise ValueError("negative ZIP offset")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.length - self.position
        size = min(size, self.length - self.position)
        chunks: list[bytes] = []
        while size:
            block_start = (self.position // BLOCK_SIZE) * BLOCK_SIZE
            block = self.blocks.get(block_start)
            if block is None:
                block_end = min(block_start + BLOCK_SIZE, self.length) - 1
                block, _ = self._request_range(block_start, block_end)
                self.blocks[block_start] = block
                if len(self.blocks) > 4:
                    self.blocks.popitem(last=False)
            else:
                self.blocks.move_to_end(block_start)
            offset = self.position - block_start
            piece = block[offset : offset + size]
            if not piece:
                raise EOFError(f"empty ZIP range at {self.position}")
            chunks.append(piece)
            self.position += len(piece)
            size -= len(piece)
        return b"".join(chunks)


def archive_inventory(archive: zipfile.ZipFile) -> dict[str, dict[str, object]]:
    """Summarize sequence lengths and annotation/image byte totals."""
    inventory: dict[str, dict[str, object]] = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        parts = PurePosixPath(info.filename).parts
        if len(parts) < 4 or parts[-3] not in {"JPEGImages", "Annotations"}:
            continue
        kind, video, filename = parts[-3:]
        suffix = Path(filename).suffix.lower()
        frame_match = re.fullmatch(r"frame(\d+)", Path(filename).stem)
        if frame_match is None or suffix not in {".jpg", ".png"}:
            continue
        frame_id = int(frame_match.group(1))
        record = inventory.setdefault(video, {"images": [], "masks": [], "bytes": 0})
        record["images" if kind == "JPEGImages" else "masks"].append(
            (frame_id, info.file_size, info.filename)
        )
        record["bytes"] = int(record["bytes"]) + info.file_size
    result: dict[str, dict[str, object]] = {}
    for video, record in inventory.items():
        images = sorted(record["images"])
        masks = sorted(record["masks"])
        result[video] = {
            "image_count": len(images),
            "mask_count": len(masks),
            "first_image": images[0][0] if images else None,
            "last_image": images[-1][0] if images else None,
            "first_mask": masks[0][0] if masks else None,
            "last_mask": masks[-1][0] if masks else None,
            "bytes_for_sequence": int(record["bytes"]),
        }
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("data/vost_subset/val"))
    parser.add_argument("--videos", nargs="*", help="sequence IDs; omit with --list-only")
    parser.add_argument("--frame-limit", type=int, help="copy image/mask entries through this numeric frame ID")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    with zipfile.ZipFile(RemoteZip()) as archive:
        inventory = archive_inventory(archive)
        if args.list_only:
            split_path = f"VOST/ImageSets/{args.split}.txt"
            split_ids = set(archive.read(split_path).decode("utf-8").split())
            selected = {key: inventory[key] for key in sorted(split_ids) if key in inventory}
            print(json.dumps(selected, indent=2, sort_keys=True))
            return
        if not args.videos:
            parser.error("provide --videos or use --list-only")
        missing = sorted(set(args.videos) - inventory.keys())
        if missing:
            parser.error(f"unknown sequence IDs: {missing}")
        output_root = args.output_root.resolve()
        manifest: list[dict[str, object]] = []
        selected_bytes = 0
        for video in args.videos:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                parts = PurePosixPath(info.filename).parts
                if len(parts) < 3 or parts[-2] != video:
                    continue
                if parts[-3] not in {"JPEGImages", "Annotations"}:
                    continue
                frame_match = re.fullmatch(r"frame(\d+)", Path(parts[-1]).stem)
                if frame_match is None:
                    continue
                if args.frame_limit is not None and int(frame_match.group(1)) > args.frame_limit:
                    continue
                relative = PurePosixPath(*parts[-3:])
                destination = output_root.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    if destination.stat().st_size != info.file_size:
                        raise FileExistsError(f"existing file has wrong size: {destination}")
                else:
                    partial = destination.with_suffix(destination.suffix + ".part")
                    with archive.open(info) as source, partial.open("wb") as target:
                        shutil.copyfileobj(source, target, 1024 * 1024)
                    partial.replace(destination)
                manifest.append({
                    "filename": str(relative),
                    "bytes": destination.stat().st_size,
                    "sha256": sha256(destination),
                })
                selected_bytes += destination.stat().st_size
            print(f"{video}: extracted selected entries", flush=True)
        metadata = {
            "dataset": "VOST validation",
            "source": ARCHIVE_URL,
            "split": args.split,
            "videos": args.videos,
            "frame_limit": args.frame_limit,
            "selected_bytes": selected_bytes,
            "files": manifest,
        }
        image_sets = output_root / "ImageSets"
        image_sets.mkdir(parents=True, exist_ok=True)
        (image_sets / f"{args.split}.txt").write_text(
            "".join(f"{video}\n" for video in args.videos), encoding="utf-8"
        )
        (output_root / "download_manifest.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({key: metadata[key] for key in metadata if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
