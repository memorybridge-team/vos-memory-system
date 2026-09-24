"""Fetch selected LVOS v2 validation clips from the official ZIP via HTTP ranges.

The official Google Drive download is one 11 GB ZIP. This script reads its ZIP
directory remotely and downloads only the requested JPEG/annotation entries.
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
import urllib.parse
import urllib.request
import zipfile
from collections import OrderedDict
from pathlib import Path, PurePosixPath


FILE_ID = "17Hwc__6i2rpF5e2s5OPqoywNxG5bzlcO"
VIDEO_IDS = (
    "2VegYEbT", "2urlAsm8", "0tCWPOrc", "9HEh93ef",
    # Lowest prior Base+ cold-start J&F clips from the local validation run.
    "x3nD3QQ9", "MKnlVo6x", "8lxxCA5h", "nfcT3owb", "q1MSEBkh",
    "ScFTYisJ", "aFytsETk", "dtHbJvYy", "xpI7xRWN", "K3OUeINk",
)
PRIOR_HARD_JF = {
    "x3nD3QQ9": 0.07730599859907754,
    "MKnlVo6x": 0.27105326832275867,
    "8lxxCA5h": 0.36444947716571596,
    "nfcT3owb": 0.4218655500270669,
    "q1MSEBkh": 0.4242214615036243,
    "ScFTYisJ": 0.542509889298446,
    "aFytsETk": 0.5449554056680113,
    "dtHbJvYy": 0.5515834527119061,
    "xpI7xRWN": 0.5655915945346737,
    "K3OUeINk": 0.5695193303102182,
}
BLOCK_SIZE = 4 * 1024 * 1024


class RemoteZipFile(io.RawIOBase):
    """Seekable, small-LRU-cached reader backed by strict HTTP 206 responses."""

    def __init__(self, file_id: str) -> None:
        self.file_id = file_id
        self.position = 0
        self.blocks: OrderedDict[int, bytes] = OrderedDict()
        self.url = self._download_url()
        response = self._request_range(0, 0)
        self.length = int(response[1].split("/")[-1])

    def _download_url(self) -> str:
        landing = (
            "https://drive.google.com/uc?"
            + urllib.parse.urlencode({"export": "download", "id": self.file_id})
        )
        with urllib.request.urlopen(landing, timeout=60) as response:
            html = response.read().decode("utf-8")
        match = re.search(r'name="uuid" value="([^"]+)"', html)
        if match is None:
            raise RuntimeError("official Drive download confirmation UUID not found")
        return "https://drive.usercontent.google.com/download?" + urllib.parse.urlencode(
            {"id": self.file_id, "export": "download", "confirm": "t", "uuid": match[1]}
        )

    def _request_range(self, start: int, end: int) -> tuple[bytes, str]:
        for attempt in range(3):
            request = urllib.request.Request(
                self.url,
                headers={"Range": f"bytes={start}-{end}", "User-Agent": "cmmt-lvos-subset/1"},
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
                        raise RuntimeError(f"incomplete HTTP range: {start}-{end}")
                    return data, content_range
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
                if attempt == 2:
                    raise
                self.url = self._download_url()
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
            raise ValueError(f"invalid seek mode: {whence}")
        if position < 0:
            raise ValueError("negative ZIP offset")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.length - self.position
        size = min(size, self.length - self.position)
        if size <= 0:
            return b""
        chunks: list[bytes] = []
        remaining = size
        while remaining:
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
            piece = block[offset : offset + remaining]
            if not piece:
                raise EOFError(f"empty ZIP range at offset {self.position}")
            chunks.append(piece)
            self.position += len(piece)
            remaining -= len(piece)
        return b"".join(chunks)


def _selected_entries(
    archive: zipfile.ZipFile, video_ids: tuple[str, ...], frame_count: int
) -> dict[str, dict[str, list[zipfile.ZipInfo]]]:
    grouped: dict[str, dict[str, list[zipfile.ZipInfo]]] = {
        video: {"JPEGImages": [], "Annotations": []} for video in video_ids
    }
    for info in archive.infolist():
        parts = PurePosixPath(info.filename).parts
        if info.is_dir() or len(parts) < 3:
            continue
        kind, video, filename = parts[-3:]
        if video not in grouped or kind not in grouped[video]:
            continue
        if not Path(filename).stem.isdigit():
            continue
        grouped[video][kind].append(info)
    for video, kinds in grouped.items():
        for kind in kinds:
            kinds[kind].sort(key=lambda info: int(PurePosixPath(info.filename).stem))
            kinds[kind] = kinds[kind][:frame_count]
        image_ids = [int(PurePosixPath(info.filename).stem) for info in kinds["JPEGImages"]]
        annotation_ids = [int(PurePosixPath(info.filename).stem) for info in kinds["Annotations"]]
        if len(image_ids) != frame_count or image_ids != annotation_ids:
            raise ValueError(
                f"{video}: expected {frame_count} paired frames, got "
                f"RGB={len(image_ids)} GT={len(annotation_ids)}"
            )
    return grouped


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("data/lvosv2_subset/val"))
    parser.add_argument("--frame-count", type=int, default=40)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()
    if args.frame_count <= 21:
        parser.error("frame count must include the fixed switch frame 20 and its continuation")

    remote = RemoteZipFile(FILE_ID)
    with zipfile.ZipFile(remote) as archive:
        grouped = _selected_entries(archive, VIDEO_IDS, args.frame_count)
        for video, kinds in grouped.items():
            ids = [int(PurePosixPath(info.filename).stem) for info in kinds["JPEGImages"]]
            total = sum(info.file_size for entries in kinds.values() for info in entries)
            print(f"{video}: {len(ids)} paired frames, IDs {ids[0]}..{ids[-1]}, {total:,} bytes", flush=True)
        if args.list_only:
            return

        output_root = args.output_root.resolve()
        records: list[dict[str, object]] = []
        for video, kinds in grouped.items():
            for kind, infos in kinds.items():
                destination_dir = output_root / kind / video
                destination_dir.mkdir(parents=True, exist_ok=True)
                for info in infos:
                    destination = destination_dir / PurePosixPath(info.filename).name
                    if destination.exists():
                        if destination.stat().st_size != info.file_size:
                            raise FileExistsError(f"existing file has wrong size: {destination}")
                    else:
                        partial = destination.with_suffix(destination.suffix + ".part")
                        try:
                            with archive.open(info) as source, partial.open("wb") as target:
                                shutil.copyfileobj(source, target, length=1024 * 1024)
                            if partial.stat().st_size != info.file_size:
                                raise IOError(f"incomplete ZIP member: {info.filename}")
                            partial.replace(destination)
                        finally:
                            partial.unlink(missing_ok=True)
                    records.append(
                        {
                            "video_id": video,
                            "kind": kind,
                            "frame_id": int(destination.stem),
                            "filename": str(destination.relative_to(output_root)),
                            "bytes": info.file_size,
                            "zip_crc32": f"{info.CRC:08x}",
                            "sha256": _sha256_file(destination),
                        }
                    )
            print(f"completed {video}", flush=True)

    manifest = {
        "source": "https://drive.google.com/file/d/" + FILE_ID + "/view",
        "archive_bytes": remote.length,
        "dataset": "LVOS v2 validation",
        "frame_policy": f"first {args.frame_count} available frames per video",
        "videos": list(VIDEO_IDS),
        "hard_video_prior_JF": PRIOR_HARD_JF,
        "files": records,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved {len(records)} files under {output_root}")


if __name__ == "__main__":
    main()
