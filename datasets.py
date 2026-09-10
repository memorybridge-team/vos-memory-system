"""Lazy acquisition and explicit split discovery. Importing never downloads."""
import hashlib
import json
import shutil
import stat
import tarfile
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

DAVIS_URL = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
MOSE_BASE = "https://huggingface.co/datasets/FudanCVL/MOSEv2/resolve/main/"
LVOS_VAL_ID = "1msjV2AAKROc-UsXh8lUic2gQpsLKfjQ0"
LVOS_META = "https://drive.google.com/drive/folders/1fOwGggoYNm_GkZIxs68ptHLk4JNF4Ebq"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, path, checksum=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (checksum is None or sha256(path) == checksum):
        return path
    partial = path.with_name(path.name + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "sam2-memory-benchmark/1"})
    with urllib.request.urlopen(request, timeout=120) as source, partial.open("wb") as dest:
        try:
            from tqdm import tqdm
            total = int(source.headers.get("Content-Length", 0)) or None
            progress = tqdm(total=total, unit="B", unit_scale=True,
                            desc=path.name, leave=True)
        except ImportError:
            progress = None
        try:
            while True:
                chunk = source.read(8 * 1024 * 1024)
                if not chunk:
                    break
                dest.write(chunk)
                if progress:
                    progress.update(len(chunk))
        finally:
            if progress:
                progress.close()
    if checksum and sha256(partial) != checksum:
        raise ValueError(f"SHA256 mismatch: {partial}")
    partial.replace(path)
    return path


def safe_target(root, name):
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Archive path escapes destination: {name}")
    return target


def extract(archive, destination):
    """Reject links, devices and traversal before writing each member."""
    archive, destination = Path(archive), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            for item in source.infolist():
                safe_target(destination, item.filename)
                if stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError("Archive symlinks are not accepted")
            source.extractall(destination)
    else:
        with tarfile.open(archive, "r|*") as source:
            for item in source:
                target = safe_target(destination, item.name)
                if item.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif item.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(item) as data, target.open("wb") as out:
                        shutil.copyfileobj(data, out)
                else:
                    raise ValueError(f"Unsupported archive entry: {item.name}")


def join_parts(parts, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as out:
        for part in parts:
            with Path(part).open("rb") as data:
                shutil.copyfileobj(data, out, 8 * 1024 * 1024)
    return destination


def acquire(spec, cache):
    """Acquire only the requested split when the runner reaches this dataset."""
    if spec.get("root"):
        root = Path(spec["root"]).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        return root
    name, split = spec["name"], spec["split"]
    cache = Path(cache).resolve()
    target = cache / "datasets" / name / split
    if (target / ".complete.json").is_file():
        return target
    archives = cache / "archives" / name
    archives.mkdir(parents=True, exist_ok=True)
    # Failed extractions never produce a completed cache. Stage under the cache
    # so renaming is atomic on the same filesystem.
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="extract-", dir=target.parent) as tmp:
        stage = Path(tmp) / "data"
        stage.mkdir()
        sources = []
        if name == "davis2017" and split in ("train", "val"):
            url = spec.get("url", DAVIS_URL)
            extract(download(url, archives / "trainval.zip"), stage)
            sources.append(url)
        elif name == "mosev2" and split in ("train", "valid"):
            if split == "train":
                filenames = ["train.tar.gz." + suffix for suffix in ("aa", "ab", "ac")]
            else:
                filenames = ["valid.tar.gz"]
            sums_path = download(MOSE_BASE + "SHA256SUMS", archives / "SHA256SUMS")
            checksums = {}
            for line in sums_path.read_text().splitlines():
                digest, filename = line.split(maxsplit=1)
                checksums[filename.lstrip("*")] = digest
            parts = []
            for filename in filenames:
                if filename not in checksums:
                    raise ValueError(f"Missing upstream checksum for {filename}")
                url = MOSE_BASE + filename
                parts.append(download(url, archives / filename, checksums[filename]))
                sources.append(url)
            if len(parts) == 1:
                extract(parts[0], stage)
            else:
                # Concatenate into a temporary archive; requires additional disk
                # equal to the compressed train split, documented in README.
                with tempfile.TemporaryDirectory(prefix="joined-", dir=archives) as joined:
                    packed = join_parts(parts, Path(joined) / "train.tar.gz")
                    extract(packed, stage)
            meta = "meta_" + split + ".json"
            shutil.copy2(download(MOSE_BASE + meta, archives / meta), stage / meta)
        elif name == "lvos" and split == "val":
            import gdown
            packed = archives / "lvos-v1-val.archive"
            if not packed.exists():
                partial = packed.with_suffix(".partial")
                result = gdown.download(id=LVOS_VAL_ID, output=str(partial), quiet=False)
                if result is None:
                    raise RuntimeError("LVOS download unavailable; set dataset root to a local LVOS v1 validation directory")
                partial.replace(packed)
            extract(packed, stage)
            metadata_dir = archives / "metadata"
            if not list(metadata_dir.glob("**/*meta.json")):
                result = gdown.download_folder(url=LVOS_META, output=str(metadata_dir), quiet=False)
                if result is None:
                    raise RuntimeError("LVOS metadata download failed")
            shutil.copytree(metadata_dir, stage / "metadata", dirs_exist_ok=True)
            sources += ["https://drive.google.com/file/d/" + LVOS_VAL_ID, LVOS_META]
        else:
            raise ValueError(f"Unsupported acquisition: {name}/{split}; provide root explicitly")
        # Validate the actual layout and metadata before declaring cache ready.
        discover(spec, stage)
        (stage / ".complete.json").write_text(json.dumps({"sources": sources, "name": name, "split": split}, indent=2))
        if target.exists():
            raise FileExistsError(f"Incomplete destination exists; inspect before retrying: {target}")
        stage.rename(target)
    return target


@dataclass
class Video:
    name: str
    frames: list
    frame_ids: list
    masks: dict
    objects: dict  # object ID -> metadata, empty means infer from initial mask
    score_available: bool
    dataset_name: str = ""
    split: str = ""
    evaluation_group: str = "all"

    def position(self, source_frame_id):
        try:
            return self.frame_ids.index(int(source_frame_id))
        except ValueError as exc:
            raise ValueError(
                f"Frame {source_frame_id} is not present in video {self.name}"
            ) from exc


def acquisition_plan(spec):
    """Return human-readable lazy-download inputs without network access."""
    if spec.get("root"):
        return [f"local:{spec['root']}"]
    name, split = spec["name"], spec["split"]
    if name == "davis2017":
        return [spec.get("url", DAVIS_URL)]
    if name == "lvos":
        return [f"gdrive:{LVOS_VAL_ID}", LVOS_META]
    if name == "mosev2":
        files = (["train.tar.gz.aa", "train.tar.gz.ab", "train.tar.gz.ac"]
                 if split == "train" else ["valid.tar.gz"])
        return [MOSE_BASE + name for name in ["SHA256SUMS", *files, f"meta_{split}.json"]]
    return ["unsupported"]


def discover(spec, root):
    root = Path(root)
    name, split = spec["name"], spec["split"]
    image_roots = list(root.rglob("JPEGImages"))
    if root.name == "JPEGImages":
        image_roots = [root]
    if name == "davis2017":
        candidates = [p / "480p" for p in image_roots if (p / "480p").is_dir()]
    else:
        aliases = {split, "valid" if split == "val" else split, "Validation" if split == "val" else split}
        candidates = [p for p in image_roots if p.parent.name in aliases]
        if not candidates and len(image_roots) == 1:
            candidates = image_roots
    if len(candidates) != 1:
        raise ValueError(f"Expected one {name}/{split} JPEGImages root under {root}, found {candidates}")
    images = candidates[0]
    annotations = images.parent.parent / "Annotations" / "480p" if name == "davis2017" else images.parent / "Annotations"
    objects = {}
    if name == "davis2017":
        split_path = images.parent.parent / "ImageSets" / "2017" / (split + ".txt")
        names = split_path.read_text().splitlines()
    elif name == "lvos":
        meta_path = spec.get("metadata")
        if meta_path:
            meta_path = Path(meta_path)
        else:
            metas = [p for p in root.rglob("*meta.json") if p.name in ("val_meta.json", "valid_meta.json")]
            if len(metas) != 1:
                raise ValueError("LVOS v1 requires one val_meta.json; set metadata in config if ambiguous")
            meta_path = metas[0]
        objects = json.loads(meta_path.read_text())["videos"]
        names = sorted(objects)
    else:
        meta_paths = list(root.rglob(f"meta_{split}.json"))
        if len(meta_paths) != 1:
            raise ValueError(f"MOSEv2 requires one official meta_{split}.json to verify completeness")
        meta = json.loads(meta_paths[0].read_text())
        names = sorted(meta["videos"])
        present = {p.name for p in images.iterdir() if p.is_dir()}
        if present != set(names):
            raise ValueError(f"MOSEv2 split mismatch: missing={sorted(set(names)-present)}, extra={sorted(present-set(names))}")
    videos = []
    for video_name in names:
        if not video_name or Path(video_name).name != video_name:
            raise ValueError(f"Invalid video name: {video_name!r}")
        frames = sorted((images / video_name).glob("*.jpg"), key=lambda p: int(p.stem))
        masks = {int(p.stem): p for p in (annotations / video_name).glob("*.png")}
        if not frames or not masks:
            raise ValueError(f"Missing frames or annotations: {video_name}")
        if len({int(p.stem) for p in frames}) != len(frames):
            raise ValueError(f"Duplicate numeric frame IDs: {video_name}")
        available = not (name == "mosev2" and split == "valid")
        # MOSEv2 train contains a small number of one-frame clips. They are
        # valid input videos and must remain in the selected video list, but
        # they cannot support a post-initialization accuracy score. Keep them
        # as completed, unscored jobs instead of rejecting the entire split.
        score_available = available and len(masks) >= 2
        frame_ids = [int(path.stem) for path in frames]
        evaluation_group = "all"
        if name == "lvos":
            unseen_path = Path(__file__).with_name("lvos_v1_unseen_videos.txt")
            unseen = set(unseen_path.read_text().splitlines())
            evaluation_group = "unseen" if video_name in unseen else "seen"
        videos.append(Video(video_name, frames, frame_ids, masks,
                            objects.get(video_name, {}).get("objects", {}),
                            score_available, name, split, evaluation_group))
    if not videos:
        raise ValueError("No videos found")
    return videos
