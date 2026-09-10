import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from src.datasets import (Video, acquisition_plan, discover, download, extract,
                          join_parts)


def image(path, value=0, rgb=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((8, 10, 3) if rgb else (8, 10), value, dtype=np.uint8)
    Image.fromarray(array).save(path)


class DatasetTests(unittest.TestCase):
    def test_davis_frame_mapping_and_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "DAVIS"
            image(root / "JPEGImages/480p/bike/00003.jpg", rgb=True)
            image(root / "JPEGImages/480p/bike/00007.jpg", rgb=True)
            image(root / "Annotations/480p/bike/00003.png", 1)
            image(root / "Annotations/480p/bike/00007.png", 1)
            split = root / "ImageSets/2017/val.txt"
            split.parent.mkdir(parents=True)
            split.write_text("bike\n")
            videos = discover({"name": "davis2017", "split": "val"}, root)
            self.assertEqual(videos[0].frame_ids, [3, 7])
            self.assertEqual(videos[0].position(7), 1)

    def test_lvos_metadata_is_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for frame in (12, 13):
                image(root / f"val/JPEGImages/clip/{frame:08d}.jpg", rgb=True)
                image(root / f"val/Annotations/clip/{frame:08d}.png", 2)
            (root / "metadata").mkdir()
            (root / "metadata/val_meta.json").write_text(json.dumps({
                "videos": {"clip": {"objects": {"2": {
                    "frame_range": {"start": 12, "end": 13}}}}}
            }))
            video = discover({"name": "lvos", "version": "v1", "split": "val"}, root)[0]
            self.assertEqual(video.objects["2"]["frame_range"]["start"], 12)

    def test_mose_train_and_valid_scoring_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split, mask_frames in (("train", (0, 1)), ("valid", (0,))):
                (root / f"meta_{split}.json").write_text(json.dumps({"videos": {"clip": {}}}))
                for frame in (0, 1):
                    image(root / f"{split}/JPEGImages/clip/{frame:05d}.jpg", rgb=True)
                for frame in mask_frames:
                    image(root / f"{split}/Annotations/clip/{frame:05d}.png", 1)
            train = discover({"name": "mosev2", "split": "train"}, root)[0]
            valid = discover({"name": "mosev2", "split": "valid"}, root)[0]
            self.assertTrue(train.score_available)
            self.assertFalse(valid.score_available)
            (root / "meta_train.json").write_text(json.dumps({"videos": {"clip": {}, "missing": {}}}))
            with self.assertRaisesRegex(ValueError, "split mismatch"):
                discover({"name": "mosev2", "split": "train"}, root)

    def test_archive_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "bad.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("../escape", b"bad")
            with self.assertRaises(ValueError):
                extract(archive, Path(tmp) / "out")

    def test_acquisition_plan_is_pure(self):
        sources = acquisition_plan({"name": "mosev2", "split": "train"})
        self.assertEqual(len(sources), 5)
        self.assertTrue(sources[0].endswith("SHA256SUMS"))

    def test_parts_are_joined_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parts = []
            for index, value in enumerate((b"abc", b"def", b"ghi")):
                path = root / f"part-{index}"
                path.write_bytes(value)
                parts.append(path)
            result = join_parts(parts, root / "joined")
            self.assertEqual(result.read_bytes(), b"abcdefghi")

    def test_download_cache_and_checksum_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.write_bytes(b"payload")
            target = root / "cache/file"
            download(source.as_uri(), target)
            first_mtime = target.stat().st_mtime_ns
            source.write_bytes(b"changed")
            download(source.as_uri(), target)
            self.assertEqual(target.read_bytes(), b"payload")
            self.assertEqual(target.stat().st_mtime_ns, first_mtime)
            bad = root / "cache/bad"
            with self.assertRaises(ValueError):
                download(source.as_uri(), bad, "0" * 64)
            self.assertFalse(bad.exists())
            expected = hashlib.sha256(b"changed").hexdigest()
            download(source.as_uri(), bad, expected)
            self.assertEqual(bad.read_bytes(), b"changed")


if __name__ == "__main__":
    unittest.main()
