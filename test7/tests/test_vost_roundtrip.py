import unittest
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from vos_memory_inspector.vost_roundtrip import load_clip


def _write_label(path: Path, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(labels.astype(np.uint8)).save(path)


class VOSTClipLoaderTests(unittest.TestCase):
    def test_load_clip_uses_first_frame_objects_and_keeps_original_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video_id = "555_tear_aluminium_foil"
            image_dir = root / "JPEGImages" / video_id
            annotation_dir = root / "Annotations" / video_id
            image_dir.mkdir(parents=True)
            for frame_id in (0, 12, 24):
                Image.new("RGB", (4, 4)).save(image_dir / f"frame{frame_id:06d}.jpg")
                labels = np.zeros((4, 4), dtype=np.uint8)
                labels[0, 0] = 1
                labels[1, 1] = 2
                _write_label(annotation_dir / f"frame{frame_id:06d}.png", labels)

            clip = load_clip(root, video_id)

            self.assertEqual(clip.original_frame_ids, (0, 12, 24))
            self.assertEqual(clip.switch_index, 1)
            self.assertEqual(
                [(prompt.object_id, prompt.frame_idx, prompt.original_frame_id)
                 for prompt in clip.prompts],
                [(1, 0, 0), (2, 0, 0)],
            )

    def test_load_clip_rejects_object_that_appears_after_initial_prompt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video_id = "6922_split_paper"
            image_dir = root / "JPEGImages" / video_id
            annotation_dir = root / "Annotations" / video_id
            image_dir.mkdir(parents=True)
            for frame_id in (0, 6, 12):
                Image.new("RGB", (4, 4)).save(image_dir / f"frame{frame_id:06d}.jpg")
                labels = np.zeros((4, 4), dtype=np.uint8)
                labels[0, 0] = 1
                if frame_id == 12:
                    labels[1, 1] = 2
                _write_label(annotation_dir / f"frame{frame_id:06d}.png", labels)

            with self.assertRaisesRegex(ValueError, "absent from the initial prompt"):
                load_clip(root, video_id)


if __name__ == "__main__":
    unittest.main()
