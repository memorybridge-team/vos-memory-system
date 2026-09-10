import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from datasets import Video
from metrics import boundary_f, evaluate_indexed_masks, region_jaccard


class MetricTests(unittest.TestCase):
    def test_lvos_delayed_prompt_is_never_scored(self):
        with tempfile.TemporaryDirectory() as tmp:
            masks = {}
            predictions = {}
            for frame in (10, 15, 20, 25):
                array = np.ones((4, 4), np.uint8)
                path = Path(tmp) / f"{frame:08d}.png"
                Image.fromarray(array).save(path)
                masks[frame], predictions[frame] = path, array
            video = Video("late", [], [10, 15, 20, 25], masks,
                          {"1": {"frame_range": {"start": 10, "end": 30}}}, True, "lvos", "val")
            result = evaluate_indexed_masks(video, predictions, {1: 15})
            self.assertEqual([f["frame_id"] for f in result["objects"][0]["frames"]], [20])

    def test_empty_and_exact_masks(self):
        empty = np.zeros((8, 8), dtype=bool)
        square = empty.copy()
        square[2:6, 2:6] = True
        self.assertEqual(region_jaccard(empty, empty), 1.0)
        self.assertEqual(boundary_f(empty, empty), 1.0)
        self.assertEqual(region_jaccard(square, square), 1.0)
        self.assertEqual(boundary_f(square, square), 1.0)
        self.assertEqual(region_jaccard(empty, square), 0.0)
        self.assertEqual(boundary_f(empty, square), 0.0)

    def test_prompt_exclusion_and_lvos_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            masks = {}
            for frame_id in (0, 1, 2, 3):
                mask = np.zeros((8, 8), dtype=np.uint8)
                mask[2:6, 2:6] = 4
                path = root / f"{frame_id:05d}.png"
                Image.fromarray(mask).save(path)
                masks[frame_id] = path
            video = Video("v", [], [0, 1, 2, 3], masks, {
                "4": {"frame_range": {"start": 1, "end": 2}}
            }, True)
            predictions = {frame: np.asarray(Image.open(path)) for frame, path in masks.items()}
            result = evaluate_indexed_masks(video, predictions, {4: 1})
            self.assertEqual([f["frame_id"] for f in result["objects"][0]["frames"]], [2])
            self.assertEqual(result["j_and_f"], 1.0)

    def test_lvos_excludes_first_and_last_on_five_frame_cadence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            masks = {}
            for frame_id in (10, 15, 20, 25):
                mask = np.zeros((4, 4), dtype=np.uint8)
                mask[1:3, 1:3] = 1
                path = root / f"{frame_id:08d}.png"
                Image.fromarray(mask).save(path)
                masks[frame_id] = path
            video = Video("v", [], [10, 15, 20, 25], masks, {
                "1": {"frame_range": {"start": 10, "end": 30, "frame_nums": 4}}
            }, True, "lvos", "val")
            predictions = {frame: np.asarray(Image.open(path)) for frame, path in masks.items()}
            result = evaluate_indexed_masks(video, predictions, {1: 10})
            self.assertEqual(
                [f["frame_id"] for f in result["objects"][0]["frames"]], [15, 20])


if __name__ == "__main__":
    unittest.main()
