from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vos_memory_inspector.base_roundtrip import (
    Clip,
    VIDEO_IDS,
    Prompt,
    load_clip,
    make_frame_rows,
    masks_by_id,
    score_frame,
    write_svg,
)


def _write_sparse_clip(root: Path, video_id: str, frame_ids: tuple[int, ...]) -> None:
    images = root / "JPEGImages" / video_id
    annotations = root / "Annotations" / video_id
    images.mkdir(parents=True)
    annotations.mkdir(parents=True)
    for frame_id in frame_ids:
        Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(images / f"{frame_id:08d}.jpg")
        Image.fromarray(np.array([[1, 2], [0, 0]], dtype=np.uint8)).save(
            annotations / f"{frame_id:08d}.png"
        )


def _manifest(video_id: str, prompts: dict[int, int]) -> dict:
    return {
        "schema_version": "cmmt.lvosv2_evaluation_manifest.v1",
        "split": "val",
        "cases": [
            {"video_id": video_id, "object_id": object_id, "first_prompt_frame": frame_id}
            for object_id, frame_id in prompts.items()
        ],
    }


class LvosBaseRoundtripTests(unittest.TestCase):
    def test_hard_video_selection_is_included(self) -> None:
        self.assertEqual(len(VIDEO_IDS), 14)
        self.assertEqual(VIDEO_IDS[:4], ("2VegYEbT", "2urlAsm8", "0tCWPOrc", "9HEh93ef"))
        self.assertIn("x3nD3QQ9", VIDEO_IDS)

    def test_overlapping_logits_have_one_winning_object(self) -> None:
        logits = torch.tensor([
            [[[2.0, -1.0], [0.5, -2.0]]],
            [[[1.0, 3.0], [0.5, -3.0]]],
        ])
        by_id, binary = masks_by_id([4, 7], logits)
        self.assertEqual(set(by_id), {4, 7})
        self.assertEqual(binary[4].tolist(), [[True, False], [True, False]])
        self.assertEqual(binary[7].tolist(), [[False, True], [False, False]])

    def test_prompt_exclusion_and_post_switch_delta(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation = root / "1.png"
            Image.fromarray(np.array([[1, 2], [0, 255]], dtype=np.uint8)).save(annotation)
            clip = Clip(
                video_id="test", frames=(root / "1.jpg",) * 4,
                annotations=(annotation,) * 4,
                original_frame_ids=(1, 6, 11, 16),
                prompts=(Prompt(1, 0, 1), Prompt(2, 1, 6)),
                switch_index=2,
            )
            perfect = torch.tensor([
                [[[2.0, -2.0], [-2.0, -2.0]]],
                [[[-2.0, 2.0], [-2.0, -2.0]]],
            ])

            def iou(gt: np.ndarray, pred: np.ndarray, *, void_pixels: np.ndarray) -> float:
                valid = ~void_pixels
                intersection = np.logical_and(gt, pred) & valid
                union = np.logical_or(gt, pred) & valid
                return float(intersection.sum() / union.sum()) if union.any() else 1.0

            native = score_frame(clip, 3, [1, 2], perfect, iou, iou)
            self.assertEqual(native[0][1]["JF"], 1.0)
            self.assertEqual(native[0][2]["JF"], 1.0)
            transferred = score_frame(clip, 3, [1, 2], perfect.clone(), iou, iou)
            rows = make_frame_rows(clip, 3, native, transferred)
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["binary_equal"] and row["delta_JF"] == 0 for row in rows))
            self.assertTrue(all(row["phase"] == "post_switch" for row in rows))
            self.assertTrue(all(row["included_in_mean"] for row in rows))
            self.assertFalse(make_frame_rows(clip, 1, native, None)[1]["included_in_mean"])
            write_svg(clip, rows, root / "curve.svg")
            self.assertIn("Transferred", (root / "curve.svg").read_text())

    def test_load_clip_maps_original_prompt_ids_to_clip_indices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_sparse_clip(root, "vid", (1, 6, 11, 16))
            clip = load_clip(
                root, _manifest("vid", {1: 1, 2: 6}), "vid", frame_count=4, switch_index=2
            )
            self.assertEqual(clip.original_frame_ids, (1, 6, 11, 16))
            self.assertEqual(
                [(p.object_id, p.frame_idx, p.original_frame_id) for p in clip.prompts],
                [(1, 0, 1), (2, 1, 6)],
            )

    def test_load_clip_rejects_prompt_after_switch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_sparse_clip(root, "vid", (1, 6, 11, 16))
            with self.assertRaisesRegex(ValueError, "prompted after switch"):
                load_clip(
                    root, _manifest("vid", {1: 1, 2: 16}), "vid", frame_count=4, switch_index=2
                )


if __name__ == "__main__":
    unittest.main()
