"""Opt-in real SAM2 smoke test.

Run with SAM2_INTEGRATION=1 and SAM2_TINY_CHECKPOINT=/path/to/checkpoint.
It is skipped in ordinary CPU-only unit test runs.
"""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from datasets import Video
from models import build_predictor
from runner import run_video


@unittest.skipUnless(os.environ.get("SAM2_INTEGRATION") == "1",
                     "set SAM2_INTEGRATION=1 to run the real model")
class RealSAM2Integration(unittest.TestCase):
    def test_tiny_two_pass_synthetic_video(self):
        checkpoint = os.environ["SAM2_TINY_CHECKPOINT"]
        repo = os.environ.get("SAM2_REPO", str(Path(__file__).parent / "vendor/sam2"))
        device = os.environ.get("SAM2_DEVICE", "cuda")
        predictor, metadata = build_predictor("tiny", checkpoint, repo, device)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames, masks = [], {}
            for frame in range(3):
                rgb = np.zeros((64, 64, 3), np.uint8)
                rgb[16:48, 16+frame:48+frame] = 255
                frame_path = root / f"{frame:05d}.jpg"
                Image.fromarray(rgb).save(frame_path)
                frames.append(frame_path)
                gt = np.zeros((64, 64), np.uint8)
                gt[16:48, 16+frame:48+frame] = 1
                mask_path = root / f"{frame:05d}.png"
                Image.fromarray(gt).save(mask_path)
                masks[frame] = mask_path
            video = Video("synthetic", frames, [0, 1, 2], masks, {}, True)
            config = {"offload_video_to_cpu": True, "offload_state_to_cpu": True,
                      "device": device, "precision": "bfloat16" if device.startswith("cuda") else "float32",
                      "max_videos": 1, "conditions": ["cold", "full_memory"]}
            result, _ = run_video(predictor, video, root / "result", config,
                                  {"integration": True, **metadata})
            self.assertEqual(result["emitted_frames"], {"cold": 3, "full_memory": 3})
            full = result["full_memory_audit"][0]
            self.assertEqual(full["second_pass_writes"], 2)
            self.assertEqual(full["expected_second_pass_writes"], 2)
            self.assertTrue(full["all_general_frames_recomputed"])
            self.assertEqual(full["overwritten_injected_records"], 2)
            self.assertEqual(full["remaining_injected_records"], 0)
            self.assertEqual(full["unique_memory_hits"]["first_pass"], 0)
            self.assertTrue(all(event["requested_frame"] == event["memory_frame"]
                                for event in full["events"]))
            self.assertFalse(any(event["is_future"] for event in full["events"]))
            self.assertEqual(result["memory"][0]["injected_non_conditioning_records"], 2)
            cold = [np.asarray(Image.open(path)) for path in
                    sorted((root / "result/cold/masks").glob("*.png"))]
            repeated, _ = run_video(predictor, video, root / "repeat", config,
                                    {"integration": "repeat", **metadata})
            repeated_cold = [np.asarray(Image.open(path)) for path in
                             sorted((root / "repeat/cold/masks").glob("*.png"))]
            self.assertTrue(all(np.array_equal(a, b) for a, b in zip(cold, repeated_cold)))
            self.assertEqual(result["metrics"]["cold"]["j_and_f"],
                             repeated["metrics"]["cold"]["j_and_f"])


if __name__ == "__main__":
    unittest.main()
