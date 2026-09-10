import json
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image

from datasets import Video
from runner import MaskAggregator, experiment_plan, find_prompts, run_video


class FakePredictor:
    def __init__(self):
        self.initial_non_conditioning_keys = []

    def init_state(self, video_path, **kwargs):
        frames = sorted(Path(video_path).glob("*.jpg"))
        with Image.open(frames[0]) as image:
            height, width = image.height, image.width
        return {"num_frames": len(frames), "video_height": height,
                "video_width": width, "obj_ids": [],
                "output_dict_per_obj": {}}

    def reset_state(self, state):
        state["obj_ids"] = []
        state["output_dict_per_obj"] = {}

    def add_new_mask(self, state, frame_idx, obj_id, mask):
        state["obj_ids"] = [obj_id]
        logits = FakeTensor(np.where(mask, 10.0, -10.0)[None, None])
        state["output_dict_per_obj"] = {0: {
            "cond_frame_outputs": {frame_idx: {"pred_masks": logits}},
            "non_cond_frame_outputs": {},
        }}

    def propagate_in_video(self, state, start_frame_idx, reverse=False):
        outputs = state["output_dict_per_obj"][0]
        self.initial_non_conditioning_keys.append(
            tuple(sorted(outputs["non_cond_frame_outputs"])))
        obj_id = state["obj_ids"][0]
        prompt = outputs["cond_frame_outputs"][start_frame_idx]["pred_masks"]
        for frame in range(start_frame_idx, state["num_frames"]):
            if frame == start_frame_idx:
                logits = prompt
            else:
                if hasattr(outputs["non_cond_frame_outputs"], "query_frame"):
                    outputs["non_cond_frame_outputs"].query_frame = frame
                outputs["non_cond_frame_outputs"].get(frame - 1)
                logits = prompt.clone()
                outputs["non_cond_frame_outputs"][frame] = {"pred_masks": logits}
            yield frame, [obj_id], logits


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    def __getitem__(self, key):
        return FakeTensor(self.value[key])

    def clone(self):
        return FakeTensor(self.value.copy())

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeTorch(types.ModuleType):
    def __init__(self):
        super().__init__("torch")
        self.cuda = types.SimpleNamespace(max_memory_allocated=lambda: 0)

    @staticmethod
    def inference_mode():
        return nullcontext()


class RunnerTests(unittest.TestCase):
    def test_max_logit_overlap_uses_lowest_id_on_tie(self):
        with tempfile.TemporaryDirectory() as tmp:
            aggregate = MaskAggregator(tmp, ("cold",))
            aggregate.update("cold", 0, 2, np.ones((2, 2)))
            aggregate.update("cold", 0, 3, np.ones((2, 2)))
            labels = np.load(Path(tmp) / "cold/00000000.label.npy")
            self.assertTrue(np.all(labels == 2))

    def test_cold_and_full_memory_recompute_and_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames, masks = [], {}
            for frame in range(3):
                frame_path = root / f"{frame:05d}.jpg"
                Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(frame_path)
                frames.append(frame_path)
                mask = np.zeros((8, 8), np.uint8)
                mask[2:6, 2:6] = 1
                mask_path = root / f"{frame:05d}.png"
                Image.fromarray(mask).save(mask_path)
                masks[frame] = mask_path
            video = Video("clip", frames, [0, 1, 2], masks, {}, True)
            config = {"offload_video_to_cpu": True, "offload_state_to_cpu": True,
                      "device": "cpu", "precision": "float32", "max_videos": None,
                      "conditions": ["cold", "full_memory"]}
            prior = sys.modules.get("torch")
            sys.modules["torch"] = FakeTorch()
            try:
                predictor = FakePredictor()
                result, resumed = run_video(predictor, video, root / "out",
                                            config, {"test": True})
            finally:
                if prior is None:
                    sys.modules.pop("torch", None)
                else:
                    sys.modules["torch"] = prior
            self.assertFalse(resumed)
            self.assertEqual(result["emitted_frames"]["full_memory"], 3)
            self.assertEqual(predictor.initial_non_conditioning_keys,
                             [(), (1, 2)])
            audit = result["full_memory_audit"][0]
            self.assertEqual(audit["second_pass_writes"], 2)
            self.assertEqual(audit["expected_second_pass_writes"], 2)
            self.assertTrue(audit["all_general_frames_recomputed"])
            self.assertEqual(audit["overwritten_injected_records"], 2)
            self.assertEqual(audit["removed_injected_records"], 0)
            self.assertEqual(audit["remaining_injected_records"], 0)
            self.assertEqual(audit["unique_memory_hits"]["first_pass"], 0)
            self.assertEqual(audit["unique_memory_hits"]["second_pass"], 1)
            self.assertFalse(any(event["is_future"] for event in audit["events"]))
            self.assertEqual(result["memory"][0]["injected_non_conditioning_records"], 2)
            self.assertEqual(result["memory"][0]["records_after_full_memory"], 3)
            self.assertEqual(result["full_memory_minus_cold"]["j_and_f"], 0.0)
            sys.modules["torch"] = FakeTorch()
            try:
                _, resumed = run_video(FakePredictor(), video, root / "out",
                                       config, {"test": True})
            finally:
                if prior is None:
                    sys.modules.pop("torch", None)
                else:
                    sys.modules["torch"] = prior
            self.assertTrue(resumed)
            next((root / "out/cold/masks").glob("*.png")).unlink()
            sys.modules["torch"] = FakeTorch()
            try:
                _, resumed = run_video(FakePredictor(), video, root / "out",
                                       config, {"test": True})
            finally:
                if prior is None:
                    sys.modules.pop("torch", None)
                else:
                    sys.modules["torch"] = prior
            self.assertFalse(resumed)
            config["save_masks"] = False
            sys.modules["torch"] = FakeTorch()
            try:
                run_video(FakePredictor(), video, root / "no-masks", config, {"test": True})
                self.assertFalse(list((root / "no-masks").rglob("*.png")))
                _, resumed = run_video(FakePredictor(), video, root / "no-masks", config, {"test": True})
                self.assertTrue(resumed)
            finally:
                if prior is None:
                    sys.modules.pop("torch", None)
                else:
                    sys.modules["torch"] = prior

    def test_default_plan_contains_all_sixteen_combinations(self):
        plan = experiment_plan(Path(__file__).with_name("benchmark.json"))
        self.assertFalse(plan["downloads_started"])
        self.assertEqual(len(plan["matrix"]), 16)
        self.assertEqual({tuple(row["conditions"]) for row in plan["matrix"]},
                         {("cold", "full_memory")})


if __name__ == "__main__":
    unittest.main()
