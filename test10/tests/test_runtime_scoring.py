"""Regression tests for per-offset scoring without changing handoff context."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import ensure_local_training_module
ensure_local_training_module()
import runtime


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.case = dict(case_id="case", switch=16, frame_stems=[str(i) for i in range(40)],
                         annotation_dir="unused", object_id=1)

    def check_payload(self, positions, expected_switch):
        result = {"post_switch": {"frames": len(positions), "J_and_F": 1}}
        masks = {i: "packed-mask" for i in positions}
        with patch("runtime.Path.is_file", return_value=True), patch("runtime.score", return_value={"scores": result}) as scorer:
            self.assertEqual(runtime.score_positions(self.case, "affine", masks), result)
        payload = scorer.call_args.args[0]
        self.assertEqual(payload["switch_position"], expected_switch)
        self.assertEqual(payload["context_position"], 16)
        self.assertEqual(payload["stems_context"], "16")
        self.assertEqual(payload["masks"], masks)

    def test_single_later_offset_has_local_scoring_anchor(self):
        self.check_payload([21], 20)

    def test_window_keeps_original_scoring_anchor(self):
        self.check_payload(list(range(17, 27)), 16)

    def test_missing_annotation_is_not_zero_quality(self):
        with patch("runtime.Path.is_file", return_value=False), patch("runtime.score") as scorer:
            result = runtime.score_positions(self.case, "affine", {21: "packed-mask"})
        scorer.assert_not_called()
        self.assertEqual(result["post_switch"]["frames"], 0)
        self.assertIsNone(result["post_switch"]["J_and_F"])


if __name__ == "__main__":
    unittest.main()
