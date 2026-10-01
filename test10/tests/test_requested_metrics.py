"""No GPU: requested metric definitions and unbiased population contracts."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("requested_metrics", Path(__file__).resolve().parents[1] / "tools/report_requested_metrics.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class RequestedMetricTests(unittest.TestCase):
    def test_retention_is_ratio_not_gap_recovery(self):
        self.assertEqual(mod.retention(.8, .9), 100 * .8 / .9)
        self.assertIsNone(mod.retention(.8, 0))
        self.assertIsNone(mod.retention(.8, None))
        self.assertGreater(mod.retention(.95, .9), 100)

    def test_absent_perfect_does_not_inflate_visible(self):
        frames = [dict(J=.2, F=.4, J_and_F=.3, gt_present=True),
                  dict(J=1, F=1, J_and_F=1, gt_present=False)]
        self.assertAlmostEqual(mod.frame_mean(frames)["J_and_F"], .65)
        self.assertEqual(mod.frame_mean(frames, True)["J_and_F"], .3)
        self.assertIsNone(mod.frame_mean(frames[1:], True)["J_and_F"])

    def test_source_only_has_no_target_frames(self):
        self.assertEqual(mod.target_counts("small_only", 100, 5, 4),
                         dict(target_frames_processed=0, target_frames_scored_at5=0, target_visible_frames_scored_at5=0))
        self.assertEqual(mod.target_counts("affine", 100, 5, 4)["target_frames_processed"], 100)

    def test_video_weighting_is_not_pair_weighting(self):
        rows = [dict(video_id=v, metrics=dict(post5=dict(J_and_F=value)))
                for v, value in (("a", 0), ("a", 0), ("b", 1))]
        values = mod.video_values(rows, "post5")
        self.assertEqual(sum(values.values()) / len(values), .5)

    def test_partial_method_groups_excluded_and_duplicates_rejected(self):
        rows = [dict(case_id="complete", method=m) for m in mod.BANK_METHODS]
        rows.append(dict(case_id="partial", method="affine"))
        self.assertEqual(set(mod.complete_groups(rows, mod.BANK_METHODS)), {"complete"})
        with self.assertRaises(ValueError):
            mod.complete_groups(rows + rows[:1], mod.BANK_METHODS)

    def test_retention_pairs_same_video_population(self):
        rows = [dict(method=m, video_id=v, metrics=dict(post5=dict(J_and_F=x))) for m, v, x in
                (("affine", "shared", .8), ("base_native", "shared", .9), ("base_native", "unmatched", .1))]
        out = mod.matched_retention(rows, "affine", "post5")
        self.assertEqual(out["matched_videos"], 1)
        self.assertAlmostEqual(out["percent"], 100 * .8 / .9)


if __name__ == "__main__":
    unittest.main()
