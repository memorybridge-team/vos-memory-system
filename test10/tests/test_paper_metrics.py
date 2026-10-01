import importlib.util
from pathlib import Path
import sys
import unittest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("paper_metrics", TOOLS / "report_paper_metrics.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class PaperMetricTests(unittest.TestCase):
    def test_average_of_ratios_not_ratio_of_means(self):
        result = mod.ratio_summary([mod.ratio(.5, .1), mod.ratio(.5, .9)])
        self.assertAlmostEqual(result["mean"], (500 + 100*.5/.9)/2)
        self.assertNotEqual(result["mean"], 100)

    def test_undefined_ratios_and_negative_gap(self):
        self.assertIsNone(mod.ratio(0, 0))
        self.assertIsNone(mod.gap_ratio(.5, .3, .3))
        self.assertEqual(mod.gap_ratio(.3, .3, .1), 0)
        self.assertAlmostEqual(mod.gap_ratio(.1, .3, .1), 100)
        self.assertEqual(mod.ratio_summary([None, 100])["undefined_cases"], 1)

    def test_void_and_strict_id_swap(self):
        import numpy as np
        labels = np.array([[1, 2, 255], [0, 0, 255]])
        pred = np.array([[False, True, True], [False, False, True]])
        self.assertTrue(mod.is_id_swap(pred, labels, 1))
        self.assertFalse(mod.is_id_swap(np.zeros_like(pred), labels, 1))
        self.assertEqual(mod.iou(pred, labels == 2, labels == 255), 1)

    def test_offset_and_length_boundaries(self):
        self.assertEqual([mod.drift_bin(i) for i in (5, 6, 10, 11, 20, 21, 50, 51, 100, 101)],
                         ["1-5", "6-10", "6-10", "11-20", "11-20", "21-50", "21-50", "51-100", "51-100", "101+"])
        self.assertEqual(mod.length_bin(16), "1-16")
        self.assertEqual(mod.length_bin(17), "17-64")

    def test_shared_boundary_matches_original_davis_including_void(self):
        import numpy as np
        sys.path.insert(0, "/home/home/test/test9/vos-memory-translator-nonlinear/src")
        from vos_memory_inspector._vendor.davis2017_metrics import db_eval_boundary
        rng = np.random.default_rng(7)
        for _ in range(8):
            gt, pred, void = (rng.random((31, 43)) > threshold for threshold in (.7, .7, .9))
            self.assertEqual(mod.boundary_f(pred, void, mod.boundary_context(gt, void)), float(db_eval_boundary(gt, pred, void)))


if __name__ == "__main__":
    unittest.main()
