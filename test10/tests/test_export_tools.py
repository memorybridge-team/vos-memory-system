"""Cost-only extension decisions and portable evidence exports."""
import importlib.util
from pathlib import Path
import unittest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def module(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.mod = module("extend_frozen_schedule")
        self.cases = [dict(case_id=f"{d}:{slot}", dataset=d, slot=slot,
                           cohort="heldout_extension", length_bin=slot % 3)
                      for slot in range(35, 40) for d in self.mod.DATASETS]
        self.timings = [dict(dataset=d, length_bin=b, seconds=10)
                        for d in self.mod.DATASETS for b in range(3)]

    def test_margin_and_reserve_boundary(self):
        _, low = self.mod.extension_plan(self.cases, self.timings, 2039)
        plan, enough = self.mod.extension_plan(self.cases, self.timings, 2040)
        self.assertFalse(low["admitted"])
        self.assertTrue(enough["admitted"])
        self.assertEqual(len(plan), 20)

    def test_fixed_round_robin_ignores_accuracy(self):
        plan, decision = self.mod.extension_plan(list(reversed(self.cases)), self.timings, 5000)
        self.assertEqual([r["case_id"].split(":")[0] for r in plan], list(self.mod.DATASETS) * 5)
        with_scores = [dict(r, scores={"J_and_F": 999}) for r in self.timings]
        self.assertEqual((plan, decision), self.mod.extension_plan(self.cases, with_scores, 5000))

    def test_no_partial_extension_admission(self):
        with self.assertRaises(AssertionError):
            self.mod.extension_plan(self.cases[:-1], self.timings, 5000)


class ExportTests(unittest.TestCase):
    def test_paths_are_portable_without_losing_identity(self):
        mod = module("export_fit_experiment")
        value = {str(mod.ROOT / "runs/example"): [str(mod.WORKSPACE / "vos-data"),
                 "/mnt/c/Users/Home/runpod-state-pairs/MOSEv2/fit/example.pt"]}
        expected = {"${TEST10_ROOT}/runs/example": ["${WORKSPACE_ROOT}/vos-data",
                    "${STATE_PAIR_ROOT}/MOSEv2/fit/example.pt"]}
        self.assertEqual(mod.portable(value), expected)


if __name__ == "__main__":
    unittest.main()
