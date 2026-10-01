"""CPU tests for handoff_diagnostics with fake caches; no SAM 2, torch or test9 import."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import Artifacts, GATE, write
import handoff_diagnostics as hd

METHODS = ("small_only", "base_native", "affine", "residual_mlp", "last_mask")


def packbits(mask):
    mask = np.asarray(mask, bool)
    return tuple(mask.shape), np.packbits(mask.ravel()).tobytes()


def fake_case(cid, video, switch=3, end=12, dataset="LVOSv2"):
    return dict(case_id=cid, dataset=dataset, video_id=video, object_id=1, first=0, switch=switch, end=end,
                frame_stems=[f"{i:05d}" for i in range(end + 1)], input_sha256="rgb", annotation_sha256="ann",
                sampling={"raw_stride": 5}, cohort="heldout_core", checkpoint_video=False, length_bin=0,
                annotation_dir="${WORKSPACE_ROOT}/vos-data/x")


def square(offset):
    mask = np.zeros((8, 8), bool)
    mask[offset:offset + 4, 2:6] = True
    return mask


class Helpers(unittest.TestCase):
    def test_restore_placeholder_nested_and_non_ascii(self):
        value = {"files": {"${WORKSPACE_ROOT}/a.py": "h"}, "dirs": ["${WORKSPACE_ROOT}/b"], "n": 1}
        out = hd.restore(value, Path("/tmp/종프2 \\x"))
        self.assertEqual(out, {"files": {"/tmp/종프2 \\x/a.py": "h"}, "dirs": ["/tmp/종프2 \\x/b"], "n": 1})

    def test_iou_rules(self):
        self.assertEqual(hd.iou(np.zeros((2, 2)), np.zeros((2, 2))), 1.)
        self.assertEqual(hd.iou(square(0), square(0)), 1.)
        self.assertAlmostEqual(hd.iou(square(0), square(2)), 8 / 24)
        with self.assertRaises(ValueError):
            hd.iou(np.zeros((2, 2)), np.zeros((2, 3)))

    def test_decoder_is_proven_against_pack(self):
        decode = hd.mask_decoder(types.SimpleNamespace(pack=packbits))
        self.assertTrue(np.array_equal(decode(packbits(square(1))), square(1)))
        little = types.SimpleNamespace(pack=lambda m: (m.shape, np.packbits(np.asarray(m, bool).ravel(), bitorder="little").tobytes()))
        self.assertTrue(np.array_equal(hd.mask_decoder(little)(little.pack(square(3))), square(3)))
        own = types.SimpleNamespace(pack=lambda m: {"m": np.asarray(m).tolist()},
                                    unpack_mask=lambda p: np.array(p["m"], bool))
        self.assertEqual(hd.mask_decoder(own), own.unpack_mask)
        with self.assertRaises(RuntimeError):
            hd.mask_decoder(types.SimpleNamespace(pack=lambda m: b"opaque"))


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name)
        self.cases = [fake_case("c1", "v1"), fake_case("c2", "v2"), fake_case("c3", "v3", end=6)]
        self.cases[2]["dataset"] = "MOSEv2"
        write(self.run_dir / "selection.json", {"cases": self.cases, "methods": list(METHODS), "seed": 7})
        prov = {"files": {"${WORKSPACE_ROOT}/x.py": "hash"}}
        write(self.run_dir / "provenance.json", prov)
        # Rows are keyed with the restored provenance, as on the original workspace.
        store = Artifacts(self.run_dir, hd.restore(prov, Path("/ws")))
        for c in self.cases:
            for m in METHODS:
                store.save(c, m, origin="executed", scores={})
            store.save(c, GATE, gate_passed=True)
        self.calls = []

    def tearDown(self):
        self.tmp.cleanup()

    def blobs(self, case, name):
        suffix = range(case["switch"] + 1, case["end"] + 1)
        # Base+ always square(0); affine matches it; MLP shifted by 2 after +2; last_mask empty.
        # Case c2 disagrees at the switch frame (Small last mask shifted).
        if name == "source_prefix":
            last = square(4) if case["case_id"] == "c2" else square(0)
            return {"masks": {p: packbits(square(0)) for p in suffix}, "last_mask": last}
        if name == "base_prefix":
            return {"masks": {p: packbits(square(0)) for p in suffix}, "last_mask": square(0)}
        method = name.removesuffix("_predictions")
        make = {"affine": lambda p: square(0),
                "residual_mlp": lambda p: square(2) if p - case["switch"] > 2 else square(0),
                "last_mask": lambda p: np.zeros((8, 8), bool)}[method]
        return {"masks": {p: packbits(make(p)) for p in suffix}}

    def score(self, case, method, masks):
        self.calls.append((case["case_id"], method, sorted(masks)))
        value = 1. if method != "last_mask" else 0.
        return {"post_switch": {"frames": len(masks), "J": value, "F": value, "J_and_F": value}}

    def run_main(self, **kw):
        args = argparse.Namespace(run_dir=self.run_dir, early_frames=kw.get("early", 6), switch_agree=.9,
                                  seed=7, workspace_root=Path("/ws"))
        return hd.main(args, load=self.blobs, decode=hd.mask_decoder(types.SimpleNamespace(pack=packbits)),
                       score=self.score)

    def test_early_window_agreement_and_outputs(self):
        report = self.run_main()
        # GT scorer only sees +1..+6, or the shorter suffix for c3.
        self.assertIn(("c1", "affine", [4, 5, 6, 7, 8, 9]), self.calls)
        self.assertIn(("c3", "affine", [4, 5, 6]), self.calls)
        rows = {(r["case_id"], r["method"]): r for r in report["rows"]}
        self.assertIsNone(rows[("c1", "base_native")]["agree_early"])
        self.assertEqual(rows[("c1", "affine")]["agree_full"], 1.)
        self.assertEqual(rows[("c1", "last_mask")]["agree_early"], 0.)
        # MLP: +1,+2 identical, +3..+6 IoU 8/24 -> early mean; suffix has 9 frames.
        self.assertAlmostEqual(rows[("c1", "residual_mlp")]["agree_early"], (2 + 4 * 8 / 24) / 6)
        self.assertAlmostEqual(rows[("c1", "residual_mlp")]["agree_full"], (2 + 7 * 8 / 24) / 9)
        self.assertEqual(rows[("c1", "small_only")]["switch_iou"], 1.)
        self.assertAlmostEqual(rows[("c2", "small_only")]["switch_iou"], 0.)
        subgroup = report["groups"]["switch_iou>=0.9"]["LVOSv2"]
        self.assertEqual((subgroup["cases"], report["groups"]["all"]["LVOSv2"]["cases"]), (1, 2))
        pair = report["groups"]["all"]["LVOSv2"]["paired"]["affine-minus-last_mask"]["early_JF"]
        self.assertEqual(pair["mean"], 1.)
        self.assertTrue((self.run_dir / "handoff_diagnostics.json").is_file())
        self.assertIn("IoU vs Base+", (self.run_dir / "handoff_diagnostics.md").read_text(encoding="utf-8"))
        json.loads((self.run_dir / "handoff_diagnostics.json").read_text(encoding="utf-8"))

    def test_wrong_workspace_root_is_explicit(self):
        args = argparse.Namespace(run_dir=self.run_dir, early_frames=6, switch_agree=.9, seed=7,
                                  workspace_root=Path("/elsewhere"))
        with self.assertRaisesRegex(ValueError, "workspace-root"):
            hd.main(args, load=self.blobs, decode=lambda x: x, score=self.score)

    def test_missing_cache_and_suffix_mismatch_fail(self):
        with self.assertRaisesRegex(FileNotFoundError, "prefix cache"):
            hd.case_rows(self.cases[0], METHODS, lambda c, n: None, None, self.score, 6)
        def short(case, name):
            blob = self.blobs(case, name)
            if name == "affine_predictions":
                blob["masks"].pop(case["end"])
            return blob
        with self.assertRaisesRegex(ValueError, "suffix frame mismatch"):
            hd.case_rows(self.cases[0], METHODS, short, hd.mask_decoder(types.SimpleNamespace(pack=packbits)), self.score, 6)


if __name__ == "__main__":
    unittest.main()
