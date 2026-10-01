"""Real example-format checks and synthetic optimizer/checkpoint integration on CPU."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import ROOT, Artifacts, write, read, suffix_scores
from state_pairs import load_pair, save_pair, validate_pair, active_state, SCHEMA, MODELS
import training
import pair_catalog
import torch
from vos_memory_inspector.state_schema import CanonicalState
from vos_memory_inspector.upstream import SUPPORTED_SAM2_COMMIT


def synthetic_pair(path, video):
    state = CanonicalState(
        spatial_memory=torch.zeros(1, 1, 1, 64, 64, 64, dtype=torch.bfloat16),
        object_pointer=torch.zeros(1, 1, 1, 256), presence_logits=torch.zeros(1, 1, 1, 1),
        frame_indices=torch.zeros(1, 1, 1, dtype=torch.int64),
        slot_order=torch.zeros(1, 1, 1, dtype=torch.int64),
        is_conditioning=torch.ones(1, 1, 1, dtype=torch.bool),
        validity=torch.ones(1, 1, 1, dtype=torch.bool), object_ids=(1,), switch_frame=0)
    target = replace(state, spatial_memory=state.spatial_memory + .125,
                     object_pointer=state.object_pointer + .125)
    meta = dict(MODELS, upstream_commit=SUPPORTED_SAM2_COMMIT, video_id=video,
                object_id=1, switch_frame=0, num_frames=11, seed=7, cache_mode="state_only",
                active_memory_only=True, num_maskmem=7, max_obj_ptrs_in_encoder=16)
    save_pair(path, active_state(state), active_state(target), meta)


class PairContract(unittest.TestCase):
    def test_evaluation_pair_cache_is_v2_and_is_reused_as_the_state_source(self):
        import runtime
        source, target, metadata = load_pair(next((ROOT / "state_pair_examples/MOSEv2").glob("*.pt")))
        case = dict(case_id="pair", dataset="MOSEv2", video_id=metadata["video_id"], object_id=1,
                    first=0, switch=4, end=25, frame_stems=list(map(str, range(26))),
                    sampling={"raw_stride": 1}, input_sha256="rgb", annotation_sha256="gt",
                    cohort="additional", checkpoint_video=False)
        evaluator = runtime.Evaluator.__new__(runtime.Evaluator)
        evaluator.seed = 7
        evaluator.small = evaluator.base = SimpleNamespace(num_maskmem=7, max_obj_ptrs_in_encoder=16)
        with tempfile.TemporaryDirectory() as tmp:
            store = Artifacts(tmp, {})
            with patch.object(evaluator, "prefix", side_effect=[dict(state=source), dict(state=target)]) as prefix:
                small, base = evaluator.paired_prefix(case, [None] * 26, store)
                self.assertTrue(all(c.kwargs["need_state"] for c in prefix.call_args_list))
            path = store.path(case, "state_pair", ".pt")
            load_pair(path)
            self.assertTrue(path.with_suffix(".prepare.json").exists())
            with patch.object(evaluator, "prefix", side_effect=[{}, {}]) as prefix:
                small, base = evaluator.paired_prefix(case, [None] * 26, store)
                self.assertTrue(all(not c.kwargs["need_state"] for c in prefix.call_args_list))
            torch.testing.assert_close(small["state"].spatial_memory, source.spatial_memory)
            torch.testing.assert_close(base["state"].object_pointer, target.object_pointer)

    def test_both_real_examples_load_with_their_checksum_and_sidecars(self):
        paths = sorted((ROOT / "state_pair_examples").glob("*/*.pt"))
        self.assertEqual(len(paths), 2)
        counts = []
        for path in paths:
            source, target, meta = load_pair(path)
            self.assertEqual(source.spatial_memory.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(source.frame_indices, target.frame_indices))
            self.assertEqual(meta["switch_frame"], source.switch_frame)
            counts.append(source.valid_record_count())
        self.assertEqual(sorted(counts), [5, 16])

    def test_active_selection_keeps_conditioning_plus_fifteen_recent_records(self):
        source, _, _ = load_pair(next((ROOT / "state_pair_examples/MOSEv2").glob("*.pt")))
        tensors = {}
        for name in ("spatial_memory", "object_pointer", "presence_logits"):
            old = getattr(source, name)
            tensors[name] = old[:, :, :1].repeat(1, 1, 20, *([1] * (old.ndim - 3)))
        frames = torch.arange(20).reshape(1, 1, 20)
        state = replace(source, **tensors, frame_indices=frames, slot_order=frames,
            validity=torch.ones_like(frames, dtype=torch.bool),
            is_conditioning=(frames == 0), switch_frame=19)
        selected = active_state(state)
        self.assertEqual(selected.frame_indices.flatten().tolist(), [0, *range(5, 20)])
        self.assertEqual(selected.slot_order.flatten().tolist(), list(range(16)))

    def test_timeline_mismatch_empty_records_and_old_schema_rejected(self):
        source, target, meta = load_pair(next((ROOT / "state_pair_examples/MOSEv2").glob("*.pt")))
        payload = dict(schema_version=SCHEMA, source_canonical=source, target_canonical=target, metadata=meta)
        with self.assertRaisesRegex(ValueError, "expected"):
            validate_pair(dict(payload, schema_version="old"))
        with self.assertRaisesRegex(ValueError, "frame_indices"):
            validate_pair(dict(payload, target_canonical=replace(target, frame_indices=target.frame_indices + 1)))
        empty = torch.zeros_like(source.validity)
        with self.assertRaisesRegex(ValueError, "no valid"):
            validate_pair(dict(payload, source_canonical=replace(source, validity=empty),
                               target_canonical=replace(target, validity=empty)))

    def test_corrupt_pair_and_mismatched_prepare_sidecar_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pair.pt"
            synthetic_pair(path, "a")
            side = path.with_suffix(".prepare.json")
            info = read(side); info["switch_frame"] = 1; write(side, info)
            with self.assertRaisesRegex(ValueError, "prepare metadata"):
                load_pair(path)
            path.write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_pair(path)


class FreshTraining(unittest.TestCase):
    def test_collection_preserves_selection_checksums_in_eager_and_budget_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for video, split in (("train", "train"), ("validation", "validation")):
                path = root / f"{video}.pt"
                synthetic_pair(path, video)
                checksum = path.with_suffix(".pt.sha256").read_text().split()[0]
                rows.append(dict(dataset="synthetic", video_id=video, split=split,
                                 path=path.name, sha256=checksum))
            selection = root / "pairs.json"
            write(selection, dict(schema="test10.pair_selection.v1", pairs=rows))
            for lazy in (False, True):
                with self.subTest(lazy=lazy):
                    result = training.collection(selection, lazy=lazy)
                    self.assertEqual([r["sha256"] for r in result["pairs"]],
                                     [r["sha256"] for r in rows])
                    reader = training.PairReader()
                    reader.load(result["pairs"][0])
                    changed = [dict(rows[0], sha256="0" * 64), rows[1]]
                    write(selection, dict(schema="test10.pair_selection.v1", pairs=changed))
                    with self.assertRaisesRegex(ValueError, "checksum"):
                        training.collection(selection, lazy=lazy)
                    write(selection, dict(schema="test10.pair_selection.v1", pairs=rows))

    def test_downloaded_pair_catalog_uses_fit_and_development_without_copying(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dataset, prefix in pair_catalog.DATASETS.items():
                for source_split in pair_catalog.SPLITS:
                    video = f"{prefix}_{source_split}"
                    directory = root / dataset / source_split
                    directory.mkdir(parents=True)
                    synthetic_pair(directory / f"{video}.pt", video)
                    write(root / "manifests" / f"{prefix}_train_v1_{source_split}.json", dict(
                        schema_version="cmmt.video_split_manifest.v1",
                        dataset="LVOS v2" if dataset == "LVOSv2" else dataset,
                        split=source_split, videos=[video], video_count=1, case_count=1))
            selection = pair_catalog.build(root)
            self.assertEqual(len(selection["pairs"]), 4)
            self.assertEqual({row["split"] for row in selection["pairs"]}, {"train", "validation"})
            self.assertTrue(all(Path(row["path"]).is_file() for row in selection["pairs"]))
            write(root / "pairs.json", selection)
            self.assertEqual(len(training.collection(root / "pairs.json")["pairs"]), 4)
            manifest = root / "manifests/mosev2_train_v1_fit.json"
            info = read(manifest); info["videos"] = ["wrong_video"]; write(manifest, info)
            with self.assertRaisesRegex(ValueError, "outside split"):
                pair_catalog.build(root)

    def test_synthetic_optimizer_checkpoint_and_evaluation_loader_roundtrip(self):
        # A one-step synthetic integration check; does not train on the example videos.
        old_threads = torch.get_num_threads(); torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for video in ("train_video", "validation_video"):
                    synthetic_pair(root / f"{video}.pt", video)
                selection = root / "pairs.json"
                rows = [dict(dataset="synthetic", video_id=v, split=s, path=f"{v}.pt")
                        for v, s in (("train_video", "train"), ("validation_video", "validation"))]
                write(selection, dict(schema="test10.pair_selection.v1", pairs=rows))
                args = training.parser().parse_args(["--pairs", str(selection), "--output-dir", str(root / "trained"),
                                                     "--epochs", "1", "--device", "cpu", "--batch-records", "1"])
                report = training.train(args)
                self.assertEqual(report["status"], "complete")
                models = training.load_models(args.output_dir, "cpu")
                self.assertEqual(set(models), {"affine", "residual_mlp", "transformer"})
                for row in report["models"].values():
                    self.assertEqual(row["optimizer_steps"], 1)
                    self.assertEqual(row["origin"], "fresh")
                write(selection, dict(schema="test10.pair_selection.v1", pairs=[dict(rows[0]),
                      dict(rows[1], video_id="train_video", path=rows[0]["path"])]))
                with self.assertRaises(ValueError): training.collection(selection)
                (args.output_dir / "residual_mlp.pt").write_bytes(b"corrupt")
                with self.assertRaisesRegex(ValueError, "checkpoint checksum"):
                    training.load_models(args.output_dir, "cpu")
        finally:
            torch.set_num_threads(old_threads)

    def test_train_validation_video_overlap_is_rejected_even_with_different_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("a", "b"): synthetic_pair(root / f"{name}.pt", "same_video")
            selection = root / "pairs.json"
            write(selection, dict(schema="test10.pair_selection.v1", pairs=[
                dict(dataset="synthetic", video_id="same_video", split=split, path=f"{name}.pt")
                for name, split in (("a", "train"), ("b", "validation"))]))
            with self.assertRaisesRegex(ValueError, "video-disjoint"):
                training.collection(selection)


class PerFrameScoring(unittest.TestCase):
    def test_missing_gt_never_calls_the_external_scorer(self):
        import runtime
        with tempfile.TemporaryDirectory() as tmp, patch("runtime.score") as scorer:
            case = dict(frame_stems=["00000", "00005"], annotation_dir=tmp)
            result = runtime.score_positions(case, "affine", {1: None})
            scorer.assert_not_called()
            self.assertEqual(result["post_switch"]["frames"], 0)
            self.assertIsNone(result["post_switch"]["J_and_F"])

    def test_all_ten_offsets_and_missing_annotation_are_recorded(self):
        case = dict(switch=10, end=20, frame_stems=[str(i * 5) for i in range(21)])
        def score(masks):
            count = sum(p % 2 == 0 for p in masks)
            return {"post_switch": dict(frames=count, J=1. if count else None,
                                        F=1. if count else None, J_and_F=1. if count else None)}
        result = suffix_scores(case, {i: None for i in range(11, 21)}, score)
        self.assertEqual(len(result["switch_frames"]), 10)
        self.assertEqual(result["switch_frames"]["+1"]["status"], "missing_annotation")
        self.assertIsNone(result["switch_frames"]["+1"]["J_and_F"])
        self.assertEqual(result["switch_frames"]["+10"]["frame_stem"], "100")
        self.assertEqual(result["switch_window"]["frames"], 5)
        short = dict(case, end=12)
        result = suffix_scores(short, {11: None, 12: None}, score)
        self.assertEqual(result["switch_frames"]["+10"]["status"], "outside_suffix")


if __name__ == "__main__":
    unittest.main()
