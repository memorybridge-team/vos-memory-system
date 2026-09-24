from __future__ import annotations

import csv
import json
from collections import OrderedDict

import torch

from vos_memory_inspector.attention_hook import MemoryAttentionProbe
from vos_memory_inspector.compatibility import compare_manifests
from vos_memory_inspector.manifest import DumpPolicy, ManifestWriter
from vos_memory_inspector.probe import ProbeConfig, StateProbe


def _frame_output(value: float, channels: int = 4) -> dict:
    return {
        "maskmem_features": torch.full((1, channels, 2, 2), value),
        "maskmem_pos_enc": [torch.zeros((1, channels, 2, 2))],
        "pred_masks": torch.full((1, 1, 4, 4), value),
        "obj_ptr": torch.full((1, 8), value),
        "object_score_logits": torch.tensor([[value]]),
    }


def _state() -> dict:
    output = {
        "cond_frame_outputs": {0: _frame_output(0.0)},
        "non_cond_frame_outputs": {1: _frame_output(1.0)},
    }
    return {
        "obj_id_to_idx": OrderedDict([(7, 0)]),
        "obj_idx_to_id": OrderedDict([(0, 7)]),
        "output_dict_per_obj": {0: output},
    }


def _config(model_id: str = "tiny") -> ProbeConfig:
    return ProbeConfig(
        model_id=model_id,
        upstream_commit="abc",
        video_id="synthetic",
        switch_frame=1,
    )


def _read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_state_probe_records_required_stats_and_frame_change(tmp_path) -> None:
    path = tmp_path / "probe.jsonl"
    csv_path = tmp_path / "probe.csv"
    with ManifestWriter(path, csv_path) as writer:
        probe = StateProbe(_config(), writer)
        assert probe.record_frame(_state(), 0) == 5
        assert probe.record_frame(_state(), 1) == 5
    rows = _read_jsonl(path)
    assert len(rows) == 10
    feature_rows = [row for row in rows if row["tensor_name"] == "maskmem_features"]
    assert feature_rows[0]["shape"] == [1, 4, 2, 2]
    assert feature_rows[0]["num_bytes"] == 64
    assert feature_rows[0]["changed_from_previous"] is None
    assert feature_rows[1]["changed_from_previous"] is True
    assert feature_rows[1]["max_abs_delta_from_previous"] == 1.0
    assert feature_rows[1]["cond_non_cond"] == "non_cond"
    position_rows = [row for row in rows if row["tensor_name"] == "maskmem_pos_enc[0]"]
    assert position_rows[1]["changed_from_previous"] is False
    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == len(rows)
    assert csv_rows[0]["model_id"] == "tiny"
    assert csv_rows[0]["shape"].startswith("[")


def test_dump_is_opt_in_and_moves_tensor_to_cpu(tmp_path) -> None:
    path = tmp_path / "probe.jsonl"
    dumps = tmp_path / "dumps"
    policy = DumpPolicy(dumps, frozenset({"obj_ptr"}))
    with ManifestWriter(path) as writer:
        StateProbe(_config(), writer, dump_policy=policy).record_frame(_state(), 0)
    rows = _read_jsonl(path)
    dumped = [row for row in rows if row["dump_path"]]
    assert [row["tensor_name"] for row in dumped] == ["obj_ptr"]
    tensor = torch.load(dumped[0]["dump_path"], weights_only=True)
    assert tensor.device.type == "cpu"


class _FakeMemoryAttention(torch.nn.Module):
    def forward(self, *, curr, curr_pos, memory, memory_pos, num_obj_ptr_tokens):
        del curr_pos, memory, memory_pos, num_obj_ptr_tokens
        return curr[-1]


class _FakePredictor:
    def __init__(self) -> None:
        self.memory_attention = _FakeMemoryAttention()

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,
    ):
        del frame_idx, is_init_cond_frame, feat_sizes, output_dict, num_frames, track_in_reverse
        memory = torch.ones((4, 1, 2))
        memory_pos = torch.zeros((4, 1, 2))
        return self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos,
            num_obj_ptr_tokens=0,
        )


def test_attention_hook_records_actual_forward_kwargs_and_restores_method(tmp_path) -> None:
    predictor = _FakePredictor()
    state = _state()
    original = predictor._prepare_memory_conditioned_features
    path = tmp_path / "attention.jsonl"
    with ManifestWriter(path) as writer:
        hook = MemoryAttentionProbe(predictor, state, _config(), writer)
        with hook:
            predictor._prepare_memory_conditioned_features(
                frame_idx=1,
                is_init_cond_frame=False,
                current_vision_feats=[torch.zeros((4, 1, 2))],
                current_vision_pos_embeds=[torch.zeros((4, 1, 2))],
                feat_sizes=[(2, 2)],
                output_dict=state["output_dict_per_obj"][0],
                num_frames=2,
            )
        assert predictor._prepare_memory_conditioned_features == original
    rows = _read_jsonl(path)
    assert [row["tensor_name"] for row in rows] == [
        "memory_attention.memory",
        "memory_attention.memory_pos",
    ]
    assert all(row["frame_idx"] == 1 for row in rows)
    assert all(row["object_id"] == 7 for row in rows)


def test_compatibility_report_distinguishes_copy_and_mapping_candidates(tmp_path) -> None:
    source = tmp_path / "source.jsonl"
    target = tmp_path / "target.jsonl"
    state_source = _state()
    state_target = _state()
    state_target["output_dict_per_obj"][0]["cond_frame_outputs"][0][
        "maskmem_features"
    ] = torch.zeros((1, 6, 2, 2))
    with ManifestWriter(source) as writer:
        StateProbe(_config("tiny"), writer).record_frame(state_source, 0)
    with ManifestWriter(target) as writer:
        StateProbe(_config("large"), writer).record_frame(state_target, 0)
    report = compare_manifests(source, target)
    dispositions = {
        row["tensor_name"]: row["disposition"] for row in report["comparisons"]
    }
    assert dispositions["maskmem_features"] == "linear_mapping_shape_candidate"
    assert dispositions["obj_ptr"] == "direct_copy_shape_candidate_semantics_unverified"
    assert (
        dispositions["pred_masks"]
        == "supporting_state_not_primary_translator_target"
    )
