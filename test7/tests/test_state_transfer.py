from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vos_memory_inspector.hf_cache import (
    flatten_kv_tokens,
    normalize_hf_cache,
    unflatten_kv_tokens,
)
from vos_memory_inspector.metrics import evaluate_state
from vos_memory_inspector.paired_experiment import run_paired_experiment
from vos_memory_inspector.sam2_state import (
    canonicalize_sam2_inference_state,
    inject_sam2_canonical_state,
    materialize_sam2_history,
)
from vos_memory_inspector.state_inspector import inspect_state, write_inspection_report
from vos_memory_inspector.state_schema import (
    CanonicalState,
    StateSpec,
    validate_paired_state_contract,
)
from vos_memory_inspector.translators import (
    DirectCopyTranslator,
    LearnedComponentPolicyTranslator,
    ResidualMLPStateTranslator,
    RidgeDirectPresenceTranslator,
    RidgeStateTranslator,
)


def _state(spatial: torch.Tensor, pointer: torch.Tensor, presence: torch.Tensor) -> CanonicalState:
    batch, objects, records = spatial.shape[:3]
    frame = torch.arange(records).view(1, 1, records).expand(batch, objects, records)
    return CanonicalState(
        spatial_memory=spatial,
        object_pointer=pointer,
        presence_logits=presence,
        frame_indices=frame.clone(),
        slot_order=frame.clone(),
        is_conditioning=torch.zeros(batch, objects, records, dtype=torch.bool),
        validity=torch.ones(batch, objects, records, dtype=torch.bool),
        object_ids=tuple(range(objects)),
        switch_frame=records - 1,
        metadata={"sentinel": "preserve"},
    ).validate()


def test_hf_cache_legacy_and_token_roundtrip() -> None:
    key = torch.arange(2 * 3 * 5 * 4).reshape(2, 3, 5, 4).float()
    value = key + 1
    cache = normalize_hf_cache(((key, value),))
    assert cache.layers[0].key.shape == (2, 3, 5, 4)
    flattened = flatten_kv_tokens(key)
    assert flattened.shape == (10, 12)
    restored = unflatten_kv_tokens(
        flattened, batch=2, heads=3, sequence=5, head_dim=4
    )
    assert torch.equal(restored, key)


def test_nested_inspector_writes_json_and_markdown(tmp_path: Path) -> None:
    class Layer:
        def __init__(self) -> None:
            self.keys = torch.zeros(1, 2, 3, 4)
            self.values = torch.ones(1, 2, 3, 4)

    class Cache:
        def __init__(self) -> None:
            self.layers = [Layer()]

    report = inspect_state({"cache": Cache(), "nested": [torch.zeros(2, 3)]})
    assert len(report.tensors) == 3
    assert report.total_bytes > 0
    write_inspection_report(report, tmp_path / "report.json", tmp_path / "report.md")
    assert "state.cache.layers[0].keys" in (tmp_path / "report.json").read_text()
    assert "Tensor state inspection" in (tmp_path / "report.md").read_text()


def test_sam2_multi_object_canonicalization_and_materialization() -> None:
    def output(value: float) -> dict[str, object]:
        return {
            "maskmem_features": torch.full((1, 3, 2, 2), value),
            "maskmem_pos_enc": [torch.full((1, 3, 2, 2), value + 0.1)],
            "pred_masks": torch.full((1, 1, 8, 8), value),
            "obj_ptr": torch.full((1, 4), value),
            "object_score_logits": torch.tensor([[value]]),
        }

    inference_state = {
        "obj_idx_to_id": {0: "a", 1: "b"},
        "obj_ids": ["a", "b"],
        "output_dict_per_obj": {
            0: {
                "cond_frame_outputs": {0: output(1)},
                "non_cond_frame_outputs": {2: output(2)},
            },
            1: {
                "cond_frame_outputs": {0: output(3)},
                "non_cond_frame_outputs": {2: output(4)},
            },
        },
        "point_inputs_per_obj": {},
        "mask_inputs_per_obj": {},
        "frames_tracked_per_obj": {0: {2: {"reverse": False}}},
        "device": "cpu",
        "storage_device": "cpu",
        "num_frames": 3,
    }
    state = canonicalize_sam2_inference_state(inference_state, switch_frame=2)
    assert state.spatial_memory.shape == (1, 2, 2, 3, 2, 2)
    assert state.object_ids == ("a", "b")
    assert state.valid_record_count() == 4
    history = materialize_sam2_history(
        state,
        positional_factory=lambda _obj, _frame, _cond, feature: [
            torch.zeros_like(feature)
        ],
    )
    assert history[0]["cond_frame_outputs"][0]["maskmem_pos_enc"][0].shape == (
        1,
        3,
        2,
        2,
    )
    assert history[1]["non_cond_frame_outputs"][2]["obj_ptr"].shape == (1, 4)
    for object_history in history.values():
        for records in object_history.values():
            for translated_record in records.values():
                assert set(translated_record) == {
                    "maskmem_features",
                    "maskmem_pos_enc",
                    "obj_ptr",
                }
    assert len(state.metadata["preserved_pred_masks"]) == 4
    assert not {
        "num_frames",
        "video_height",
        "video_width",
        "object_indices",
        "frames_tracked_per_obj",
        "preserved_inputs",
    } & state.metadata.keys()


def test_sam2_canonicalization_synchronizes_cuda_cpu_offload(monkeypatch) -> None:
    output = {
        "maskmem_features": torch.ones((1, 3, 2, 2)),
        "maskmem_pos_enc": [torch.zeros((1, 3, 2, 2))],
        "pred_masks": torch.ones((1, 1, 8, 8)),
        "obj_ptr": torch.ones((1, 4)),
        "object_score_logits": torch.ones((1, 1)),
    }
    inference_state = {
        "obj_idx_to_id": {0: 1},
        "obj_ids": [1],
        "output_dict_per_obj": {
            0: {
                "cond_frame_outputs": {0: output},
                "non_cond_frame_outputs": {},
            }
        },
        "point_inputs_per_obj": {0: {}},
        "mask_inputs_per_obj": {0: {}},
        "frames_tracked_per_obj": {0: {0: {"reverse": False}}},
        "device": "cuda:0",
        "storage_device": "cpu",
        "num_frames": 1,
        "video_height": 8,
        "video_width": 8,
    }
    synchronized: list[torch.device] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronized.append)

    canonicalize_sam2_inference_state(inference_state, switch_frame=0)

    assert synchronized == [torch.device("cuda:0")]


def test_sam2_reexport_allows_missing_diagnostic_presence_logits() -> None:
    injected_record = {
        "maskmem_features": torch.ones((1, 3, 2, 2)),
        "maskmem_pos_enc": [torch.zeros((1, 3, 2, 2))],
        "obj_ptr": torch.ones((1, 4)),
    }
    native_record = {
        **injected_record,
        "object_score_logits": torch.tensor([[2.0]]),
    }
    inference_state = {
        "obj_idx_to_id": {0: 1},
        "obj_ids": [1],
        "output_dict_per_obj": {
            0: {
                "cond_frame_outputs": {0: injected_record},
                "non_cond_frame_outputs": {1: native_record},
            }
        },
        "device": "cpu",
        "storage_device": "cpu",
    }

    state = canonicalize_sam2_inference_state(inference_state, switch_frame=1)

    assert state.valid_record_count() == 2
    assert state.presence_logits[0, 0, 0, 0].item() == 0.0
    assert state.presence_logits[0, 0, 1, 0].item() == 2.0
    assert state.metadata["missing_presence_records"] == ["object=0/record=0"]


def test_sam2_injection_restores_registry_history_and_target_position() -> None:
    class Position(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.full_like(x, 9.0)

    class Predictor:
        def __init__(self) -> None:
            self.memory_encoder = type("MemoryEncoder", (), {})()
            self.memory_encoder.position_encoding = Position()

    def output(value: float) -> dict[str, object]:
        return {
            "maskmem_features": torch.full((1, 3, 2, 2), value),
            "maskmem_pos_enc": [torch.full((1, 3, 2, 2), -1.0)],
            "pred_masks": torch.full((1, 1, 8, 8), value),
            "obj_ptr": torch.full((1, 4), value),
            "object_score_logits": torch.tensor([[value]]),
        }

    source = {
        "obj_idx_to_id": {0: 7},
        "obj_ids": [7],
        "output_dict_per_obj": {
            0: {
                "cond_frame_outputs": {0: output(1)},
                "non_cond_frame_outputs": {1: output(2)},
            }
        },
        "point_inputs_per_obj": {0: {}},
        "mask_inputs_per_obj": {0: {0: torch.ones(1, 1, 8, 8)}},
        "frames_tracked_per_obj": {0: {0: {"reverse": False}}},
        "device": "cpu",
        "storage_device": "cpu",
        "num_frames": 3,
        "video_height": 8,
        "video_width": 8,
    }
    canonical = canonicalize_sam2_inference_state(source, switch_frame=1)
    target = {
        "device": torch.device("cpu"),
        "storage_device": torch.device("cpu"),
        "num_frames": 3,
        "video_height": 8,
        "video_width": 8,
        "constants": {},
        "obj_id_to_idx": {},
        "obj_idx_to_id": {},
        "obj_ids": [],
        "point_inputs_per_obj": {},
        "mask_inputs_per_obj": {},
        "output_dict_per_obj": {},
        "temp_output_dict_per_obj": {},
        "frames_tracked_per_obj": {},
    }
    summary = inject_sam2_canonical_state(
        canonical,
        predictor=Predictor(),
        inference_state=target,
    )
    assert summary == {"objects": 1, "records": 2, "switch_frame": 1}
    assert target["obj_id_to_idx"] == {7: 0}
    assert target["point_inputs_per_obj"][0] == {}
    assert target["mask_inputs_per_obj"][0] == {}
    assert target["frames_tracked_per_obj"][0] == {}
    restored = target["output_dict_per_obj"][0]
    assert torch.all(restored["cond_frame_outputs"][0]["maskmem_pos_enc"][0] == 9)
    assert not torch.any(restored["cond_frame_outputs"][0]["maskmem_pos_enc"][0] == -1)
    for records in restored.values():
        for translated_record in records.values():
            assert "pred_masks" not in translated_record
            assert "object_score_logits" not in translated_record
    assert len(canonical.metadata["preserved_pred_masks"]) == 2


def test_handoff_bytes_counts_only_injected_payload() -> None:
    state = _state(
        torch.zeros(1, 1, 2, 3, 2, 2),
        torch.zeros(1, 1, 2, 4),
        torch.zeros(1, 1, 2, 1),
    )
    state.metadata.update(
        {
            "num_frames": 999,
            "video_height": 4096,
            "video_width": 4096,
            "video_fingerprint": "not-transferred",
        }
    )
    expected_tensor_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (
            state.spatial_memory,
            state.object_pointer,
            state.frame_indices,
            state.slot_order,
            state.is_conditioning,
            state.validity,
        )
    )
    # object_ids=(0,) serializes as "[0]" and switch_frame is int64.
    assert state.handoff_bytes() == expected_tensor_bytes + 3 + 8
    assert state.continuous_bytes() > (
        state.spatial_memory.numel() * state.spatial_memory.element_size()
        + state.object_pointer.numel() * state.object_pointer.element_size()
    )
    assert evaluate_state(state, state)["translated_bytes"] == state.handoff_bytes()


def test_direct_adapts_shape_and_preserves_discrete_state() -> None:
    source = _state(
        torch.randn(1, 1, 2, 3, 2, 4),
        torch.randn(1, 1, 2, 5),
        torch.randn(1, 1, 2, 1),
    )
    translated = DirectCopyTranslator(StateSpec(4, 3, 2, 3)).translate(source)
    assert translated.spatial_memory.shape == (1, 1, 2, 4, 3, 2)
    assert translated.object_pointer.shape == (1, 1, 2, 3)
    assert torch.equal(translated.frame_indices, source.frame_indices)
    assert translated.metadata["sentinel"] == "preserve"
    assert translated.positional_information == {"policy": "regenerate_at_target"}


def test_paired_contract_requires_one_discrete_timeline() -> None:
    source = _state(
        torch.randn(1, 1, 2, 3, 2, 2),
        torch.randn(1, 1, 2, 4),
        torch.randn(1, 1, 2, 1),
    )
    target = source.with_continuous(
        spatial_memory=torch.randn(1, 1, 2, 5, 3, 3),
        object_pointer=torch.randn(1, 1, 2, 6),
        presence_logits=torch.randn(1, 1, 2, 1),
    )
    assert torch.equal(validate_paired_state_contract(source, target), source.validity)

    target.slot_order = target.slot_order.clone()
    target.slot_order[0, 0, 1] = 99
    with pytest.raises(ValueError, match="slot_order"):
        validate_paired_state_contract(source, target)


def test_paired_contract_rejects_padding_or_switch_mismatch() -> None:
    source = _state(
        torch.randn(1, 1, 2, 3, 2, 2),
        torch.randn(1, 1, 2, 4),
        torch.randn(1, 1, 2, 1),
    )
    target = source.with_continuous(
        spatial_memory=source.spatial_memory.clone(),
        object_pointer=source.object_pointer.clone(),
        presence_logits=source.presence_logits.clone(),
    )
    target.validity = target.validity.clone()
    target.validity[0, 0, 1] = False
    with pytest.raises(ValueError, match="validity"):
        validate_paired_state_contract(source, target)

    target.validity = source.validity.clone()
    target.switch_frame += 1
    with pytest.raises(ValueError, match="switch_frame"):
        validate_paired_state_contract(source, target)


def test_ridge_recovers_affine_components() -> None:
    generator = torch.Generator().manual_seed(11)
    sources = []
    targets = []
    feature_weight = torch.randn(3, 4, generator=generator)
    pointer_weight = torch.randn(5, 2, generator=generator)
    for _ in range(3):
        source = _state(
            torch.randn(1, 2, 3, 3, 2, 2, generator=generator),
            torch.randn(1, 2, 3, 5, generator=generator),
            torch.randn(1, 2, 3, 1, generator=generator),
        )
        target = source.with_continuous(
            spatial_memory=(source.spatial_memory.movedim(3, -1) @ feature_weight + 0.3).movedim(-1, 3),
            object_pointer=source.object_pointer @ pointer_weight - 0.2,
            presence_logits=source.presence_logits * 1.7 + 0.4,
        )
        sources.append(source)
        targets.append(target)
    ridge = RidgeStateTranslator.fit(list(zip(sources, targets)), ridge_lambda=0.0)
    result = evaluate_state(ridge.translate(sources[-1]), targets[-1])
    assert result["aggregate_mse"] < 1e-9

    restored = RidgeStateTranslator.from_payload(ridge.to_payload())
    restored_result = evaluate_state(restored.translate(sources[-1]), targets[-1])
    assert restored_result["aggregate_mse"] < 1e-9


def test_ridge_direct_presence_uses_ridge_for_memory_only() -> None:
    source = _state(
        torch.randn(1, 1, 2, 3, 2, 2),
        torch.randn(1, 1, 2, 4),
        torch.randn(1, 1, 2, 1),
    )
    target = source.with_continuous(
        spatial_memory=source.spatial_memory * 2,
        object_pointer=source.object_pointer * 3,
        presence_logits=source.presence_logits + 100,
    )
    ridge = RidgeStateTranslator.fit([(source, target)], ridge_lambda=0.01)

    translated = RidgeDirectPresenceTranslator(ridge).translate(source)

    assert torch.allclose(translated.presence_logits, source.presence_logits)
    assert not torch.allclose(translated.spatial_memory, source.spatial_memory)
    assert translated.metadata["translation"]["presence_logits"] == "direct"


def test_mlp_residuals_are_only_enabled_for_equal_feature_dimensions() -> None:
    source_spec = StateSpec(3, 2, 2, 5)
    mismatched = ResidualMLPStateTranslator(source_spec, StateSpec(4, 2, 2, 2))
    assert not mismatched.feature_residual
    assert not mismatched.pointer_residual
    matched = ResidualMLPStateTranslator(source_spec, source_spec)
    assert matched.feature_residual
    assert matched.pointer_residual
    grid_mismatch = ResidualMLPStateTranslator(source_spec, StateSpec(3, 3, 2, 5))
    assert not grid_mismatch.feature_residual


def test_residual_mlp_payload_roundtrip_preserves_predictions() -> None:
    spec = StateSpec(3, 2, 2, 5)
    translator = ResidualMLPStateTranslator(spec, spec, hidden_dim=7)
    source = _state(
        torch.randn(1, 1, 2, 3, 2, 2),
        torch.randn(1, 1, 2, 5),
        torch.randn(1, 1, 2, 1),
    )

    restored = ResidualMLPStateTranslator.from_payload(translator.to_payload())

    assert restored.hidden_dim == 7
    assert restored.source_spec == spec
    assert restored.target_spec == spec
    expected = translator.translate(source)
    actual = restored.translate(source)
    assert torch.allclose(actual.spatial_memory, expected.spatial_memory)
    assert torch.allclose(actual.object_pointer, expected.object_pointer)
    assert torch.allclose(actual.presence_logits, expected.presence_logits)


def test_learned_component_policy_uses_direct_for_unselected_components() -> None:
    spec = StateSpec(3, 2, 2, 5)
    learned = ResidualMLPStateTranslator(spec, spec, hidden_dim=7)
    source = _state(
        torch.randn(1, 1, 2, 3, 2, 2),
        torch.randn(1, 1, 2, 5),
        torch.randn(1, 1, 2, 1),
    )
    policy = LearnedComponentPolicyTranslator(
        learned,
        learned_components=("spatial_memory", "object_pointer"),
    )

    translated = policy.translate(source)
    learned_state = learned.translate(source)

    assert torch.allclose(translated.spatial_memory, learned_state.spatial_memory)
    assert torch.allclose(translated.object_pointer, learned_state.object_pointer)
    assert torch.allclose(translated.presence_logits, source.presence_logits)
    assert translated.metadata["translation"]["component_policy"] == {
        "object_pointer": "learned",
        "spatial_memory": "learned",
    }
    assert translated.metadata["translation"]["presence_logits"] == "diagnostic_only"


def test_paired_experiment_serializes_residual_mlp_contract(tmp_path: Path) -> None:
    source = _state(
        torch.randn(1, 1, 2, 3, 2, 2),
        torch.randn(1, 1, 2, 4),
        torch.randn(1, 1, 2, 1),
    )
    target = source.with_continuous(
        spatial_memory=source.spatial_memory * 1.1,
        object_pointer=source.object_pointer * 0.9,
        presence_logits=source.presence_logits + 0.2,
    )

    report = run_paired_experiment(
        [(source, target)],
        [(source, target)],
        tmp_path,
        epochs=1,
        hidden_dim=6,
        translator_names=("residual_mlp",),
        device="cpu",
        spatial_samples_per_pair=3,
    )

    assert report["training"]["device"] == "cpu"
    assert report["training"]["spatial_samples_per_pair"] == 3
    saved = torch.load(
        tmp_path / "paired_translators.pt", map_location="cpu", weights_only=True
    )
    payload = saved["residual_mlp"]
    assert payload["schema_version"] == "cmmt.residual_mlp_translator.v2"
    assert payload["hidden_dim"] == 6
    assert payload["source_spec"] == source.spec.to_dict()
    assert payload["target_spec"] == target.spec.to_dict()


def test_paired_experiment_can_run_direct_and_ridge_only(tmp_path: Path) -> None:
    source = _state(
        torch.randn(1, 1, 2, 3, 2, 2),
        torch.randn(1, 1, 2, 4),
        torch.randn(1, 1, 2, 1),
    )
    target = source.with_continuous(
        spatial_memory=source.spatial_memory * 1.2 + 0.1,
        object_pointer=source.object_pointer * 0.8 - 0.2,
        presence_logits=source.presence_logits + 0.3,
    )
    report = run_paired_experiment(
        [(source, target)],
        [(source, target)],
        tmp_path,
        translator_names=("direct", "ridge"),
    )
    assert report["translator_names"] == ["direct", "ridge"]
    assert set(report["results"]) == {"direct", "ridge"}
    assert "linear_initial_loss" not in report["training"]
    saved = torch.load(
        tmp_path / "paired_translators.pt", map_location="cpu", weights_only=True
    )
    assert set(saved) == {"ridge"}
    assert set(saved["ridge"]) >= {"feature_weight", "pointer_weight"}
