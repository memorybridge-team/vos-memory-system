from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from vos_memory_inspector.pair_bank import (
    PairBankWriter,
    SemanticRecordKey,
    load_bank_snapshot,
    load_pair_bank,
)
from vos_memory_inspector.precondition_experiment import (
    evaluate_translator_state,
    fit_transform_suite,
    sha256_file,
    train_two_stage,
)
from vos_memory_inspector.precondition_manifest import (
    _assign_groups,
    build_train_split_manifest,
    validate_train_split_manifest,
    write_train_split_manifest,
)
from vos_memory_inspector.preconditioning import (
    FixedCanonicalTransform,
    PreconditionedStateTranslator,
    anchor_transform,
    fit_preconditioned_steps,
    fit_standardization,
    fit_zca,
    identity_transform,
    output_projection_svd_transform,
)
from vos_memory_inspector.roundtrip import plan_cached_baseline
from vos_memory_inspector.consumer_diagnostics import compare_same_query_memory_attention
from vos_memory_inspector.state_schema import CanonicalState
from vos_memory_inspector.translators import ResidualMLPStateTranslator


def _state(spatial: torch.Tensor, pointer: torch.Tensor, *, dataset: str = "LVOS_V2") -> CanonicalState:
    records = spatial.shape[2]
    return CanonicalState(
        spatial_memory=spatial,
        object_pointer=pointer,
        presence_logits=torch.ones(1, 1, records, 1),
        frame_indices=torch.arange(records).view(1, 1, records),
        slot_order=torch.arange(records).view(1, 1, records),
        is_conditioning=torch.tensor([[[True] + [False] * (records - 1)]]),
        validity=torch.ones(1, 1, records, dtype=torch.bool),
        object_ids=(1,),
        switch_frame=records - 1,
        metadata={"dataset": dataset},
    ).validate()


def _pairs() -> list[tuple[CanonicalState, CanonicalState]]:
    generator = torch.Generator().manual_seed(5)
    result = []
    for dataset, shift in (("LVOS_V2", 0.5), ("VOST", -0.4)):
        spatial = torch.randn(1, 1, 2, 3, 3, 3, generator=generator)
        pointer = torch.randn(1, 1, 2, 4, generator=generator)
        source = _state(spatial, pointer, dataset=dataset)
        target = _state(spatial * 1.4 + shift, pointer * 0.8 - shift, dataset=dataset)
        result.append((source, target))
    return result


def _labelled(
    pairs: list[tuple[CanonicalState, CanonicalState]], *, split: str, pair_type: str
) -> list[tuple[CanonicalState, CanonicalState]]:
    """Attach the identity metadata that ``load_pair_bank`` adds to every record."""

    for source, target in pairs:
        for state in (source, target):
            state.metadata.update({"split": split, "pair_type": pair_type})
    return pairs


def _write_identity_suite(path: Path) -> Path:
    payload = {
        "schema_version": "cmmt.precondition_transform_suite.v1",
        "fit_manifest_sha256": "test-manifest",
        "spatial": {"M0_raw": identity_transform(3, 3).to_payload()},
        "pointer": {"P0_raw": identity_transform(4, 4).to_payload()},
    }
    torch.save(payload, path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{sha256_file(path)}  {path.name}\n", encoding="ascii"
    )
    return path


def test_two_stage_training_is_reproducible_from_seed_argument(tmp_path: Path) -> None:
    suite = _write_identity_suite(tmp_path / "suite.pt")
    aligned = _labelled(_pairs(), split="fit", pair_type="aligned")
    native = _labelled(_pairs(), split="fit", pair_type="native")

    def run(ambient_seed: int, output: Path) -> tuple[dict, dict[str, torch.Tensor]]:
        # A different global RNG state must not change the seeded result.
        torch.manual_seed(ambient_seed)
        report = train_two_stage(
            transform_suite=suite,
            aligned_pairs=aligned,
            native_pairs=native,
            spatial_method="M0_raw",
            pointer_method="P0_raw",
            architecture="mlp",
            output_dir=output,
            seed=7,
            device="cpu",
            aligned_steps=2,
            native_steps=1,
            batch_records=2,
            spatial_positions_per_record=4,
        )
        payload = torch.load(
            output / "translator_native_finetuned.pt", map_location="cpu", weights_only=True
        )
        return report, payload["state_dict"]

    first_report, first = run(1, tmp_path / "a")
    _second_report, second = run(2, tmp_path / "b")
    assert all(torch.equal(first[name], second[name]) for name in first)
    for stage in ("aligned", "native_finetune"):
        assert not Path(first_report[stage]["artifact"]).is_absolute()


def test_training_and_fitting_reject_wrong_split_or_pair_type(tmp_path: Path) -> None:
    for split, pair_type in (("dev", "aligned"), ("fit", "native")):
        with pytest.raises(ValueError, match="split='fit' pair_type='aligned'"):
            fit_transform_suite(
                _labelled(_pairs(), split=split, pair_type=pair_type),
                checkpoint_matrices=tmp_path / "unused.pt",
                output=tmp_path / "suite.pt",
                fit_manifest_sha256="x",
            )
    suite = _write_identity_suite(tmp_path / "identity.pt")
    with pytest.raises(ValueError, match="split='fit' pair_type='native'"):
        train_two_stage(
            transform_suite=suite,
            aligned_pairs=_labelled(_pairs(), split="fit", pair_type="aligned"),
            native_pairs=_labelled(_pairs(), split="test", pair_type="native"),
            spatial_method="M0_raw",
            pointer_method="P0_raw",
            architecture="mlp",
            output_dir=tmp_path / "out",
            device="cpu",
        )
    with pytest.raises(ValueError, match="missing"):
        fit_transform_suite(
            _pairs(),
            checkpoint_matrices=tmp_path / "unused.pt",
            output=tmp_path / "suite.pt",
            fit_manifest_sha256="x",
        )


def test_state_evaluation_requires_matching_labels(tmp_path: Path) -> None:
    pairs = _labelled(_pairs(), split="dev", pair_type="native")
    translator = PreconditionedStateTranslator(
        pairs[0][0].spec,
        pairs[0][1].spec,
        identity_transform(3, 3),
        identity_transform(4, 4),
        spatial_hidden_dim=4,
        pointer_hidden_dim=4,
    )
    artifact = tmp_path / "translator.pt"
    torch.save(translator.to_payload(), artifact)
    with pytest.raises(ValueError, match="split='test' pair_type='native'"):
        evaluate_translator_state(
            artifact, pairs, output=tmp_path / "bad.json", pair_type="native", split="test"
        )
    report = evaluate_translator_state(
        artifact, pairs, output=tmp_path / "eval.json", pair_type="native", split="dev"
    )
    assert report["record_count"] == len(pairs)
    assert report["artifact"] == "translator.pt"


def test_group_assignment_never_leaves_a_split_empty() -> None:
    for count in range(3, 40):
        assignment = _assign_groups([f"group-{index}" for index in range(count)], 7)
        sizes = {split: list(assignment.values()).count(split) for split in ("fit", "dev", "test")}
        assert min(sizes.values()) >= 1, (count, sizes)
    sizes_20 = list(_assign_groups([f"g{index}" for index in range(20)], 7).values())
    assert (sizes_20.count("fit"), sizes_20.count("dev"), sizes_20.count("test")) == (16, 2, 2)


def test_output_projection_svd_flags_column_space_truncation() -> None:
    generator = torch.Generator().manual_seed(3)
    wide = torch.randn(3, 5, generator=generator)
    tall = torch.randn(5, 3, generator=generator)
    exact = output_projection_svd_transform(wide, torch.zeros(3), wide, torch.zeros(3))
    lossy = output_projection_svd_transform(tall, torch.zeros(5), tall, torch.zeros(5))
    assert exact.settings["lossy_or_regularized"] is False
    assert exact.roundtrip_metrics(torch.randn(4, 3, generator=generator))["relative_l2"] < 1e-5
    assert lossy.settings["target"]["column_space_truncated"] is True
    assert lossy.settings["lossy_or_regularized"] is True
    assert lossy.roundtrip_metrics(torch.randn(4, 5, generator=generator))["relative_l2"] > 1e-2


def test_standardization_and_zca_roundtrip_raw_target() -> None:
    pairs = _pairs()
    for transform in (
        fit_standardization(pairs, "spatial_memory", epsilon=1e-4),
        fit_zca(pairs, "spatial_memory", shrinkage=1e-3, eigenvalue_floor=1e-4),
    ):
        value = pairs[0][1].spatial_memory.movedim(3, -1)
        reconstructed = transform.canonical_to_target(transform.target_to_canonical(value))
        assert torch.allclose(reconstructed, value.float(), atol=2e-5, rtol=2e-5)
        metrics = transform.roundtrip_metrics(value)
        assert metrics["finite"]


def test_anchor_preserving_pointer_maps_anchor_exactly() -> None:
    pairs = _pairs()
    source_anchor = torch.arange(4, dtype=torch.float32)
    target_anchor = torch.arange(4, dtype=torch.float32) + 10
    translator = PreconditionedStateTranslator(
        pairs[0][0].spec,
        pairs[0][1].spec,
        identity_transform(3, 3),
        anchor_transform(source_anchor, target_anchor),
        architecture="mlp",
        spatial_hidden_dim=5,
        pointer_hidden_dim=7,
        pointer_anchor_preserving=True,
    )
    predicted = translator.transform_pointer(source_anchor.view(1, -1))
    assert torch.equal(predicted, target_anchor.view(1, -1))


def test_final_linear_exact_anchor_bypasses_unstable_branch() -> None:
    pairs = _pairs()
    source_anchor = torch.arange(4, dtype=torch.float32)
    target_anchor = source_anchor + 20
    transform = FixedCanonicalTransform(
        name="sentinel_bypass_test",
        source_offset=torch.zeros(4),
        source_forward=torch.full((4, 4), float("nan")),
        target_offset=torch.zeros(4),
        target_forward=torch.eye(4),
        target_inverse=torch.full((4, 4), float("nan")),
        settings={
            "bypass_exact_anchor": True,
            "source_anchor": source_anchor,
            "target_anchor": target_anchor,
        },
    )
    translator = PreconditionedStateTranslator(
        pairs[0][0].spec,
        pairs[0][1].spec,
        identity_transform(3, 3),
        transform,
        pointer_hidden_dim=7,
    )
    assert torch.equal(
        translator.transform_pointer(source_anchor.view(1, -1)),
        target_anchor.view(1, -1),
    )


def test_component_hidden_dimensions_use_v3_artifact_and_roundtrip() -> None:
    pairs = _pairs()
    translator = PreconditionedStateTranslator(
        pairs[0][0].spec,
        pairs[0][1].spec,
        identity_transform(3, 3),
        identity_transform(4, 4),
        spatial_hidden_dim=8,
        pointer_hidden_dim=11,
    )
    payload = translator.to_payload()
    restored = PreconditionedStateTranslator.from_payload(payload)
    expected = translator.translate(pairs[0][0])
    actual = restored.translate(pairs[0][0])
    assert torch.allclose(expected.spatial_memory, actual.spatial_memory)
    assert torch.allclose(expected.object_pointer, actual.object_pointer)

    legacy_class = ResidualMLPStateTranslator(
        pairs[0][0].spec,
        pairs[0][1].spec,
        hidden_dim=None,
        feature_hidden_dim=8,
        pointer_hidden_dim=11,
    )
    assert legacy_class.to_payload()["schema_version"] == "cmmt.residual_mlp_translator.v3"


def test_step_training_is_deterministic_and_samples_each_record() -> None:
    pairs = _pairs()

    def train() -> dict[str, torch.Tensor]:
        torch.manual_seed(19)
        translator = PreconditionedStateTranslator(
            pairs[0][0].spec,
            pairs[0][1].spec,
            identity_transform(3, 3),
            identity_transform(4, 4),
            spatial_hidden_dim=6,
            pointer_hidden_dim=8,
        )
        history = fit_preconditioned_steps(
            translator,
            pairs,
            steps=3,
            batch_records=2,
            spatial_positions_per_record=4,
            learning_rate=1e-2,
            seed=7,
            device="cpu",
        )
        assert history[-1]["step"] == 3
        return {name: value.detach().clone() for name, value in translator.state_dict().items()}

    first = train()
    second = train()
    assert all(torch.equal(first[name], second[name]) for name in first)


def test_pair_bank_shares_records_across_snapshots(tmp_path: Path) -> None:
    source, target = _pairs()[0]
    source_one = _state(source.spatial_memory[:, :, :1], source.object_pointer[:, :, :1])
    target_one = _state(target.spatial_memory[:, :, :1], target.object_pointer[:, :, :1])
    provenance = {"manifest_sha256": "abc", "source_checkpoint_sha256": "s", "target_checkpoint_sha256": "t"}
    writer = PairBankWriter(
        tmp_path,
        dataset="LVOS_V2",
        release="v2",
        pair_type="aligned",
        split="fit",
        provenance=provenance,
    )
    key = SemanticRecordKey("LVOS_V2", "v2", "fit", "aligned", "video", "1", 0, True)
    record_id = writer.write_record(
        key, source_one, target_one, frame_id="00000001", predictor_frame_idx=0
    )
    first_snapshot = writer.write_snapshot(video_id="video", object_id="1", switch_frame=1, record_ids=[record_id], frame_id_map={"00000001": 0})
    writer.write_snapshot(video_id="video", object_id="1", switch_frame=2, record_ids=[record_id], frame_id_map={"00000001": 0})
    index = json.loads(writer.index_path.read_text(encoding="utf-8"))
    assert len(index["records"]) == 1
    assert len(index["snapshots"]) == 2
    loaded = load_pair_bank(writer.index_path)
    assert len(loaded) == 1
    source_snapshot, target_snapshot, metadata = load_bank_snapshot(
        writer.index_path, first_snapshot
    )
    assert source_snapshot.switch_frame == target_snapshot.switch_frame == 1
    assert source_snapshot.valid_record_count() == 1
    assert metadata["record_ids"] == [record_id]


def test_pair_bank_excludes_snapshot_only_records_from_training(tmp_path: Path) -> None:
    source, target = _pairs()[0]
    source_one = _state(source.spatial_memory[:, :, :1], source.object_pointer[:, :, :1])
    target_one = _state(target.spatial_memory[:, :, :1], target.object_pointer[:, :, :1])
    writer = PairBankWriter(
        tmp_path,
        dataset="LVOS_V2",
        release="v2",
        pair_type="native",
        split="fit",
        provenance={"manifest": "x"},
    )
    key = SemanticRecordKey("LVOS_V2", "v2", "fit", "native", "v", "1", 0, True)
    writer.write_record(
        key,
        source_one,
        target_one,
        frame_id="00000001",
        predictor_frame_idx=0,
        training_eligible=False,
    )
    assert load_pair_bank(writer.index_path) == []
    writer.write_record(
        key,
        source_one,
        target_one,
        frame_id="00000001",
        predictor_frame_idx=0,
        training_eligible=True,
    )
    assert len(load_pair_bank(writer.index_path)) == 1


def test_video_split_is_group_disjoint_and_hash_checked(tmp_path: Path) -> None:
    root = tmp_path / "LVOS_V2"
    for index in range(25):
        video = f"video-{index:02d}"
        image_dir = root / "train" / "JPEGImages" / video
        annotation_dir = root / "train" / "Annotations" / video
        image_dir.mkdir(parents=True)
        annotation_dir.mkdir(parents=True)
        (image_dir / "00000001.jpg").write_bytes(b"image")
        (annotation_dir / "00000001.png").write_bytes(b"mask")
    manifest = build_train_split_manifest(root, dataset="LVOS_V2", release="v2")
    validate_train_split_manifest(manifest)
    output = tmp_path / "manifest.json"
    write_train_split_manifest(output, manifest)
    assert output.is_file()
    manifest["videos"][0]["split"] = "test"
    try:
        validate_train_split_manifest(manifest)
    except ValueError as exc:
        assert "SHA-256" in str(exc)
    else:
        raise AssertionError("tampered split manifest was accepted")


def test_empty_memory_reset_is_distinct_from_legacy_empty_mask_proxy() -> None:
    empty = plan_cached_baseline("empty_memory_reset", switch_frame=10)
    legacy = plan_cached_baseline("target_reset", switch_frame=10)
    assert empty["prompt_frame"] == 11
    assert empty["history_frames_reprocessed"] == 0
    assert empty["prompt_source"] == "none"
    assert legacy["prompt_source"] == "blank_mask"


class _FakeCrossAttention(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.k_proj = nn.Linear(dimension, dimension, bias=False)
        self.v_proj = nn.Linear(dimension, dimension, bias=False)

    def forward(self, *, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return q + self.k_proj(k).mean(dim=-2, keepdim=True) + self.v_proj(v).mean(dim=-2, keepdim=True)


class _FakeLayer(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.cross_attn_image = _FakeCrossAttention(dimension)


class _FakeMemoryAttention(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer(dimension), _FakeLayer(dimension)])

    def forward(self, *, curr, memory, curr_pos, memory_pos, num_obj_ptr_tokens):
        del num_obj_ptr_tokens
        output = curr
        for layer in self.layers:
            output = layer.cross_attn_image(
                q=output + curr_pos, k=memory + memory_pos, v=memory
            )
        return output


def test_same_query_consumer_diagnostic_captures_each_layer() -> None:
    module = _FakeMemoryAttention(4)
    curr = torch.randn(1, 2, 4)
    memory = torch.randn(1, 3, 4)
    report = compare_same_query_memory_attention(
        module,
        curr=curr,
        curr_pos=torch.zeros_like(curr),
        native_memory=memory,
        native_memory_pos=torch.zeros_like(memory),
        translated_memory=memory.clone(),
        translated_memory_pos=torch.zeros_like(memory),
        num_obj_ptr_tokens=0,
    )
    assert len(report["layers"]) == 2
    assert report["full_memory_attention_output"]["mse"] == 0
    assert all(row["k_projection"]["mse"] == 0 for row in report["layers"])
