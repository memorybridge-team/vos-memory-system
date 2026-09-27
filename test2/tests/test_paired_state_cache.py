from pathlib import Path

import pytest
import torch

from vos_memory_inspector.paired_state_cache import (
    load_paired_state_cache,
    write_paired_state_cache,
)
from vos_memory_inspector.state_schema import CanonicalState


def _state(scale: float) -> CanonicalState:
    return CanonicalState(
        spatial_memory=torch.ones(1, 1, 2, 3, 2, 2) * scale,
        object_pointer=torch.ones(1, 1, 2, 4) * scale,
        presence_logits=torch.ones(1, 1, 2, 1) * scale,
        frame_indices=torch.tensor([[[0, 2]]]),
        slot_order=torch.tensor([[[0, 1]]]),
        is_conditioning=torch.tensor([[[True, False]]]),
        validity=torch.tensor([[[True, True]]]),
        object_ids=(1,),
        switch_frame=2,
    ).validate()


def test_compact_paired_state_cache_roundtrip(tmp_path: Path) -> None:
    output = tmp_path / "pair.pt"
    report = write_paired_state_cache(
        output,
        source_canonical=_state(1.0),
        target_canonical=_state(2.0),
        metadata={"sequence": "demo"},
    )

    source, target, metadata = load_paired_state_cache(output)

    assert report["bytes"] == output.stat().st_size
    assert torch.equal(source.spatial_memory, torch.ones_like(source.spatial_memory))
    assert torch.equal(target.spatial_memory, torch.ones_like(target.spatial_memory) * 2)
    assert metadata["sequence"] == "demo"


def test_compact_paired_state_cache_rejects_bad_checksum(tmp_path: Path) -> None:
    output = tmp_path / "pair.pt"
    write_paired_state_cache(
        output,
        source_canonical=_state(1.0),
        target_canonical=_state(2.0),
        metadata={"sequence": "demo"},
    )
    output.with_suffix(".pt.sha256").write_text("0" * 64, encoding="ascii")

    with pytest.raises(ValueError, match="SHA-256"):
        load_paired_state_cache(output)
