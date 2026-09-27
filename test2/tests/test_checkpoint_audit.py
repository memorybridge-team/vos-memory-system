from __future__ import annotations

from pathlib import Path

import torch

from vos_memory_inspector import checkpoint_audit
from vos_memory_inspector.checkpoint_audit import _l2_distance, code_audit


def test_l2_distance_skips_missing_or_mismatched_tensors() -> None:
    assert _l2_distance(torch.tensor([3.0, 0.0]), torch.tensor([0.0, 4.0])) == 5.0
    assert _l2_distance(torch.ones(1, 4), torch.zeros(4)) == 2.0
    assert _l2_distance(torch.empty(0), torch.ones(4)) is None
    assert _l2_distance(torch.ones(3), torch.ones(4)) is None


def _write(path: Path, *lines: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_code_audit_records_checkout_relative_paths(tmp_path: Path, monkeypatch) -> None:
    modeling = tmp_path / "sam2" / "modeling"
    _write(
        modeling / "sam2_base.py",
        "def _prepare_memory_conditioned_features(",
        "def _encode_new_memory(",
        "obj_ptr = self.obj_ptr_proj(sam_output_token)",
        "obj_ptrs = obj_ptrs.reshape(",
        "maskmem_features += (",
        'current_out["maskmem_features"] = maskmem_features',
        'current_out["maskmem_pos_enc"] = maskmem_pos_enc',
    )
    _write(
        modeling / "memory_encoder.py",
        "x = self.pix_feat_proj(pix_feat)",
        "x = x + masks",
        "x = self.fuser(x)",
        "x = self.out_proj(x)",
        "pos = self.position_encoding(x).to(x.dtype)",
    )
    _write(
        modeling / "memory_attention.py",
        'kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}',
    )
    _write(
        modeling / "sam" / "transformer.py",
        "num_k_rope = k.size(-2) - num_k_exclude_rope",
    )
    monkeypatch.setattr(checkpoint_audit, "verify_sam2_checkout", lambda _repo: "abc123")
    report = code_audit(tmp_path)
    paths = {row["path"] for row in report["verified_locations"].values()}
    assert paths == {
        "sam2/modeling/sam2_base.py",
        "sam2/modeling/memory_encoder.py",
        "sam2/modeling/memory_attention.py",
        "sam2/modeling/sam/transformer.py",
    }
    assert report["verified_locations"]["memory_producer"]["lines"] == [2]
    assert str(tmp_path) not in str(report)
