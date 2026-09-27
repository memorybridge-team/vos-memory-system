from __future__ import annotations

import subprocess
from pathlib import Path


SUPPORTED_SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"


def verify_sam2_checkout(
    repository: str | Path, *, allow_mismatch: bool = False
) -> str:
    repository = Path(repository).resolve()
    if not (repository / "sam2" / "sam2_video_predictor.py").is_file():
        raise FileNotFoundError(
            f"Not a facebookresearch/sam2 checkout: {repository}"
        )
    result = subprocess.run(
        ["git", "-c", f"safe.directory={repository}", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if commit != SUPPORTED_SAM2_COMMIT and not allow_mismatch:
        raise RuntimeError(
            "SAM 2 checkout mismatch: "
            f"expected {SUPPORTED_SAM2_COMMIT}, found {commit}. "
            "Checkout the pinned commit or pass --allow-upstream-mismatch and record the risk."
        )
    verify_pinned_source_contract(repository)
    return commit


def verify_pinned_source_contract(repository: str | Path) -> None:
    """Fail before inference if the private state/hook contract is absent."""

    repository = Path(repository).resolve()
    contracts = {
        repository / "sam2" / "modeling" / "sam2_base.py": (
            "def _prepare_memory_conditioned_features(",
            'prev["maskmem_features"]',
            'prev["maskmem_pos_enc"][-1]',
            'out["obj_ptr"]',
            "self.memory_attention(",
            "def _encode_new_memory(",
        ),
        repository / "sam2" / "sam2_video_predictor.py": (
            'inference_state["output_dict_per_obj"]',
            '"cond_frame_outputs"',
            '"non_cond_frame_outputs"',
            '"maskmem_features": maskmem_features',
            '"maskmem_pos_enc": maskmem_pos_enc',
            '"pred_masks": pred_masks',
            '"obj_ptr": obj_ptr',
            '"object_score_logits": object_score_logits',
        ),
    }
    missing: list[str] = []
    for path, snippets in contracts.items():
        if not path.is_file():
            missing.append(str(path))
            continue
        source = path.read_text(encoding="utf-8")
        missing.extend(f"{path}:{snippet}" for snippet in snippets if snippet not in source)
    if missing:
        raise RuntimeError(
            "Pinned SAM 2 private source contract is not satisfied; missing: "
            + "; ".join(missing)
        )
