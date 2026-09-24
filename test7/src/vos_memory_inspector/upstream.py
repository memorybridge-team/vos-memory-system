from __future__ import annotations

import os
import subprocess
from pathlib import Path


SAM2_UPSTREAM_COMMIT_ENV = "SAM2_UPSTREAM_COMMIT"
DEFAULT_SAM2_UPSTREAM_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"


def _dotenv_candidates() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [Path.cwd() / ".env", repo_root / ".env"]
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _read_dotenv_value(path: Path, name: str) -> str:
    if not path.is_file():
        return ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != name:
            continue
        return value.strip().strip('"').strip("'")
    return ""


def sam2_upstream_commit() -> str:
    """Return the pinned facebookresearch/sam2 commit.

    An exported ``SAM2_UPSTREAM_COMMIT`` wins. Otherwise the first ``.env``
    in the working directory or repository root is used. The baked default is
    the SAM 2.1 commit previously stored in ``SAM2_UPSTREAM_COMMIT``.
    """

    from_env = os.environ.get(SAM2_UPSTREAM_COMMIT_ENV, "").strip()
    if from_env:
        return from_env
    for path in _dotenv_candidates():
        from_file = _read_dotenv_value(path, SAM2_UPSTREAM_COMMIT_ENV)
        if from_file:
            return from_file
    return DEFAULT_SAM2_UPSTREAM_COMMIT


SUPPORTED_SAM2_COMMIT = sam2_upstream_commit()


def verify_sam2_checkout(
    repository: str | Path, *, allow_mismatch: bool = False
) -> str:
    repository = Path(repository).resolve()
    if not (repository / "sam2" / "sam2_video_predictor.py").is_file():
        raise FileNotFoundError(
            f"Not a facebookresearch/sam2 checkout: {repository}"
        )
    expected = sam2_upstream_commit()
    result = subprocess.run(
        ["git", "-c", f"safe.directory={repository}", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if commit != expected and not allow_mismatch:
        raise RuntimeError(
            "SAM 2 checkout mismatch: "
            f"expected {expected}, found {commit}. "
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
