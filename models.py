"""Pinned SAM2 source/checkpoint validation and lazy model loading."""
import hashlib
import subprocess
import sys
from pathlib import Path

from datasets import download, sha256
from protocol import MODEL_BASE_URL, MODELS, SAM2_COMMIT


def source_hash(directory):
    digest = hashlib.sha256()
    paths = list(Path(directory).glob("*.py"))
    paths.append(Path(directory) / "lvos_v1_unseen_videos.txt")
    for path in sorted(path for path in paths if path.is_file()):
        if path.name.startswith("test_"):
            continue
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def verify_sam2_source(repo):
    repo = Path(repo).resolve()
    if not (repo / "sam2" / "build_sam.py").is_file():
        raise FileNotFoundError(
            f"SAM2 source not found at {repo}. Run ./setup.sh first or set sam2_repo."
        )
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    )
    commit = result.stdout.strip()
    if commit != SAM2_COMMIT:
        raise RuntimeError(
            f"SAM2 commit mismatch: expected {SAM2_COMMIT}, found {commit}. "
            "This benchmark intentionally pins original SAM2."
        )
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                           check=True, capture_output=True, text=True).stdout
    if dirty.strip():
        raise RuntimeError("SAM2 tracked source/config files have local changes")
    return repo, commit


def checkpoint_plan(model_name, cache, explicit=None):
    if explicit:
        return str(Path(explicit).expanduser())
    return str(Path(cache) / "checkpoints" / MODELS[model_name][1])


def acquire_checkpoint(model_name, cache, explicit=None):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
    else:
        filename = MODELS[model_name][1]
        path = download(f"{MODEL_BASE_URL}/{filename}",
                        Path(cache) / "checkpoints" / filename)
    return path, sha256(path)


def build_predictor(model_name, checkpoint, sam2_repo, device):
    repo, commit = verify_sam2_source(sam2_repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from sam2.build_sam import build_sam2_video_predictor
    config, _ = MODELS[model_name]
    predictor = build_sam2_video_predictor(
        config_file=config,
        ckpt_path=str(checkpoint),
        device=device,
        hydra_overrides_extra=["++model.non_overlap_masks=false"],
        # Meta's VOS evaluation tool leaves hole-filling/dynamic postprocessing
        # disabled. Keep the same setting for comparable benchmark masks.
        apply_postprocessing=False,
        vos_optimized=False,
    )
    return predictor, {"sam2_commit": commit, "config": config}
