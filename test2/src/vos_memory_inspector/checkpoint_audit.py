"""Source and checkpoint audit for the pinned SAM 2.1 memory boundary."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .io_utils import atomic_torch_save, atomic_write_text
from .upstream import verify_sam2_checkout


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _matrix_diagnostics(value: torch.Tensor) -> dict[str, Any]:
    matrix = value.detach().float().reshape(value.shape[0], -1).cpu()
    singular = torch.linalg.svdvals(matrix.double())
    maximum = float(singular.max()) if singular.numel() else 0.0
    minimum = float(singular.min()) if singular.numel() else 0.0
    ranks = {
        str(tolerance): int((singular > maximum * tolerance).sum())
        for tolerance in (1e-3, 1e-4, 1e-5, 1e-6)
    }
    return {
        "shape": list(value.shape),
        "singular_values": [float(item) for item in singular],
        "max_singular_value": maximum,
        "min_singular_value": minimum,
        "condition_number": None if minimum == 0 else maximum / minimum,
        "numerical_rank_relative": ranks,
        "frobenius_norm": float(matrix.norm()),
        "operator_norm": maximum,
    }


def _parameter_summary(value: torch.Tensor) -> dict[str, Any]:
    flat = value.detach().float().reshape(-1).cpu()
    return {
        "shape": list(value.shape),
        "norm": float(flat.norm()),
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)),
        "min": float(flat.min()),
        "max": float(flat.max()),
    }


def _named_linear_structure(module: torch.nn.Module) -> list[dict[str, Any]]:
    result = []
    for name, child in module.named_modules():
        if isinstance(child, torch.nn.Linear):
            result.append(
                {
                    "name": name or "<root>",
                    "type": type(child).__name__,
                    "weight": _matrix_diagnostics(child.weight),
                    "bias_shape": None if child.bias is None else list(child.bias.shape),
                }
            )
        elif name and isinstance(child, (torch.nn.GELU, torch.nn.ReLU)):
            result.append({"name": name, "type": type(child).__name__})
    return result


def _cross_attention_projections(model: torch.nn.Module) -> tuple[list[dict[str, Any]], torch.Tensor]:
    rows: list[dict[str, Any]] = []
    metric: torch.Tensor | None = None
    for name, module in model.named_modules():
        lowered = name.lower()
        if "cross" not in lowered or "att" not in lowered:
            continue
        k_proj = getattr(module, "k_proj", None)
        v_proj = getattr(module, "v_proj", None)
        if not isinstance(k_proj, torch.nn.Linear) or not isinstance(v_proj, torch.nn.Linear):
            continue
        k_weight = k_proj.weight.detach().float().cpu()
        v_weight = v_proj.weight.detach().float().cpu()
        if k_weight.shape[1] != v_weight.shape[1]:
            raise ValueError(f"K/V input dimension mismatch in {name}")
        local = k_weight.T @ k_weight + v_weight.T @ v_weight
        metric = local if metric is None else metric + local
        rows.append(
            {
                "module": name,
                "k_projection": _matrix_diagnostics(k_weight),
                "v_projection": _matrix_diagnostics(v_weight),
            }
        )
    if metric is None:
        raise RuntimeError("no memory cross-attention K/V projections were found")
    return rows, metric


def _load_predictor(
    sam2_repo: Path,
    config: str,
    checkpoint: Path,
    device: str,
) -> torch.nn.Module:
    repo_text = str(sam2_repo)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    from sam2.build_sam import build_sam2_video_predictor

    predictor = build_sam2_video_predictor(
        config_file=config,
        ckpt_path=str(checkpoint),
        device=device,
    )
    predictor.eval()
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)
    return predictor


def _runtime_memory_probe(
    model: torch.nn.Module, *, device: str, inference_dtype: str
) -> dict[str, Any]:
    from .sam2_state import init_sam2_inference_state_without_warmup

    dtype = getattr(torch, inference_dtype)
    autocast = (
        torch.autocast(device_type="cuda", dtype=dtype)
        if str(device).startswith("cuda")
        else nullcontext()
    )
    with tempfile.TemporaryDirectory(prefix="cmmt-sam21-audit-") as directory:
        frame = np.zeros((1024, 1024, 3), dtype=np.uint8)
        frame[256:768, 256:768] = 127
        Image.fromarray(frame).save(Path(directory) / "00000000.jpg")
        mask = np.zeros((1024, 1024), dtype=bool)
        mask[320:704, 320:704] = True
        state = init_sam2_inference_state_without_warmup(
            model,
            video_path=directory,
            offload_video_to_cpu=True,
            offload_state_to_cpu=False,
        )
        with torch.inference_mode(), autocast:
            model.add_new_mask(state, frame_idx=0, obj_id=1, mask=mask)
            model.propagate_in_video_preflight(state)
        output = state["output_dict_per_obj"][0]["cond_frame_outputs"][0]
        spatial = output["maskmem_features"]
        pointer = output["obj_ptr"]
        positional = output["maskmem_pos_enc"]
        return {
            "input_resolution": [1024, 1024],
            "maskmem_features": {
                "shape": list(spatial.shape),
                "dtype": str(spatial.dtype),
            },
            "maskmem_pos_enc": [
                {"shape": list(value.shape), "dtype": str(value.dtype)}
                for value in positional
            ],
            "obj_ptr": {"shape": list(pointer.shape), "dtype": str(pointer.dtype)},
            "stored_separately": spatial.data_ptr() != positional[-1].data_ptr(),
        }


def _model_audit(
    model: torch.nn.Module, *, device: str, inference_dtype: str
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    out_proj = model.memory_encoder.out_proj
    if not hasattr(out_proj, "weight"):
        raise RuntimeError("memory_encoder.out_proj has no weight")
    out_weight = out_proj.weight.detach().float().cpu()
    out_bias = (
        torch.zeros(out_weight.shape[0])
        if out_proj.bias is None
        else out_proj.bias.detach().float().cpu()
    )
    pointer = model.obj_ptr_proj
    pointer_linears = [child for child in pointer.modules() if isinstance(child, torch.nn.Linear)]
    pointer_last = pointer_linears[-1] if pointer_linears else None
    attention_rows, consumer_metric = _cross_attention_projections(model.memory_attention)
    report: dict[str, Any] = {
        "hidden_dim": int(model.hidden_dim),
        "memory_dim": int(model.mem_dim),
        "num_maskmem": int(model.num_maskmem),
        "memory_encoder_out_proj": {
            "weight": _matrix_diagnostics(out_weight),
            "bias": _parameter_summary(out_bias),
        },
        "obj_ptr_proj": {
            "type": type(pointer).__name__,
            "layers": _named_linear_structure(pointer),
        },
        "no_obj_ptr": (
            None
            if getattr(model, "no_obj_ptr", None) is None
            else _parameter_summary(model.no_obj_ptr)
        ),
        "no_obj_embed_spatial": (
            None
            if getattr(model, "no_obj_embed_spatial", None) is None
            else _parameter_summary(model.no_obj_embed_spatial)
        ),
        "memory_attention_cross_attention": attention_rows,
        "consumer_metric": _matrix_diagnostics(consumer_metric),
        "pointer_consumption": {
            "stored_dimension": int(model.hidden_dim),
            "token_dimension": int(model.mem_dim),
            "tokens_per_pointer": int(model.hidden_dim // model.mem_dim),
            "temporal_pe_added_at_consumption": bool(model.add_tpos_enc_to_obj_ptrs),
            "temporal_pe_projected": bool(model.proj_tpos_enc_in_obj_ptrs),
        },
        "object_presence_handling": {
            "pred_obj_scores": bool(model.pred_obj_scores),
            "fixed_no_obj_ptr": bool(model.fixed_no_obj_ptr),
            "soft_no_obj_ptr": bool(model.soft_no_obj_ptr),
            "no_obj_embed_spatial_enabled": model.no_obj_embed_spatial is not None,
        },
        "runtime_state_probe": _runtime_memory_probe(
            model, device=device, inference_dtype=inference_dtype
        ),
    }
    matrices = {
        "out_weight": out_weight,
        "out_bias": out_bias,
        "consumer_metric": consumer_metric,
        "no_obj_ptr": (
            torch.empty(0)
            if getattr(model, "no_obj_ptr", None) is None
            else model.no_obj_ptr.detach().float().cpu()
        ),
        "no_obj_embed_spatial": (
            torch.empty(0)
            if getattr(model, "no_obj_embed_spatial", None) is None
            else model.no_obj_embed_spatial.detach().float().cpu()
        ),
        "obj_ptr_final_weight": (
            torch.empty(0)
            if pointer_last is None
            else pointer_last.weight.detach().float().cpu()
        ),
        "obj_ptr_final_bias": (
            torch.empty(0)
            if pointer_last is None or pointer_last.bias is None
            else pointer_last.bias.detach().float().cpu()
        ),
    }
    return report, matrices


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    if left.numel() != right.numel() or left.numel() == 0:
        return None
    return float(torch.nn.functional.cosine_similarity(left.reshape(1, -1), right.reshape(1, -1)))


def _l2_distance(left: torch.Tensor, right: torch.Tensor) -> float | None:
    """L2 distance, or ``None`` when either model lacks the tensor or shapes differ."""

    if left.numel() != right.numel() or left.numel() == 0:
        return None
    return float((left.reshape(-1) - right.reshape(-1)).norm())


def _subspace_similarity(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any] | None:
    left = left.reshape(left.shape[0], -1).double()
    right = right.reshape(right.shape[0], -1).double()
    if left.shape[1] != right.shape[1]:
        return None
    left_basis = torch.linalg.svd(left, full_matrices=False).Vh.T
    right_basis = torch.linalg.svd(right, full_matrices=False).Vh.T
    cosines = torch.linalg.svdvals(left_basis.T @ right_basis).clamp(0, 1)
    return {
        "principal_cosines": [float(item) for item in cosines],
        "mean_principal_cosine": float(cosines.mean()),
    }


def code_audit(sam2_repo: str | Path) -> dict[str, Any]:
    sam2_repo = Path(sam2_repo).resolve()
    commit = verify_sam2_checkout(sam2_repo)
    source_path = sam2_repo / "sam2" / "modeling" / "sam2_base.py"
    paths = {
        "sam2_base": source_path,
        "memory_encoder": sam2_repo / "sam2" / "modeling" / "memory_encoder.py",
        "memory_attention": sam2_repo / "sam2" / "modeling" / "memory_attention.py",
        "transformer": sam2_repo / "sam2" / "modeling" / "sam" / "transformer.py",
    }
    needles = {
        "memory_consumer": ("sam2_base", "def _prepare_memory_conditioned_features("),
        "memory_producer": ("sam2_base", "def _encode_new_memory("),
        "memory_encoder_pix_projection": ("memory_encoder", "x = self.pix_feat_proj(pix_feat)"),
        "memory_encoder_mask_addition": ("memory_encoder", "x = x + masks"),
        "memory_encoder_fuser": ("memory_encoder", "x = self.fuser(x)"),
        "memory_encoder_output_projection": ("memory_encoder", "x = self.out_proj(x)"),
        "memory_encoder_positional_output": ("memory_encoder", "pos = self.position_encoding(x).to(x.dtype)"),
        "pointer_projection": ("sam2_base", "obj_ptr = self.obj_ptr_proj(sam_output_token)"),
        "pointer_token_split": ("sam2_base", "obj_ptrs = obj_ptrs.reshape("),
        "no_object_spatial_addition": ("sam2_base", "maskmem_features += ("),
        "stored_memory_assignment": ("sam2_base", 'current_out["maskmem_features"] = maskmem_features'),
        "stored_positional_assignment": ("sam2_base", 'current_out["maskmem_pos_enc"] = maskmem_pos_enc'),
        "pointer_rope_exclusion": ("memory_attention", 'kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}'),
        "rope_suffix_exclusion": ("transformer", "num_k_rope = k.size(-2) - num_k_exclude_rope"),
    }
    locations: dict[str, Any] = {}
    for name, (path_key, needle) in needles.items():
        path = paths[path_key]
        source = path.read_text(encoding="utf-8").splitlines()
        matches = [index + 1 for index, line in enumerate(source) if needle in line]
        if not matches:
            raise RuntimeError(f"pinned source audit could not find {needle!r}")
        locations[name] = {"path": path.relative_to(sam2_repo).as_posix(), "lines": matches}
    return {
        "schema_version": "cmmt.sam21_code_audit.v1",
        "git_commit": commit,
        "path_policy": "relative_to_sam2_checkout",
        "verified_locations": locations,
        "verified_contract": {
            "maskmem_features_and_positional_encoding_stored_separately": True,
            "no_object_spatial_embedding_added_after_memory_encoder": True,
            "stored_pointer_has_no_temporal_pe": True,
            "pointer_temporal_pe_created_at_consumption": True,
            "pointer_split_into_memory_width_tokens_at_consumption": True,
            "source_positional_encoding_must_not_be_subtracted": True,
        },
    }


def run_checkpoint_audit(
    *,
    sam2_repo: str | Path,
    source_config: str,
    source_checkpoint: str | Path,
    target_config: str,
    target_checkpoint: str | Path,
    output_dir: str | Path,
    device: str = "cpu",
    inference_dtype: str = "bfloat16",
    translator_dtype: str = "float32",
) -> dict[str, Any]:
    sam2_repo = Path(sam2_repo).resolve()
    source_checkpoint = Path(source_checkpoint).resolve()
    target_checkpoint = Path(target_checkpoint).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_source = code_audit(sam2_repo)
    source_model = _load_predictor(sam2_repo, source_config, source_checkpoint, device)
    source_report, source_matrices = _model_audit(
        source_model, device=device, inference_dtype=inference_dtype
    )
    del source_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    target_model = _load_predictor(sam2_repo, target_config, target_checkpoint, device)
    target_report, target_matrices = _model_audit(
        target_model, device=device, inference_dtype=inference_dtype
    )
    del target_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    config_root = sam2_repo / "sam2" / "configs" / "sam2.1"
    config_files = {
        "source": config_root / "sam2.1_hiera_s.yaml",
        "target": config_root / "sam2.1_hiera_b+.yaml",
    }
    git_status = subprocess.run(
        ["git", "-c", f"safe.directory={sam2_repo}", "status", "--short"],
        cwd=sam2_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    report = {
        "schema_version": "cmmt.sam21_checkpoint_audit.v1",
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device": device,
            "inference_dtype": inference_dtype,
            "translator_training_dtype": translator_dtype,
            "git_commit": audit_source["git_commit"],
            "git_status": git_status,
        },
        "inputs": {
            "path_policy": "file_name_and_sha256_only",
            "source_checkpoint": source_checkpoint.name,
            "source_checkpoint_sha256": sha256_file(source_checkpoint),
            "target_checkpoint": target_checkpoint.name,
            "target_checkpoint_sha256": sha256_file(target_checkpoint),
            "source_config": source_config,
            "target_config": target_config,
            "source_config_sha256": sha256_file(config_files["source"]),
            "target_config_sha256": sha256_file(config_files["target"]),
        },
        "source": source_report,
        "target": target_report,
        "cross_model": {
            "out_proj_weight_cosine": _cosine(source_matrices["out_weight"], target_matrices["out_weight"]),
            "out_proj_row_subspace": _subspace_similarity(source_matrices["out_weight"], target_matrices["out_weight"]),
            "no_obj_ptr_cosine": _cosine(source_matrices["no_obj_ptr"], target_matrices["no_obj_ptr"]),
            "no_obj_ptr_l2": _l2_distance(source_matrices["no_obj_ptr"], target_matrices["no_obj_ptr"]),
            "no_obj_embed_spatial_cosine": _cosine(source_matrices["no_obj_embed_spatial"], target_matrices["no_obj_embed_spatial"]),
            "no_obj_embed_spatial_l2": _l2_distance(source_matrices["no_obj_embed_spatial"], target_matrices["no_obj_embed_spatial"]),
            "warning": "weight/subspace similarity does not establish coordinate alignment",
        },
        "code_audit": audit_source,
    }
    json_path = output_dir / "checkpoint_audit.json"
    atomic_write_text(json_path, json.dumps(report, indent=2, ensure_ascii=False))
    matrix_path = output_dir / "checkpoint_matrices.pt"
    atomic_torch_save({"source": source_matrices, "target": target_matrices}, matrix_path)
    code_md = output_dir / "code_audit.md"
    location_lines = "\n".join(
        f"- `{name}`: `{row['path']}` line(s) {', '.join(map(str, row['lines']))}"
        for name, row in audit_source["verified_locations"].items()
    )
    atomic_write_text(
        code_md,
        "# Verified SAM 2.1 memory path\n\n"
        f"- Upstream commit: `{audit_source['git_commit']}`\n"
        "- Producer path: `_track_step`/decoder → `_encode_new_memory` → "
        "`MemoryEncoder` → output dictionary.\n"
        "- Consumer path: `_prepare_memory_conditioned_features` → "
        "target-native temporal assembly → `MemoryAttention`.\n"
        "- `maskmem_features` and `maskmem_pos_enc` are stored separately.\n"
        "- `no_obj_embed_spatial` is added after memory encoding.\n"
        "- Stored `obj_ptr` has no temporal PE; PE and 4×64 tokenization happen at consumption.\n"
        "- Pointer tokens are passed as `num_k_exclude_rope`, so their suffix is excluded from spatial RoPE.\n"
        "- Source PE/RoPE stripping is therefore not part of this experiment.\n\n"
        "## Verified source locations\n\n"
        "Paths are relative to the SAM 2 checkout at the upstream commit above.\n\n"
        f"{location_lines}\n",
    )
    checkpoint_md = output_dir / "checkpoint_audit.md"
    atomic_write_text(
        checkpoint_md,
        "# Checkpoint audit\n\n"
        f"- Source checkpoint SHA-256: `{report['inputs']['source_checkpoint_sha256']}`\n"
        f"- Target checkpoint SHA-256: `{report['inputs']['target_checkpoint_sha256']}`\n"
        f"- Source out-proj condition number: `{source_report['memory_encoder_out_proj']['weight']['condition_number']}`\n"
        f"- Target out-proj condition number: `{target_report['memory_encoder_out_proj']['weight']['condition_number']}`\n"
        "- Full spectra, layer diagnostics, environment, and the SAM 2 checkout's Git status are in `checkpoint_audit.json`.\n",
    )
    return report
