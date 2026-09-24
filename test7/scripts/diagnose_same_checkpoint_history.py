#!/usr/bin/env python3
"""Compare native and injected SAM 2 history before future continuation.

This is a narrow runtime diagnostic for the same-checkpoint round-trip gate.  It
does not evaluate downstream masks.  Instead, it verifies that the three fields
read by SAM 2 memory attention are identical immediately after injection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

from vos_memory_inspector.device import resolve_device
from vos_memory_inspector.runner import load_binary_prompt
from vos_memory_inspector.sam2_state import (
    canonicalize_sam2_inference_state,
    init_sam2_inference_state_without_warmup,
    inject_sam2_canonical_state,
)
from vos_memory_inspector.upstream import verify_sam2_checkout


def _tensor_difference(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    reference_f = reference.detach().cpu().float()
    candidate_f = candidate.detach().cpu().float()
    if reference_f.shape != candidate_f.shape:
        return {
            "reference_shape": list(reference_f.shape),
            "candidate_shape": list(candidate_f.shape),
            "shape_equal": False,
        }
    difference = candidate_f - reference_f
    return {
        "reference_shape": list(reference_f.shape),
        "candidate_shape": list(candidate_f.shape),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "shape_equal": True,
        "exact_equal": bool(torch.equal(reference_f, candidate_f)),
        "mse": float(difference.square().mean()),
        "max_abs_error": float(difference.abs().max()),
    }


def _compare_histories(
    native_state: dict[str, Any], injected_state: dict[str, Any]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for object_index, native_history in native_state["output_dict_per_obj"].items():
        injected_history = injected_state["output_dict_per_obj"][object_index]
        for storage_key in ("cond_frame_outputs", "non_cond_frame_outputs"):
            for frame_index, native_output in native_history[storage_key].items():
                injected_output = injected_history[storage_key][frame_index]
                rows.append(
                    {
                        "object_index": int(object_index),
                        "storage_key": storage_key,
                        "frame_index": int(frame_index),
                        "maskmem_features": _tensor_difference(
                            native_output["maskmem_features"],
                            injected_output["maskmem_features"],
                        ),
                        "maskmem_pos_enc": _tensor_difference(
                            native_output["maskmem_pos_enc"][-1],
                            injected_output["maskmem_pos_enc"][-1],
                        ),
                        "obj_ptr": _tensor_difference(
                            native_output["obj_ptr"], injected_output["obj_ptr"]
                        ),
                        "injected_keys": sorted(injected_output),
                    }
                )
    summary: dict[str, Any] = {}
    for field in ("maskmem_features", "maskmem_pos_enc", "obj_ptr"):
        field_rows = [row[field] for row in rows]
        summary[field] = {
            "records": len(field_rows),
            "all_exact": all(row.get("exact_equal", False) for row in field_rows),
            "max_abs_error": max(
                (float(row.get("max_abs_error", float("inf"))) for row in field_rows),
                default=0.0,
            ),
        }
    return {"summary": summary, "records": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2-repo", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--prompt-mask", required=True)
    parser.add_argument("--object-id", type=int, required=True)
    parser.add_argument("--switch-frame", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    args.device = resolve_device(args.device, allow_mps=False)

    sam2_repo = Path(args.sam2_repo).resolve()
    verify_sam2_checkout(sam2_repo)
    if str(sam2_repo) not in sys.path:
        sys.path.insert(0, str(sam2_repo))
    from sam2.build_sam import build_sam2_video_predictor

    prompt = load_binary_prompt(args.prompt_mask, args.object_id)
    native_predictor = build_sam2_video_predictor(
        config_file=args.config,
        ckpt_path=str(Path(args.checkpoint).resolve()),
        device=args.device,
    )
    native_state = native_predictor.init_state(
        video_path=str(Path(args.video_dir).resolve()),
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    )
    native_predictor.add_new_mask(
        native_state, frame_idx=0, obj_id=args.object_id, mask=prompt
    )
    for frame_index, _object_ids, _masks in native_predictor.propagate_in_video(
        native_state,
        start_frame_idx=0,
        max_frame_num_to_track=args.switch_frame,
        reverse=False,
    ):
        if int(frame_index) == args.switch_frame:
            break
    canonical = canonicalize_sam2_inference_state(
        native_state, switch_frame=args.switch_frame, strict=True
    )

    injected_predictor = build_sam2_video_predictor(
        config_file=args.config,
        ckpt_path=str(Path(args.checkpoint).resolve()),
        device=args.device,
    )
    injected_state = init_sam2_inference_state_without_warmup(
        injected_predictor,
        video_path=str(Path(args.video_dir).resolve()),
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    )
    injection = inject_sam2_canonical_state(
        canonical, predictor=injected_predictor, inference_state=injected_state
    )
    report = {
        "schema_version": "cmmt.same_checkpoint_history_diagnostic.v1",
        "video_id": Path(args.video_dir).name,
        "switch_frame": args.switch_frame,
        "injection": injection,
        **_compare_histories(native_state, injected_state),
    }
    rendered = json.dumps(report, indent=2)
    if args.json:
        output = Path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2) if args.summary_only else rendered)


if __name__ == "__main__":
    main()
