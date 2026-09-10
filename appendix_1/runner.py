"""End-to-end SAM2 cold-versus-native-prefilled-memory benchmark runner."""
import hashlib
import json
import os
import platform
import random
import resource
import shutil
import tempfile
import time
import traceback
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from PIL import Image

from datasets import acquire, acquisition_plan, discover, sha256
from metrics import evaluate_indexed_masks
from models import (acquire_checkpoint, build_predictor, checkpoint_plan,
                    source_hash, verify_sam2_source)
from protocol import (CONDITIONS, MODELS, PROTOCOL_NAME, audit_summary,
                      instrument_memory_state, memory_record_count,
                      set_audit_active)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False,
                                     suffix=".tmp") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def resolve_config(path):
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text())
    base = path.parent
    for key in ("cache", "output", "sam2_repo"):
        if key in config:
            candidate = Path(config[key]).expanduser()
            config[key] = str(candidate if candidate.is_absolute() else (base / candidate).resolve())
    config.setdefault("sam2_repo", str((Path(__file__).parent / "vendor" / "sam2").resolve()))
    config.setdefault("cache", str((base / "cache").resolve()))
    config.setdefault("output", str((base / "results").resolve()))
    config.setdefault("models", list(MODELS))
    config.setdefault("conditions", list(CONDITIONS))
    if config["conditions"] != list(CONDITIONS):
        raise ValueError("conditions must be exactly cold and full_memory")
    config.setdefault("device", "cuda")
    config.setdefault("precision", "bfloat16")
    config.setdefault("seed", 42)
    config.setdefault("save_masks", True)
    config.setdefault("offload_video_to_cpu", True)
    config.setdefault("offload_state_to_cpu", True)
    config.setdefault("max_videos", None)
    config.setdefault("bootstrap_samples", 10000)
    if config["precision"] not in ("float32", "float16", "bfloat16"):
        raise ValueError("precision must be float32, float16, or bfloat16")
    if config["max_videos"] is not None and int(config["max_videos"]) < 1:
        raise ValueError("max_videos must be a positive integer or null")
    for model in config["models"]:
        if model not in MODELS:
            raise ValueError(f"Unknown model {model!r}; choose from {list(MODELS)}")
    for spec in config["datasets"]:
        for key in ("root", "metadata"):
            if spec.get(key):
                candidate = Path(spec[key]).expanduser()
                spec[key] = str(candidate if candidate.is_absolute() else (base / candidate).resolve())
        if spec["name"] == "lvos" and spec.get("version", "v1") != "v1":
            raise ValueError("This benchmark is locked to LVOS v1")
    checkpoints = config.setdefault("checkpoints", {})
    for model, value in list(checkpoints.items()):
        candidate = Path(value).expanduser()
        checkpoints[model] = str(candidate if candidate.is_absolute() else (base / candidate).resolve())
    return path, config


def experiment_plan(config_path, max_videos=None):
    _, config = resolve_config(config_path)
    if max_videos is not None:
        if max_videos < 1:
            raise ValueError("max_videos must be positive")
        config["max_videos"] = max_videos
    rows = []
    for spec in config["datasets"]:
        for model in config["models"]:
            rows.append({
                "dataset": spec["name"], "split": spec["split"],
                "model": model, "conditions": config["conditions"],
                "max_videos": config["max_videos"],
            })
    return {
        "downloads_started": False,
        "sam2_repo": config["sam2_repo"],
        "checkpoints": {
            model: checkpoint_plan(model, config["cache"], config["checkpoints"].get(model))
            for model in config["models"]
        },
        "datasets": [
            {"name": spec["name"], "split": spec["split"],
             "sources": acquisition_plan(spec)}
            for spec in config["datasets"]
        ],
        "matrix": rows,
    }


def _numeric(value):
    return int(Path(str(value)).stem)


def find_prompts(video):
    """Return object ID -> first labeled source frame and binary mask."""
    arrays = {}

    def load(frame_id):
        if frame_id not in arrays:
            arrays[frame_id] = np.asarray(Image.open(video.masks[frame_id]))
        return arrays[frame_id]

    prompts = {}
    if video.objects:
        candidates = []
        for raw_id, metadata in video.objects.items():
            object_id = int(raw_id)
            start = _numeric(metadata.get("frame_range", {}).get("start", min(video.masks)))
            candidates.append((object_id, start))
        for object_id, start in sorted(candidates):
            for frame_id in sorted(frame for frame in video.masks if frame >= start):
                mask = load(frame_id) == object_id
                if mask.any():
                    prompts[object_id] = (frame_id, mask)
                    break
            if object_id not in prompts:
                raise ValueError(f"Object {object_id} has no initialization mask in {video.name}")
    else:
        for frame_id in sorted(video.masks):
            array = load(frame_id)
            for object_id in sorted(int(value) for value in np.unique(array) if value > 0):
                prompts.setdefault(object_id, (frame_id, array == object_id))
    if not prompts:
        raise ValueError(f"No object prompts found in {video.name}")
    if max(prompts) > 255:
        raise ValueError("Indexed PNG output supports object IDs only up to 255")
    return prompts


def input_manifest(video, prompts):
    files = []
    for path in [*video.frames, *video.masks.values()]:
        files.append((path.name, sha256(path)))
    return stable_hash({
        "video": video.name, "frames": files,
        "objects": video.objects, "frame_ids": video.frame_ids,
        "evaluation_group": video.evaluation_group,
        "prompts": {str(obj): frame for obj, (frame, _) in prompts.items()},
    })


def make_palette():
    palette = []
    for value in range(256):
        red = green = blue = 0
        number = value
        for bit in range(8):
            red |= ((number >> 0) & 1) << (7 - bit)
            green |= ((number >> 1) & 1) << (7 - bit)
            blue |= ((number >> 2) & 1) << (7 - bit)
            number >>= 3
        palette.extend((red, green, blue))
    return palette


PALETTE = make_palette()


class MaskAggregator:
    """Disk-backed exact max-logit merge, avoiding video-sized RAM buffers."""

    def __init__(self, root, conditions):
        self.root = Path(root)
        self.conditions = conditions

    def update(self, condition, frame_index, object_id, logits):
        condition_dir = self.root / condition
        condition_dir.mkdir(parents=True, exist_ok=True)
        score_path = condition_dir / f"{frame_index:08d}.score.npy"
        label_path = condition_dir / f"{frame_index:08d}.label.npy"
        score = np.asarray(logits, dtype=np.float32).squeeze()
        if score.ndim != 2:
            raise ValueError(f"Expected one HxW logit mask, got {score.shape}")
        if score_path.exists():
            best = np.load(score_path)
            labels = np.load(label_path)
        else:
            best = np.full(score.shape, -np.inf, dtype=np.float32)
            labels = np.zeros(score.shape, dtype=np.uint8)
        # Objects are processed by increasing ID; strict comparison gives the
        # first object the same tie behavior as torch.argmax in SAM2's tool.
        replace = (score > best) & (score > 0.0)
        best[replace] = score[replace]
        labels[replace] = object_id
        np.save(score_path, best, allow_pickle=False)
        np.save(label_path, labels, allow_pickle=False)

    def finalize(self, video, output_root, save_masks=True):
        predictions = {condition: {} for condition in self.conditions}
        for condition in self.conditions:
            mask_dir = Path(output_root) / condition / "masks"
            if save_masks:
                mask_dir.mkdir(parents=True, exist_ok=True)
            for index, (frame_id, frame_path) in enumerate(zip(video.frame_ids, video.frames)):
                label_path = self.root / condition / f"{index:08d}.label.npy"
                if label_path.exists():
                    labels = np.load(label_path)
                else:
                    with Image.open(frame_path) as image:
                        labels = np.zeros((image.height, image.width), dtype=np.uint8)
                predictions[condition][frame_id] = labels
                output = Image.fromarray(labels, mode="P")
                output.putpalette(PALETTE)
                if save_masks:
                    output.save(mask_dir / f"{frame_path.stem}.png")
        return predictions


def aggregation_tmpdir():
    """Prefer a RAM-backed temporary filesystem when one is available."""
    requested = os.environ.get("SAM2_AGGREGATION_TMPDIR")
    candidates = [Path(requested)] if requested else []
    candidates.append(Path("/dev/shm"))
    for candidate in candidates:
        try:
            if candidate.is_dir() and os.access(candidate, os.W_OK):
                return str(candidate)
        except OSError:
            pass
    return None


def _inference_context(torch, device, precision):
    stack = ExitStack()
    stack.enter_context(torch.inference_mode())
    if str(device).startswith("cuda"):
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}[precision]
        if dtype != torch.float32:
            stack.enter_context(torch.autocast("cuda", dtype=dtype))
    return stack


def _run_pass(predictor, state, start, object_id, condition, aggregator, torch,
              device, precision, resources=None):
    emitted = 0
    cuda = str(device).startswith("cuda")
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    rss_peak = None
    try:
        import psutil
        process = psutil.Process()
        rss_peak = process.memory_info().rss
    except ImportError:
        process = None
    started = time.perf_counter()
    original = getattr(predictor, "_run_single_frame_inference", None)
    def observed(*args, **kwargs):
        for outputs in state["output_dict_per_obj"].values():
            bank = outputs["non_cond_frame_outputs"]
            if hasattr(bank, "query_frame"):
                bank.query_frame = kwargs["frame_idx"]
        return original(*args, **kwargs)
    with ExitStack() as scope:
        scope.enter_context(_inference_context(torch, device, precision))
        if original is not None:
            predictor._run_single_frame_inference = observed
            scope.callback(setattr, predictor, "_run_single_frame_inference", original)
        for frame_index, object_ids, logits in predictor.propagate_in_video(
                state, start_frame_idx=start, reverse=False):
            ids = [int(item) for item in object_ids]
            position = ids.index(object_id)
            aggregator.update(condition, frame_index, object_id,
                              logits[position].detach().float().cpu().numpy())
            emitted += 1
            if process:
                rss_peak = max(rss_peak, process.memory_info().rss)
    if cuda:
        torch.cuda.synchronize(device)
    if resources is not None:
        prior = resources.setdefault(condition, {})
        prior["sampled_cpu_rss_bytes"] = max(prior.get("sampled_cpu_rss_bytes") or 0, rss_peak or 0) or None
        prior["peak_cuda_allocated_bytes"] = max(prior.get("peak_cuda_allocated_bytes") or 0,
            int(torch.cuda.max_memory_allocated(device)) if cuda else 0) or None
    return time.perf_counter() - started, emitted


def _max_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def run_video(predictor, video, output_dir, config, identity):
    import torch

    output_dir = Path(output_dir)
    conditions = tuple(config.get("conditions", CONDITIONS))
    save_masks = config.get("save_masks", True) or not video.score_available
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "result.json"
    try:
        prompts = find_prompts(video)
        expected = {**identity, "input_manifest": input_manifest(video, prompts)}
    except Exception:
        status_path.unlink(missing_ok=True)
        for condition in conditions:
            condition_dir = output_dir / condition
            if condition_dir.exists():
                shutil.rmtree(condition_dir)
        raise
    if status_path.is_file():
        prior = json.loads(status_path.read_text())
        mask_counts = {
            condition: len(list((Path(output_dir) / condition / "masks").glob("*.png")))
            for condition in conditions
        }
        if (prior.get("identity") == expected and prior.get("status") == "complete"
                and (not save_masks or all(count == len(video.frames) for count in mask_counts.values()))):
            return prior, True

    status_path.unlink(missing_ok=True)
    (output_dir / "error.json").unlink(missing_ok=True)
    for condition in conditions:
        condition_dir = output_dir / condition
        if condition_dir.exists():
            shutil.rmtree(condition_dir)
    with tempfile.TemporaryDirectory(prefix="sam2-aggregate-",
                                      dir=aggregation_tmpdir()) as temporary:
        aggregator = MaskAggregator(temporary, conditions)
        state = predictor.init_state(
            video_path=str(video.frames[0].parent),
            offload_video_to_cpu=config["offload_video_to_cpu"],
            offload_state_to_cpu=config["offload_state_to_cpu"],
            async_loading_frames=False,
        )
        timings = {condition: 0.0 for condition in conditions}
        emitted = {condition: 0 for condition in conditions}
        memory = []
        full_memory_audits = []
        pass_resources = {}
        for object_id, (prompt_frame, prompt_mask) in sorted(prompts.items()):
            predictor.reset_state(state)
            start = video.position(prompt_frame)
            with _inference_context(torch, config["device"], config["precision"]):
                predictor.add_new_mask(state, frame_idx=start, obj_id=object_id,
                                       mask=prompt_mask)
            duration, count = _run_pass(
                predictor, state, start, object_id, "cold", aggregator, torch,
                config["device"], config["precision"], pass_resources)
            timings["cold"] += duration
            emitted["cold"] += count
            records_before = memory_record_count(state)
            snapshot = dict(state["output_dict_per_obj"][0]["non_cond_frame_outputs"])
            predictor.reset_state(state)
            with _inference_context(torch, config["device"], config["precision"]):
                predictor.add_new_mask(state, frame_idx=start, obj_id=object_id, mask=prompt_mask)
            state["output_dict_per_obj"][0]["non_cond_frame_outputs"] = snapshot
            audit_maps = instrument_memory_state(state, start)
            set_audit_active(audit_maps, True)
            try:
                duration, count = _run_pass(
                    predictor, state, start, object_id, "full_memory", aggregator,
                    torch, config["device"], config["precision"], pass_resources)
            finally:
                set_audit_active(audit_maps, False)
            timings["full_memory"] += duration
            emitted["full_memory"] += count
            second_pass_writes = sum(bank.writes for bank in audit_maps.values())
            expected_second_pass_writes = max(0, count - 1)
            full_memory_audits.append({
                "object_id": object_id,
                **audit_summary(audit_maps),
                "second_pass_writes": second_pass_writes,
                "expected_second_pass_writes": expected_second_pass_writes,
                "all_general_frames_recomputed": (
                    second_pass_writes == expected_second_pass_writes),
                "overwritten_injected_records": sum(
                    len(bank.overwritten_first_pass_keys) for bank in audit_maps.values()),
                "removed_injected_records": sum(
                    len(bank.removed_first_pass_keys) for bank in audit_maps.values()),
                "remaining_injected_records": sum(
                    bank.remaining_first_pass_records() for bank in audit_maps.values()),
            })
            memory.append({
                "object_id": object_id,
                "first_pass_records": records_before,
                "injected_non_conditioning_records": len(snapshot),
                "records_after_full_memory": memory_record_count(state),
            })
        predictions = aggregator.finalize(video, output_dir, save_masks)

    metrics = {}
    prompt_frames = {object_id: value[0] for object_id, value in prompts.items()}
    for condition in conditions:
        metrics[condition] = (evaluate_indexed_masks(video, predictions[condition], prompt_frames)
                              if video.score_available else None)
    delta = ({key: metrics["full_memory"][key] - metrics["cold"][key]
              for key in ("j", "f", "j_and_f")} if video.score_available else None)
    result = {
        "status": "complete", "protocol": PROTOCOL_NAME,
        "identity": expected, "video": video.name,
        "partial_run": config["max_videos"] is not None,
        "frame_count": len(video.frames),
        "prompt_frames": {str(key): value for key, value in prompt_frames.items()},
        "metrics": metrics, "full_memory_minus_cold": delta,
        "timing_seconds": {**timings,
                           "full_memory_including_first_pass": timings["cold"] + timings["full_memory"]},
        "emitted_frames": emitted, "memory": memory,
        "full_memory_audit": full_memory_audits,
        "pass_resources": pass_resources,
        "condition_deltas": {condition: {key: metrics[condition][key] - metrics["cold"][key]
            for key in ("j", "f", "j_and_f")} if video.score_available else None
            for condition in conditions if condition != "cold"},
        "timing_note": "Propagation plus mask aggregation I/O; full_memory includes audit overhead. Initialization excluded.",
        "peak_cpu_rss_bytes": _max_rss_bytes(),
        "peak_cuda_allocated_bytes": max((item["peak_cuda_allocated_bytes"] or 0
            for item in pass_resources.values()), default=0) or None,
    }
    atomic_json(status_path, result)
    return result, False


def _runtime_metadata(torch, config):
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "cuda_device": (torch.cuda.get_device_name(config["device"])
                        if str(config["device"]).startswith("cuda") else None),
        "precision": config["precision"], "seed": config["seed"],
    }


def run_benchmark(config_path, max_videos=None):
    import torch

    config_path, config = resolve_config(config_path)
    if max_videos is not None:
        if max_videos < 1:
            raise ValueError("max_videos must be positive")
        config["max_videos"] = max_videos
    if str(config["device"]).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    sam2_repo, sam2_commit = verify_sam2_source(config["sam2_repo"])
    code_digest = source_hash(Path(__file__).parent)
    config_digest = stable_hash(config)
    runtime = _runtime_metadata(torch, config)
    run_id = stable_hash({"config": config_digest, "code": code_digest,
                          "sam2": sam2_commit, "runtime": runtime})[:16]
    run_root = Path(config["output"]) / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    atomic_json(run_root / "run.json", {
        "run_id": run_id, "config": config, "config_hash": config_digest,
        "code_hash": code_digest, "sam2_commit": sam2_commit,
        "protocol": PROTOCOL_NAME, "runtime": runtime,
    })
    report = {"run_id": run_id, "completed": 0, "resumed": 0,
              "failed": 0, "failed_combinations": 0}
    for model_name in config["models"]:
        print(f"[model] {model_name}", flush=True)
        try:
            checkpoint, checkpoint_hash = acquire_checkpoint(
                model_name, config["cache"], config["checkpoints"].get(model_name))
            predictor, model_meta = build_predictor(
                model_name, checkpoint, sam2_repo, config["device"])
            (run_root / "errors" / f"model-{model_name}.json").unlink(missing_ok=True)
        except Exception as exc:
            report["failed_combinations"] += len(config["datasets"])
            atomic_json(run_root / "errors" / f"model-{model_name}.json", {
                "status": "failed", "model": model_name, "error": str(exc),
                "traceback": traceback.format_exc(),
            })
            continue
        model_identity = {
            "run_id": run_id, "model": model_name,
            "checkpoint_sha256": checkpoint_hash,
            "protocol": PROTOCOL_NAME, **model_meta,
        }
        for spec in config["datasets"]:
            print(f"[dataset] {spec['name']}/{spec['split']} model={model_name}", flush=True)
            combo = run_root / f"{spec['name']}_{spec['split']}" / model_name
            try:
                dataset_root = acquire(spec, config["cache"])
                videos = discover(spec, dataset_root)
                (combo / "error.json").unlink(missing_ok=True)
            except Exception as exc:
                report["failed_combinations"] += 1
                atomic_json(combo / "error.json", {
                    "status": "failed", "dataset": spec["name"],
                    "split": spec["split"], "model": model_name,
                    "error": str(exc), "traceback": traceback.format_exc(),
                })
                continue
            videos.sort(key=lambda item: item.name)
            if config["max_videos"] is not None:
                videos = videos[:int(config["max_videos"])]
            # A fixed remote root can be reused one shard at a time. Preserve
            # the union of previously processed videos so that summarize()
            # does not discard completed shards when selection.json is updated.
            selection_path = combo / "selection.json"
            selected_names = {video.name for video in videos}
            if selection_path.is_file():
                prior_selection = json.loads(selection_path.read_text())
                selected_names.update(prior_selection.get("videos", []))
            atomic_json(selection_path, {
                "videos": sorted(selected_names),
                "partial_run": config["max_videos"] is not None,
            })
            for video in videos:
                print(f"[video] {video.name}", flush=True)
                output_dir = combo / "videos" / video.name
                identity = {**model_identity, "dataset": spec["name"],
                            "split": spec["split"]}
                try:
                    if str(config["device"]).startswith("cuda"):
                        torch.cuda.reset_peak_memory_stats()
                    _, resumed = run_video(predictor, video, output_dir,
                                           config, identity)
                    report["resumed" if resumed else "completed"] += 1
                except Exception as exc:
                    report["failed"] += 1
                    atomic_json(output_dir / "error.json", {
                        "status": "failed", "video": video.name,
                        "error": str(exc), "traceback": traceback.format_exc(),
                        "identity": identity,
                    })
            if spec["name"] == "mosev2" and spec["split"] == "valid":
                for condition in config["conditions"]:
                    source = combo / "videos"
                    staging = combo / f"submission-{condition}"
                    if staging.exists():
                        shutil.rmtree(staging)
                    missing = []
                    selected = json.loads((combo / "selection.json").read_text())["videos"]
                    for video_name in selected:
                        masks = source / video_name / condition / "masks"
                        result_path = source / video_name / "result.json"
                        if masks.is_dir() and result_path.is_file():
                            shutil.copytree(masks, staging / video_name)
                        else:
                            missing.append(video_name)
                    archive_base = combo / f"mosev2-valid-{model_name}-{condition}"
                    archive = archive_base.with_suffix(".zip")
                    if missing:
                        archive.unlink(missing_ok=True)
                    elif staging.exists():
                        shutil.make_archive(str(archive_base), "zip", staging)
                    if staging.exists():
                        shutil.rmtree(staging)
                    atomic_json(combo / f"submission-{condition}.json", {
                        "condition": condition, "archive": str(archive) if not missing else None,
                        "complete": not missing, "missing_videos": missing,
                    })
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    atomic_json(run_root / "run-status.json", report)
    return report
