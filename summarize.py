"""Paired aggregation, bootstrap intervals, and MOSEv2 server imports."""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from runner import atomic_json, resolve_config


def resolve_run_root(config_path, run_id=None):
    _, config = resolve_config(config_path)
    output = Path(config["output"])
    if run_id:
        root = output / run_id
        if not (root / "run.json").is_file():
            raise FileNotFoundError(root / "run.json")
        return root
    candidates = [path for path in output.iterdir() if (path / "run.json").is_file()] if output.exists() else []
    if not candidates:
        raise FileNotFoundError(f"No benchmark runs under {output}")
    return max(candidates, key=lambda path: (path / "run.json").stat().st_mtime_ns)


def bootstrap_interval(values, samples=10000, seed=42):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return [None, None]
    if values.size == 1:
        return [float(values[0]), float(values[0])]
    rng = np.random.default_rng(seed)
    # Chunking avoids allocating samples x videos for large benchmark suites.
    means = []
    remaining = samples
    while remaining:
        count = min(1000, remaining)
        draws = rng.integers(0, len(values), size=(count, len(values)))
        means.append(values[draws].mean(axis=1))
        remaining -= count
    low, high = np.quantile(np.concatenate(means), [0.025, 0.975])
    return [float(low), float(high)]


def write_csv(path, rows, fields):
    path = Path(path)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def paired_official_interval(results, dataset, samples=10000, seed=42):
    """Resample video clusters and recompute the SAME headline statistic."""
    if samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    groups = ("seen", "unseen") if dataset == "lvos" else ("all",)
    sums, counts = [], []
    for result in results:
        cold = {obj["object_id"]: obj for obj in result["metrics"]["cold"]["objects"]}
        totals, sizes = [], []
        for group in groups:
            diffs = [obj["j_and_f"] - cold[obj["object_id"]]["j_and_f"]
                     for obj in result["metrics"]["full_memory"]["objects"]
                     if dataset != "lvos" or obj.get("evaluation_group") == group]
            totals.append(sum(diffs))
            sizes.append(len(diffs))
        sums.append(totals)
        counts.append(sizes)
    sums, counts = np.asarray(sums), np.asarray(counts)
    if not results or np.any(counts.sum(axis=0) == 0):
        return [None, None]
    rng = np.random.default_rng(seed)
    statistics = []
    # LVOS stratification preserves both official groups in each draw.
    for _ in range(samples):
        if dataset == "lvos":
            group_stats = []
            for index in range(2):
                eligible = np.flatnonzero(counts[:, index])
                selected = rng.choice(eligible, size=len(eligible), replace=True)
                group_stats.append(sums[selected, index].sum() / counts[selected, index].sum())
            statistics.append(np.mean(group_stats))
        else:
            selected = rng.integers(0, len(results), size=len(results))
            statistics.append(sums[selected].sum() / counts[selected].sum())
    return np.quantile(statistics, [0.025, 0.975]).tolist()


def summarize_run(config_path, run_id=None):
    _, config = resolve_config(config_path)
    root = resolve_run_root(config_path, run_id)
    groups = defaultdict(list)
    video_rows, object_rows, failures = [], [], []
    combination_failures = [json.loads(path.read_text()) for path in [
        *root.glob("errors/*.json"), *root.glob("*/*/error.json")]]
    for error_path in root.glob("*/*/videos/*/error.json"):
        failures.append(json.loads(error_path.read_text()))
    for path in root.glob("*/*/videos/*/result.json"):
        result = json.loads(path.read_text())
        if result.get("status", "complete") != "complete":
            continue
        combo = path.parents[2]
        selection_path = combo / "selection.json"
        if (combo / "error.json").exists():
            continue
        if selection_path.exists() and result["video"] not in json.loads(selection_path.read_text())["videos"]:
            continue
        identity = result["identity"]
        key = (identity["dataset"], identity["split"], identity["model"])
        groups[key].append(result)
        metrics = result["metrics"]
        row = {
            "dataset": key[0], "split": key[1], "model": key[2],
            "video": result["video"], "partial_run": result["partial_run"],
            "protocol": result.get("protocol", identity.get("protocol", "legacy")),
            "cold_seconds": result["timing_seconds"]["cold"],
            "full_memory_seconds": result["timing_seconds"]["full_memory"],
            "full_memory_total_seconds": result["timing_seconds"]["full_memory_including_first_pass"],
            "full_memory_hits": sum(
                audit["unique_memory_hits"]["first_pass"]
                for audit in result["full_memory_audit"]),
            "future_memory_hits": sum(
                event["is_future"] for audit in result["full_memory_audit"]
                for event in audit["events"]),
            "full_memory_second_pass_writes": sum(
                audit.get("second_pass_writes", 0)
                for audit in result["full_memory_audit"]),
            "overwritten_injected_records": sum(
                audit.get("overwritten_injected_records", 0)
                for audit in result["full_memory_audit"]),
            "removed_injected_records": sum(
                audit.get("removed_injected_records", 0)
                for audit in result["full_memory_audit"]),
            "remaining_injected_records": sum(
                audit.get("remaining_injected_records", 0)
                for audit in result["full_memory_audit"]),
            "all_general_frames_recomputed": all(
                audit.get("all_general_frames_recomputed", False)
                for audit in result["full_memory_audit"]),
        }
        cold_objects = {obj["object_id"]: obj for obj in (metrics["cold"] or {}).get("objects", [])}
        for condition in metrics:
            if metrics[condition] is not None:
                for metric in ("j", "f", "j_and_f"):
                    row[f"{condition}_{metric}"] = metrics[condition][metric]
                for obj in metrics[condition]["objects"]:
                    object_rows.append({
                        "dataset": key[0], "split": key[1], "model": key[2],
                        "video": result["video"], "condition": condition,
                        "object_id": obj["object_id"], "prompt_frame": obj["prompt_frame"],
                        "evaluation_group": obj.get("evaluation_group", "all"),
                        "j": obj["j"], "f": obj["f"], "j_and_f": obj["j_and_f"],
                        "scored_frames": len(obj["frames"]),
                        **{f"delta_{key}": obj[key] - cold_objects[obj["object_id"]][key]
                           for key in ("j", "f", "j_and_f") if obj["object_id"] in cold_objects},
                    })
        if result["full_memory_minus_cold"] is not None:
            for metric, value in result["full_memory_minus_cold"].items():
                row[f"delta_{metric}"] = value
        for condition, delta in result.get("condition_deltas", {}).items():
            if delta:
                row.update({f"{condition}_delta_{key}": value for key, value in delta.items()})
        video_rows.append(row)

    group_rows = []
    for (dataset, split, model), results in sorted(groups.items()):
        scored = [result for result in results if result["full_memory_minus_cold"] is not None]
        row = {"dataset": dataset, "split": split, "model": model,
               "comparison": "full_memory_minus_cold",
               "completed_videos": len(results), "scored_videos": len(scored),
               "partial_run": any(item["partial_run"] for item in results)}
        if scored:
            for condition in ("cold", "full_memory"):
                for metric in ("j", "f", "j_and_f"):
                    row[f"video_macro_{condition}_{metric}"] = float(np.mean([
                        result["metrics"][condition][metric] for result in scored]))
                    objects = [obj for result in scored
                               for obj in result["metrics"][condition]["objects"]]
                    object_macro = float(np.mean([obj[metric] for obj in objects]))
                    row[f"object_macro_{condition}_{metric}"] = object_macro
                    if dataset == "lvos":
                        group_means = {}
                        for group in ("seen", "unseen"):
                            values = [obj[metric] for obj in objects
                                      if obj.get("evaluation_group") == group]
                            if values:
                                group_means[group] = float(np.mean(values))
                                row[f"{condition}_{group}_{metric}"] = group_means[group]
                        # The official LVOS headline gives seen and unseen groups
                        # equal weight. Partial subsets lacking either group do not
                        # claim an official aggregate.
                        primary = (float(np.mean(list(group_means.values())))
                                   if len(group_means) == 2 else None)
                    else:
                        primary = object_macro
                    row[f"{condition}_{metric}"] = primary
            differences = [result["full_memory_minus_cold"]["j_and_f"] for result in scored]
            for metric in ("j", "f", "j_and_f"):
                cold, full_memory = row[f"cold_{metric}"], row[f"full_memory_{metric}"]
                row[f"delta_{metric}"] = (full_memory - cold
                                           if cold is not None and full_memory is not None else None)
                row[f"video_macro_delta_{metric}"] = float(np.mean([
                    r["full_memory_minus_cold"][metric] for r in scored]))
            row["delta_j_and_f_ci95"] = paired_official_interval(
                scored, dataset, config["bootstrap_samples"], config["seed"])
            row["video_macro_delta_j_and_f_ci95"] = bootstrap_interval(
                differences, config["bootstrap_samples"], config["seed"])
            epsilon = 1e-12
            row["improved_videos"] = sum(value > epsilon for value in differences)
            row["equal_videos"] = sum(abs(value) <= epsilon for value in differences)
            row["decreased_videos"] = sum(value < -epsilon for value in differences)
        group_rows.append(row)

    server_results = []
    server_dir = root / "server-results"
    if server_dir.is_dir():
        server_results = [json.loads(path.read_text())
                          for path in sorted(server_dir.glob("*.json"))]
    expected = 0
    for selection in root.glob("*/*/selection.json"):
        expected += len(json.loads(selection.read_text())["videos"])
    summary = {
        "run_id": root.name, "expected_video_jobs": expected,
        "completed_video_jobs": len(video_rows), "failed_video_jobs": len(failures),
        "missing_video_jobs": max(0, expected - len(video_rows) - len(failures)),
        "groups": group_rows, "server_results": server_results,
        "failures": failures, "combination_failures": combination_failures,
        "complete": expected > 0 and len(video_rows) == expected and not failures and not combination_failures,
    }
    atomic_json(root / "summary.json", summary)
    group_fields = sorted({key for row in group_rows for key in row})
    video_fields = sorted({key for row in video_rows for key in row})
    object_fields = sorted({key for row in object_rows for key in row})
    if group_fields:
        write_csv(root / "summary.csv", group_rows, group_fields)
    if video_fields:
        write_csv(root / "videos.csv", video_rows, video_fields)
    if object_fields:
        write_csv(root / "objects.csv", object_rows, object_fields)
    return summary


def import_server_results(config_path, source, submission_id, model, condition,
                          run_id=None):
    if condition not in ("cold", "full_memory"):
        raise ValueError("unknown condition")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", submission_id):
        raise ValueError("submission_id may contain only letters, numbers, dot, underscore, and dash")
    if model not in ("tiny", "small", "plus", "large"):
        raise ValueError("unknown SAM2 model")
    source = Path(source)
    if source.suffix.lower() == ".json":
        metrics = json.loads(source.read_text())
    elif source.suffix.lower() == ".csv":
        with source.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 1:
            raise ValueError("Server CSV must contain exactly one result row")
        metrics = {key: float(value) if value else None for key, value in rows[0].items()}
    else:
        raise ValueError("Server result must be JSON or one-row CSV")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("Server result must be a non-empty metric object")
    root = resolve_run_root(config_path, run_id)
    record = {
        "dataset": "mosev2", "split": "valid", "model": model,
        "condition": condition, "submission_id": submission_id,
        "metrics": metrics, "source_file": str(source.resolve()),
    }
    name = f"{submission_id}-{model}-{condition}.json"
    atomic_json(root / "server-results" / name, record)
    return record
