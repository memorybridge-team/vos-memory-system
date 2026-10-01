#!/usr/bin/env python3
"""Time real state-pair training steps on one GPU without saving checkpoints."""
import argparse
from collections import defaultdict
from pathlib import Path
import json
import statistics
import time

from core import read, rank, sha, write
from training import PRESETS, batches, fresh_model, losses

import torch


def select(selection, per_dataset, seed):
    chosen = []
    for dataset in ("MOSEv2", "LVOSv2"):
        rows = [r for r in selection["pairs"] if r["dataset"] == dataset and r["split"] == "train"]
        rows.sort(key=lambda r: rank(seed, dataset, r["video_id"], r["path"]))
        seen = set()
        for row in rows:
            if row["video_id"] in seen:
                continue
            seen.add(row["video_id"])
            chosen.append(row)
            if len(seen) == per_dataset:
                break
        if len(seen) < per_dataset:
            raise ValueError(f"only {len(seen)} fit videos available for {dataset}")
    return chosen


def one_step(model, optimizer, row, batch_records, device):
    torch.cuda.synchronize()
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    first_batch = None
    records = 0
    for *tensors, weight in batches(row, batch_records, device):
        if first_batch is None:
            torch.cuda.synchronize()
            first_batch = time.perf_counter()
        records += tensors[0].shape[2]
        loss, _, _ = losses(model, tensors)
        (loss * weight).backward()
    optimizer.step()
    torch.cuda.synchronize()
    end = time.perf_counter()
    return dict(seconds=end - start, first_batch_seconds=first_batch - start,
                after_first_batch_seconds=end - first_batch, valid_records=records)


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if args.output.exists():
        raise ValueError(f"output already exists: {args.output}")
    selection = read(args.pairs)
    if selection.get("schema") != "test10.pair_selection.v1":
        raise ValueError("expected test10 pair selection")
    rows = select(selection, args.count // 2, args.seed)
    if args.count != 2 * (args.count // 2):
        raise ValueError("count must be even for equal dataset allocation")
    device = torch.device("cuda")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    header = dict(kind="header", selection_sha256=sha(args.pairs), seed=args.seed,
                  count=len(rows), dataset_counts={"MOSEv2": len(rows)//2, "LVOSv2": len(rows)//2},
                  methods=list(PRESETS), batch_records=args.batch_records,
                  gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                  measurement="one optimizer step per pair; wall time includes checksum, torch.load, H2D, forward/backward, optimizer; CUDA synchronized")
    with args.output.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(header) + "\n"); stream.flush()
        for method in PRESETS:
            torch.manual_seed(args.seed)
            model = fresh_model(method).to(device).train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            # Warm-up is a separate real pair and is excluded from the 500 timed rows.
            one_step(model, optimizer, rows[0], args.batch_records, device)
            torch.cuda.reset_peak_memory_stats()
            for index, row in enumerate(rows, 1):
                timing = one_step(model, optimizer, row, args.batch_records, device)
                record = dict(kind="pair", method=method, index=index,
                              dataset=row["dataset"], video_id=row["video_id"],
                              path=row["path"], **timing)
                stream.write(json.dumps(record) + "\n")
                if index % 10 == 0:
                    stream.flush()
                    print(f"{method}: {index}/{len(rows)} pairs", flush=True)
            stream.write(json.dumps(dict(kind="method_done", method=method,
                peak_allocated_bytes=torch.cuda.max_memory_allocated())) + "\n")
            stream.flush()
            del model, optimizer
            torch.cuda.empty_cache()
    summarize(args.output)


def summarize(path):
    data = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    groups = defaultdict(list)
    for row in data:
        if row["kind"] == "pair":
            groups[row["method"], row["dataset"]].append(row["seconds"])
            groups[row["method"], "all"].append(row["seconds"])
    result = {}
    for (method, dataset), values in groups.items():
        ordered = sorted(values)
        result[f"{method}/{dataset}"] = dict(n=len(values), mean_seconds=statistics.mean(values),
            median_seconds=statistics.median(values), p90_seconds=ordered[int(.9 * (len(ordered)-1))],
            total_seconds=sum(values))
    out = Path(path).with_suffix(".summary.json")
    write(out, result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--batch-records", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
