#!/usr/bin/env python3
"""Fresh affine/MLP/spatial Transformer training on prepared_handoff_case.v2 pairs."""
import argparse
from collections import defaultdict, OrderedDict, deque
from pathlib import Path
import math
import random
import time

from core import ROOT, REPO, read, write, sha, digest, snapshot, verify_snapshot
from manifest import library_imports

library_imports()
import torch
from state_pairs import SCHEMA, MODELS, load_pair
from vos_memory_inspector.transformer_translator import build_translator, SAM21_MEMORY_SPEC
from vos_memory_inspector.upstream import SUPPORTED_SAM2_COMMIT

PRESETS = {"affine": "linear", "residual_mlp": "residual_mlp", "transformer": "base"}
REPORT_SCHEMA = "test10.pair_training.v1"
CHECKPOINT_SCHEMA = "test10.trained_translator.v1"


def signature(path):
    s = Path(path).stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def pair_files(path):
    path = Path(path)
    return (path, path.with_suffix(".pt.sha256"), path.with_suffix(".prepare.json"))


class PairReader:
    """Bounded CPU cache. Files changing during this run are rejected, even on cache hits."""
    def __init__(self, max_bytes=0):
        self.max_bytes = max_bytes
        self.cache = OrderedDict()
        self.bytes = 0
        self.verified = set()

    def load(self, row):
        path = row["path"]
        current = [list(signature(p)) for p in pair_files(path)]
        if current != row["file_signatures"]:
            raise ValueError(f"pair files changed during training: {path}")
        if path in self.cache:
            result, size = self.cache.pop(path)
            self.cache[path] = (result, size)
            return result
        result = load_pair(path, expected_sha=row["sha256"])
        if (result[2]["video_id"] != row["video_id"] or
                result[2]["upstream_commit"] != row["upstream_commit"] or
                result[0].valid_record_count() != row["valid_records"]):
            raise ValueError(f"pair/selection contract mismatch: {path}")
        if [list(signature(p)) for p in pair_files(path)] != current:
            raise ValueError(f"pair files changed while loading: {path}")
        self.verified.add(path)
        size = max(Path(path).stat().st_size,
                   sum(t.numel() * t.element_size() for state in result[:2]
                       for t in vars(state).values() if isinstance(t, torch.Tensor)))
        if size <= self.max_bytes:
            while self.cache and self.bytes + size > self.max_bytes:
                _, (_, old_size) = self.cache.popitem(last=False)
                self.bytes -= old_size
            self.cache[path] = (result, size)
            self.bytes += size
        return result


def balanced_order(rows, seed, equal_datasets=False):
    """Visit every pair once, cycle videos; dataset proportions match the collection."""
    rng = random.Random(seed)
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row["dataset"]][row["video_id"]].append(row)
    queues, sizes = {}, {}
    for dataset, videos in sorted(groups.items()):
        buckets = []
        for video in sorted(videos):
            bucket = videos[video][:]; rng.shuffle(bucket); buckets.append(deque(bucket))
        rng.shuffle(buckets)
        q = []
        while buckets:
            for bucket in buckets:
                q.append(bucket.popleft())
            buckets = [b for b in buckets if b]
        queues[dataset] = deque(q)
        sizes[dataset] = 1 if equal_datasets else len(q)
    spent = dict.fromkeys(queues, 0)
    order = []
    while any(queues.values()):
        dataset = min((d for d in queues if queues[d]), key=lambda d: (spent[d] / sizes[d], d))
        order.append(queues[dataset].popleft()); spent[dataset] += 1
    return order


def coverage(rows, total):
    unique = {r["path"]: r for r in rows}
    return dict(unique_pairs=len(unique), unique_videos=len({(r["dataset"], r["video_id"]) for r in unique.values()}),
                available_pairs=total, datasets={d: sum(r["dataset"] == d for r in unique.values())
                for d in sorted({r["dataset"] for r in rows})}, paths=sorted(unique))


def collection(path, *, lazy=False, deadline=None):
    """An explicit video-disjoint train/validation split; examples are never auto-selected."""
    path = Path(path).resolve()
    selection = read(path)
    if selection.get("schema") != "test10.pair_selection.v1":
        raise ValueError("expected test10.pair_selection.v1")
    rows, seen, videos = [], set(), {"train": set(), "validation": set()}
    for item in selection["pairs"]:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("training budget exhausted while reading collection")
        split = item["split"]
        if split not in videos:
            raise ValueError("pair split must be train or validation")
        file = (path.parent / item["path"]).resolve()
        if file in seen:
            raise ValueError("duplicate pair path")
        seen.add(file)
        if lazy:
            # Only small sidecars here. Each consumed tensor pair is fully verified by PairReader.
            metadata = read(file.with_suffix(".prepare.json"))
            checksum = file.with_suffix(".pt.sha256").read_text(encoding="ascii").split()[0]
            if (len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum) or
                    metadata["cache"]["schema_version"] != SCHEMA or metadata["cache"]["sha256"] != checksum or
                    any(metadata.get(k) != v for k, v in MODELS.items())):
                raise ValueError(f"invalid prepared pair sidecar: {file}")
            records = metadata["source_contract"]["valid_record_count"]
            if records < 1 or records != metadata["target_contract"]["valid_record_count"]:
                raise ValueError(f"invalid pair record counts: {file}")
        else:
            source, _, metadata = load_pair(file)
            checksum = file.with_suffix(".pt.sha256").read_text(encoding="ascii").split()[0]
            records = source.valid_record_count()
        if metadata["video_id"] != item["video_id"]:
            raise ValueError("pair/selection video mismatch")
        video = (item["dataset"], item["video_id"])
        videos[split].add(video)
        rows.append(dict(item, path=str(file), sha256=checksum,
                         valid_records=records, file_signatures=[list(signature(p)) for p in pair_files(file)],
                         upstream_commit=metadata["upstream_commit"]))
    if not all(videos.values()) or videos["train"] & videos["validation"]:
        raise ValueError("nonempty, video-disjoint train/validation splits required")
    if len({r["upstream_commit"] for r in rows}) != 1:
        raise ValueError("mixed SAM2 upstream commits in pair collection")
    if rows[0]["upstream_commit"] != SUPPORTED_SAM2_COMMIT:
        raise ValueError("pair collection uses a different SAM2 upstream commit")
    return dict(selection=str(path), selection_sha256=sha(path), pairs=rows,
                fingerprint=digest(rows), schema=SCHEMA)


def batches(row, batch_records, device, reader=None):
    source, target, _ = (reader.load(row) if reader else load_pair(row["path"], expected_sha=row["sha256"]))
    valid = source.validity
    # Whole CxHxW frames for every method: Transformer must not see sampled pixels.
    x, p = source.spatial_memory[valid], source.object_pointer[valid]
    y, q = target.spatial_memory[valid], target.object_pointer[valid]
    n = len(x)
    for start in range(0, n, batch_records):
        end = min(start + batch_records, n)
        tensors = [t[start:end].unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
                   for t in (x, p, y, q)]
        yield (*tensors, (end - start) / n)


def losses(model, tensors):
    x, p, y, q = tensors
    spatial, pointer = model.translate_tensors(x, p)
    spatial_loss = (spatial - y).square().mean()
    pointer_loss = (pointer - q).square().mean()
    loss = spatial_loss + pointer_loss
    if not torch.isfinite(loss):
        raise ValueError("nonfinite training/validation loss")
    return loss, spatial_loss, pointer_loss


def validate(model, rows, batch_records, device, reader=None, deadline=None):
    model.eval()
    videos = defaultdict(list)
    with torch.no_grad():
        for row in rows:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("validation budget exhausted; partial validation cannot select a checkpoint")
            totals = [0., 0., 0.]
            for *tensors, weight in batches(row, batch_records, device, reader):
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("validation budget exhausted; partial validation cannot select a checkpoint")
                for i, loss in enumerate(losses(model, tensors)):
                    totals[i] += float(loss.cpu()) * weight
            videos[row["dataset"], row["video_id"]].append(totals)
    # Equal video weights; checkpoint choice never reads test10 GT evaluation results.
    result = [sum(sum(t[i] for t in group) / len(group) for group in videos.values()) / len(videos)
              for i in range(3)]
    return dict(loss=result[0], spatial_mse=result[1], pointer_mse=result[2], videos=len(videos), pairs=len(rows))


def fresh_model(method):
    model = build_translator(PRESETS[method], SAM21_MEMORY_SPEC, SAM21_MEMORY_SPEC)
    # All methods start at identity; no historical MLP/Transformer weights enter training.
    with torch.no_grad():
        if method == "affine":
            for layer in (model.feature, model.pointer):
                layer.weight.copy_(torch.eye(layer.out_features, layer.in_features))
                layer.bias.zero_()
        elif method == "residual_mlp":
            for layer in (model.feature[-1], model.pointer[-1]):
                layer.weight.zero_(); layer.bias.zero_()
    return model


def train(args):
    started_all = time.monotonic()
    budget_hours = getattr(args, "budget_hours", None)
    val_limit = getattr(args, "validation_pairs", 64)
    interval = getattr(args, "validate_every", 2048)
    cache_gib = getattr(args, "cache_gib", 2.)
    if (args.epochs < 1 or args.batch_records < 1 or not math.isfinite(args.learning_rate) or
            args.learning_rate <= 0 or val_limit < 1 or interval < 1 or
            not math.isfinite(cache_gib) or cache_gib < 0 or
            (budget_hours is not None and (not math.isfinite(budget_hours) or budget_hours <= 0))):
        raise ValueError("positive epochs, batch-records, learning-rate, validation-pairs, validate-every and budget required")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("fresh training requires an empty output directory")
    deadline = started_all + budget_hours * 3600 if budget_hours is not None else None
    data = collection(args.pairs, lazy=deadline is not None, deadline=deadline)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
    code = snapshot([ROOT / "training.py", ROOT / "state_pairs.py", ROOT / "core.py",
                     ROOT / "manifest.py", *sorted((REPO / "src").rglob("*.py"))])
    train_rows = [r for r in data["pairs"] if r["split"] == "train"]
    all_val_rows = [r for r in data["pairs"] if r["split"] == "validation"]
    val_rows = (balanced_order(all_val_rows, args.seed, equal_datasets=True)[:val_limit]
                if deadline is not None else all_val_rows)
    # Keep a fixed validation cohort, chosen without scores. Cache bounds exclude transient GPU batches.
    train_reader = PairReader(int(cache_gib * 1024**3 / 2))
    val_reader = PairReader(int(cache_gib * 1024**3 / 2))
    report = dict(schema=REPORT_SCHEMA, status="training", state_pair_schema=SCHEMA,
                  collection=data, files=code, seed=args.seed, device=str(device),
                  torch=torch.__version__, config=dict(epochs=args.epochs, learning_rate=args.learning_rate,
                  batch_records=args.batch_records, budget_hours=budget_hours, cache_gib=cache_gib,
                  validate_every=interval if deadline is not None else "epoch",
                  validation=coverage(val_rows, len(all_val_rows)),
                  pair_verification="full checksum/tensor validation at first use" if deadline else "eager and at use",
                  loss="spatial_MSE + pointer_MSE; valid whole frames",
                  checkpoint_selection="minimum fixed-cohort validation state loss, equal video weights"),
                  models={}, attempts={})
    write(args.output_dir / "training_inputs.json", report)
    write(args.output_dir / "train_report.json", report)
    for index, method in enumerate(PRESETS):
        started = time.monotonic()
        method_deadline = (started + max(0., deadline - started) / (len(PRESETS) - index)
                           if deadline is not None else None)
        reserve = max(30., (method_deadline - started) * .08) if method_deadline else 0.
        training_stop = method_deadline - reserve if method_deadline else None
        torch.manual_seed(args.seed)
        model = fresh_model(method).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        best, history, steps, last_validated = math.inf, [], 0, 0
        used = []
        total_loss = 0.
        stop_reason = "epochs_completed"

        def checkpoint(epoch, epoch_steps):
            nonlocal best, last_validated, reserve, training_stop
            t = time.monotonic()
            try:
                result = validate(model, val_rows, args.batch_records, device, val_reader, method_deadline)
            except TimeoutError:
                return False
            last_validated = steps
            history.append(dict(epoch=epoch, epoch_pairs=epoch_steps, epoch_complete=epoch_steps == len(train_rows),
                                optimizer_steps=steps, train_loss=epoch_loss / epoch_steps,
                                cumulative_train_loss=total_loss / steps, validation=result))
            if result["loss"] < best:
                best = result["loss"]
                path = args.output_dir / f"{method}.pt"
                payload = dict(schema=CHECKPOINT_SCHEMA, method=method, preset=PRESETS[method],
                               epoch=epoch, optimizer_steps=steps, seed=args.seed,
                               training_fingerprint=data["fingerprint"], state_pair_schema=SCHEMA,
                               state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()})
                temp = path.with_suffix(".partial")
                torch.save(payload, temp); temp.replace(path)
                report["models"][method] = dict(checkpoint=path.name, sha256=sha(path),
                    preset=PRESETS[method], epoch=epoch, optimizer_steps=steps,
                    parameters=sum(p.numel() for p in model.parameters()), validation=result, origin="fresh",
                    training_coverage=coverage(used, len(train_rows)), effective_epochs=steps / len(train_rows))
            write(args.output_dir / f"{method}.history.json", history)
            write(args.output_dir / "train_report.json", report)
            print(f"{method} epoch {epoch}/{args.epochs}, steps={steps}: "
                  f"train={epoch_loss / epoch_steps:.6g}, val={result['loss']:.6g}", flush=True)
            if method_deadline:
                reserve = max(reserve, (time.monotonic() - t) * 1.5 + 15.)
                training_stop = method_deadline - reserve
            model.train()
            return True

        for epoch in range(1, args.epochs + 1):
            model.train()
            order = balanced_order(train_rows, args.seed + epoch - 1)
            epoch_steps = 0
            epoch_loss = 0.
            for row in order:
                if training_stop is not None and time.monotonic() >= training_stop:
                    stop_reason = "time_budget"
                    break
                optimizer.zero_grad(set_to_none=True)
                pair_loss, interrupted = 0., False
                for *tensors, weight in batches(row, args.batch_records, device, train_reader):
                    if training_stop is not None and time.monotonic() >= training_stop:
                        interrupted = True
                        break
                    loss, _, _ = losses(model, tensors)
                    (loss * weight).backward()
                    pair_loss += float(loss.detach().cpu()) * weight
                if interrupted:
                    optimizer.zero_grad(set_to_none=True)
                    stop_reason = "time_budget"
                    break  # Never step on an incomplete pair's gradient.
                finite = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
                if not finite or not bool(torch.stack(finite).all()):
                    raise ValueError("nonfinite training gradient")
                optimizer.step(); steps += 1; epoch_steps += 1
                used.append(row); total_loss += pair_loss; epoch_loss += pair_loss
                if deadline is not None and steps % interval == 0:
                    if not checkpoint(epoch, epoch_steps):
                        stop_reason = "validation_time_budget"
                        break
            if steps > last_validated:
                if not checkpoint(epoch, epoch_steps):
                    stop_reason = "validation_time_budget"
            if stop_reason != "epochs_completed":
                break
        report["attempts"][method] = dict(seconds=time.monotonic() - started, optimizer_steps=steps,
            effective_epochs=steps / len(train_rows), training_coverage=coverage(used, len(train_rows)),
            stop_reason=stop_reason, validated_checkpoint=method in report["models"])
        if method in report["models"]:
            report["models"][method]["seconds"] = time.monotonic() - started
        write(args.output_dir / "train_report.json", report)
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    verify_snapshot(code)
    verified = (train_reader.verified | val_reader.verified if deadline is not None else
                {r["path"] for r in data["pairs"]})
    report["integrity"] = dict(fully_verified_pairs=len(verified), collection_pairs=len(data["pairs"]),
                               unconsumed_pairs_not_tensor_verified=len(data["pairs"]) - len(verified))
    report["seconds"] = time.monotonic() - started_all
    # Complete means three trained, fully validated checkpoints exist; it does not mean 30 epochs ran.
    report["status"] = "complete" if set(report["models"]) == set(PRESETS) else "incomplete"
    write(args.output_dir / "train_report.json", report)
    return report


def training_report(directory):
    directory = Path(directory).resolve()
    report = read(directory / "train_report.json")
    if report.get("schema") != REPORT_SCHEMA or report.get("status") != "complete":
        raise ValueError("completed fresh pair training report required")
    if report.get("state_pair_schema") != SCHEMA:
        raise ValueError("training pair schema mismatch")
    verify_snapshot(report["files"])
    if set(report["models"]) != set(PRESETS):
        raise ValueError("affine/MLP/Transformer training must all be complete")
    for method, row in report["models"].items():
        if row.get("origin") != "fresh" or row["optimizer_steps"] < 1 or row["preset"] != PRESETS[method]:
            raise ValueError("old or untrained translator is not allowed")
        if sha(directory / row["checkpoint"]) != row["sha256"]:
            raise ValueError(f"trained checkpoint checksum mismatch: {method}")
    return report


def load_models(directory, device):
    report = training_report(directory)
    models = {}
    for method, row in report["models"].items():
        payload = torch.load(Path(directory) / row["checkpoint"], map_location="cpu", weights_only=True)
        if (payload.get("schema") != CHECKPOINT_SCHEMA or payload.get("method") != method or
                payload.get("state_pair_schema") != SCHEMA or payload.get("preset") != row["preset"] or
                payload.get("training_fingerprint") != report["collection"]["fingerprint"] or
                payload.get("optimizer_steps") != row["optimizer_steps"] or payload.get("epoch") != row["epoch"]):
            raise ValueError("checkpoint and training report mismatch")
        model = fresh_model(method)
        model.load_state_dict(payload["state_dict"], strict=True)
        models[method] = model.to(device).eval()
    return models


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", type=Path, required=True, help="test10.pair_selection.v1 JSON")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--batch-records", type=int, default=4, help="whole memory frames per gradient microbatch")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--budget-hours", type=float, help="total training wall time, including input setup and validation")
    p.add_argument("--validation-pairs", type=int, default=64, help="fixed dataset/video stratified validation cap in budget mode")
    p.add_argument("--validate-every", type=int, default=2048, help="pair optimizer steps between validation in budget mode")
    p.add_argument("--cache-gib", type=float, default=2., help="combined train/validation CPU tensor cache limit")
    return p


if __name__ == "__main__":
    result = train(parser().parse_args())
    if result["status"] != "complete":
        raise SystemExit("budget ended without all three validated checkpoints; inspect train_report.json")
