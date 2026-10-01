#!/usr/bin/env python3
"""Fresh affine/MLP/spatial Transformer training on prepared_handoff_case.v2 pairs."""
import argparse
from collections import defaultdict
from pathlib import Path
import math
import random
import time

from core import ROOT, REPO, read, write, sha, digest, snapshot, verify_snapshot
from manifest import library_imports

library_imports()
import torch
from state_pairs import SCHEMA, load_pair
from vos_memory_inspector.transformer_translator import build_translator, SAM21_MEMORY_SPEC
from vos_memory_inspector.upstream import SUPPORTED_SAM2_COMMIT

PRESETS = {"affine": "linear", "residual_mlp": "residual_mlp", "transformer": "base"}
REPORT_SCHEMA = "test10.pair_training.v1"
CHECKPOINT_SCHEMA = "test10.trained_translator.v1"


def collection(path):
    """An explicit video-disjoint train/validation split; examples are never auto-selected."""
    path = Path(path).resolve()
    selection = read(path)
    if selection.get("schema") != "test10.pair_selection.v1":
        raise ValueError("expected test10.pair_selection.v1")
    rows, seen, videos = [], set(), {"train": set(), "validation": set()}
    for item in selection["pairs"]:
        split = item["split"]
        if split not in videos:
            raise ValueError("pair split must be train or validation")
        file = (path.parent / item["path"]).resolve()
        if file in seen:
            raise ValueError("duplicate pair path")
        seen.add(file)
        source, _, metadata = load_pair(file, expected_sha=item.get("sha256"))
        if metadata["video_id"] != item["video_id"]:
            raise ValueError("pair/selection video mismatch")
        video = (item["dataset"], item["video_id"])
        videos[split].add(video)
        rows.append(dict(item, path=str(file), sha256=item.get("sha256") or sha(file),
                         valid_records=source.valid_record_count(),
                         upstream_commit=metadata["upstream_commit"]))
    if not all(videos.values()) or videos["train"] & videos["validation"]:
        raise ValueError("nonempty, video-disjoint train/validation splits required")
    if len({r["upstream_commit"] for r in rows}) != 1:
        raise ValueError("mixed SAM2 upstream commits in pair collection")
    if rows[0]["upstream_commit"] != SUPPORTED_SAM2_COMMIT:
        raise ValueError("pair collection uses a different SAM2 upstream commit")
    return dict(selection=str(path), selection_sha256=sha(path), pairs=rows,
                fingerprint=digest(rows), schema=SCHEMA)


def batches(row, batch_records, device):
    source, target, _ = load_pair(row["path"], expected_sha=row["sha256"])
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


def validate(model, rows, batch_records, device):
    model.eval()
    videos = defaultdict(list)
    with torch.no_grad():
        for row in rows:
            totals = [0., 0., 0.]
            for *tensors, weight in batches(row, batch_records, device):
                for i, loss in enumerate(losses(model, tensors)):
                    totals[i] += float(loss.cpu()) * weight
            videos[row["dataset"], row["video_id"]].append(totals)
    # Equal video weights; checkpoint choice never reads test10 GT evaluation results.
    result = [sum(sum(t[i] for t in group) / len(group) for group in videos.values()) / len(videos)
              for i in range(3)]
    return dict(loss=result[0], spatial_mse=result[1], pointer_mse=result[2], videos=len(videos))


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
    if args.epochs < 1 or args.batch_records < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("epochs, batch-records and learning-rate must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("fresh training requires an empty output directory")
    data = collection(args.pairs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
    code = snapshot([ROOT / "training.py", ROOT / "state_pairs.py", ROOT / "core.py",
                     ROOT / "manifest.py", *sorted((REPO / "src").rglob("*.py"))])
    train_rows = [r for r in data["pairs"] if r["split"] == "train"]
    val_rows = [r for r in data["pairs"] if r["split"] == "validation"]
    report = dict(schema=REPORT_SCHEMA, status="training", state_pair_schema=SCHEMA,
                  collection=data, files=code, seed=args.seed, device=str(device),
                  torch=torch.__version__, config=dict(epochs=args.epochs, learning_rate=args.learning_rate,
                  batch_records=args.batch_records, loss="spatial_MSE + pointer_MSE; valid whole frames",
                  checkpoint_selection="minimum validation state loss, equal video weights"), models={})
    write(args.output_dir / "training_inputs.json", report)
    for method in PRESETS:
        torch.manual_seed(args.seed)
        rng = random.Random(args.seed)
        model = fresh_model(method).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        best, history, steps = math.inf, [], 0
        started = time.monotonic()
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = list(train_rows); rng.shuffle(order)
            epoch_loss = 0.
            for row in order:
                optimizer.zero_grad(set_to_none=True)
                for *tensors, weight in batches(row, args.batch_records, device):
                    loss, _, _ = losses(model, tensors)
                    (loss * weight).backward()
                    epoch_loss += float(loss.detach().cpu()) * weight / len(order)
                if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise ValueError("nonfinite training gradient")
                optimizer.step(); steps += 1
            result = validate(model, val_rows, args.batch_records, device)
            history.append(dict(epoch=epoch, train_loss=epoch_loss, validation=result))
            if result["loss"] < best:
                best = result["loss"]
                path = args.output_dir / f"{method}.pt"
                payload = dict(schema=CHECKPOINT_SCHEMA, method=method, preset=PRESETS[method],
                               epoch=epoch, optimizer_steps=steps, seed=args.seed,
                               training_fingerprint=data["fingerprint"],
                               state_pair_schema=SCHEMA,
                               state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()})
                temp = path.with_suffix(".partial")
                torch.save(payload, temp); temp.replace(path)
                report["models"][method] = dict(checkpoint=path.name, sha256=sha(path),
                    preset=PRESETS[method], epoch=epoch, optimizer_steps=steps,
                    parameters=sum(p.numel() for p in model.parameters()), validation=result, origin="fresh")
            write(args.output_dir / f"{method}.history.json", history)
            print(f"{method} epoch {epoch}/{args.epochs}: train={epoch_loss:.6g}, val={result['loss']:.6g}", flush=True)
        report["models"][method]["seconds"] = time.monotonic() - started
        del model, optimizer
    verify_snapshot(code)
    report["status"] = "complete"
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
    return p


if __name__ == "__main__":
    train(parser().parse_args())
