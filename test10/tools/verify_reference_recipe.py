#!/usr/bin/env python3
"""Differential check: test10 Affine training vs the reference LVOS DDP state trainer.

The reference trainer (vos-memory-translator-nonlinear, lvos_ddp.py) is locked to the
Transformer body, so it cannot train Affine as shipped. This tool runs the pinned
reference revision unmodified except for two injected functions: the model factory
returns an identity-initialised Affine, and frozen_cases returns a synthetic LVOS case
list. Raw cache files, raw_index.json, scheduling, loss, optimizer, scheduler, clipping,
DDP accumulation and development evaluation all run through the reference code.
test10 then trains on the same raw_index.json and both trajectories are compared.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]

REFERENCE_SCRIPT = r'''
import json, os, sys
from pathlib import Path
import torch
torch.set_num_threads(1)
work, epochs = Path(sys.argv[1]), int(sys.argv[2])
from vos_memory_inspector import lvos_raw_cache as raw, lvos_ddp as ddp
from vos_memory_inspector.case_cache import write_case_cache
from vos_memory_inspector.state_schema import CanonicalState
from vos_memory_inspector.frozen_tensor_api import _LearnedStateTranslator
from vos_memory_inspector.transformer_translator import SAM21_MEMORY_SPEC
from vos_memory_inspector.lvos_training import LVOSConfig

FIT, DEV, SLOTS = [16, 9, 13, 16, 5, 16, 11, 7], [12, 16], 16
g = torch.Generator().manual_seed(20261005)
mix_s = torch.eye(64) + .05 * torch.randn(64, 64, generator=g); shift_s = .1 * torch.randn(64, generator=g)
mix_p = torch.eye(256) + .02 * torch.randn(256, 256, generator=g); shift_p = .1 * torch.randn(256, generator=g)

def state(spatial, pointer, k):
    frames = torch.full((1, 1, SLOTS), -1, dtype=torch.int64); frames[0, 0, :k] = torch.arange(k)
    valid = torch.zeros(1, 1, SLOTS, dtype=torch.bool); valid[0, 0, :k] = True
    cond = torch.zeros(1, 1, SLOTS, dtype=torch.bool); cond[0, 0, 0] = True
    slots = torch.arange(SLOTS, dtype=torch.int64).reshape(1, 1, SLOTS)
    return CanonicalState(spatial, pointer, torch.zeros(1, 1, SLOTS, 1), frames, slots, cond, valid, (1,), k - 1)

cases = []
for split, counts in (("fit", FIT), ("development", DEV)):
    for i, k in enumerate(counts):
        video, total = f"{split[0]}{i:02d}vid", k + 12
        cases.append(dict(case_id=f"synthetic_{video}_obj1", video_id=video, object_id=1, first_prompt_frame=0,
                          switch_frame=k - 1, future_end_frame=total - 1, paired_split=split))
        rgb = work / "rgb" / video; rgb.mkdir(parents=True, exist_ok=True)
        for frame in range(total):
            (rgb / f"{frame:05d}.jpg").write_bytes(b"differential fixture filename map only")
        x = torch.randn(1, 1, SLOTS, 64, 64, 64, generator=g); p = torch.randn(1, 1, SLOTS, 256, generator=g)
        y = torch.einsum("...chw,dc->...dhw", x, mix_s) + shift_s.view(64, 1, 1) + .05 * torch.randn(x.shape, generator=g)
        q = p @ mix_p.T + shift_p + .05 * torch.randn(p.shape, generator=g)
        for tensor in (x, p, y, q):
            tensor[0, 0, k:] = 0
        write_case_cache(work / split / f"lvos_{video}_obj1_switch{k - 1}.pt",
            source_canonical=state(x.bfloat16(), p, k), target_canonical=state(y.bfloat16(), q, k),
            metadata=dict(source_model_id="sam2.1-small", target_model_id="sam2.1-base-plus", video_id=video,
                          object_id=1, switch_frame=k - 1, num_frames=total, cache_mode="state_only",
                          active_memory_only=True, num_maskmem=7, max_obj_ptrs_in_encoder=16, synthetic=True))

splits = dict(source="differential", fit="differential", development="differential")
raw.frozen_cases = lambda: ([dict(c) for c in cases], splits)
raw.build_index(work / "fit", work / "development", work / "rgb", work / "index",
                operator_completed="differential fixture", stable_seconds=0, max_wall_seconds=1800)

class ReferenceAffine(_LearnedStateTranslator):
    """Position-shared Wx+b on the reference Transformer's tensor API.

    The DDP trainer reads parameters by the spatial.* / pointer.* prefixes."""
    def __init__(self):
        super().__init__(SAM21_MEMORY_SPEC, SAM21_MEMORY_SPEC)
        self.spatial = torch.nn.Linear(64, 64)
        self.pointer = torch.nn.Linear(256, 256)
        with torch.no_grad():
            for layer in (self.spatial, self.pointer):
                layer.weight.copy_(torch.eye(layer.out_features, layer.in_features)); layer.bias.zero_()
    def _feature_head(self, tensor):
        return self.spatial(tensor.to(device=self.spatial.weight.device, dtype=self.spatial.weight.dtype))
    def _pointer_head(self, tensor):
        return self.pointer(tensor.to(device=self.pointer.weight.device, dtype=self.pointer.weight.dtype))
    def to_payload(self):
        return {"state_dict": {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}}

lock = dict(digest="differential-affine", approved_by=None)
ddp.model_lock = lambda approved_by=None: dict(lock)
ddp.validate_model_lock = lambda value, require_approval=True: ReferenceAffine()
os.environ.update(RANK="0", LOCAL_RANK="0", WORLD_SIZE="1", LVOS_CPU_STORE=str(work / "gloo_store"))
ddp.train(work / "index/raw_index.json", work / "reference_run", global_batch=64, mode="differential",
          device="cpu", max_wall_seconds=7200, config=LVOSConfig(workers=0, accumulation=4, max_epochs=epochs))
'''

TEST10_SCRIPT = r'''
import sys
from pathlib import Path
import torch
torch.set_num_threads(1)
root, index, output, epochs = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
sys.path.insert(0, str(root))
import comparable_training as cmp
fixed = cmp.recipe
cmp.recipe = lambda: dict(fixed(), epochs=epochs)
args = cmp.parser().parse_args(["train", "--index", index, "--output-dir", output, "--device", "cpu", "--cache-gib", "0"])
report = cmp.train(args)
if report["status"] != "complete": raise SystemExit("test10 training did not complete")
'''


def run(command, env, log):
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    log.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise SystemExit(f"failed ({result.returncode}), see {log}\n{result.stderr[-4000:]}")


def max_difference(left, right):
    return max(float((left[k].float() - right[k].float()).abs().max()) for k in left)


def compare(work, epochs):
    import torch
    reference = json.loads((work / "reference_run/metrics/history.json").read_text())
    report = json.loads((work / "test10_run/train_report.json").read_text())
    ours = report["history"]
    if [r["epoch"] for r in reference] != list(range(1, epochs + 1)) or [r["epoch"] for r in ours] != list(range(1, epochs + 1)):
        raise SystemExit("epoch coverage differs")
    rows = []
    for ref, mine in zip(reference, ours):
        row = dict(epoch=ref["epoch"],
                   optimizer_steps=[ref["optimizer_step"], mine["optimizer_steps"]],
                   fit_records=[ref["records"], mine["training_records"]],
                   dev_records=[ref["dev"]["records"], mine["validation"]["records"]],
                   dev_loss=[ref["dev"]["loss"], mine["validation"]["loss"]],
                   dev_loss_abs_difference=abs(ref["dev"]["loss"] - mine["validation"]["loss"]))
        exported = work / f"reference_run/translator/epoch-{ref['epoch']:05d}.pth"
        candidate = work / f"test10_run/epochs/affine-epoch-{ref['epoch']:05d}.pt"
        if candidate.is_file():
            theirs = {k.replace("spatial.", "feature.", 1): v
                      for k, v in torch.load(exported, map_location="cpu", weights_only=True)["state_dict"].items()}
            row["weight_max_abs_difference"] = max_difference(
                theirs, torch.load(candidate, map_location="cpu", weights_only=True)["state_dict"])
        rows.append(row)
    best = report["models"]["affine"]
    exported = work / f"reference_run/translator/epoch-{best['epoch']:05d}.pth"
    theirs = {k.replace("spatial.", "feature.", 1): v
              for k, v in torch.load(exported, map_location="cpu", weights_only=True)["state_dict"].items()}
    selected = torch.load(work / "test10_run" / best["checkpoint"], map_location="cpu", weights_only=True)["state_dict"]
    scales = json.loads((work / "index/raw_index.json").read_text())["normalization"]["scales"]
    return dict(epochs=epochs, rows=rows,
                exact_counts=all(r["optimizer_steps"][0] == r["optimizer_steps"][1] and r["fit_records"][0] == r["fit_records"][1]
                                 and r["dev_records"][0] == r["dev_records"][1] for r in rows),
                max_dev_loss_abs_difference=max(r["dev_loss_abs_difference"] for r in rows),
                max_dev_loss_relative_difference=max(r["dev_loss_abs_difference"] / abs(r["dev_loss"][0]) for r in rows),
                max_weight_abs_difference=max((r["weight_max_abs_difference"] for r in rows if "weight_max_abs_difference" in r),
                                              default=None),
                selected_epoch=best["epoch"], selected_weight_max_abs_difference=max_difference(theirs, selected),
                index_scales=scales, test10_scales=report["normalization"]["scales"],
                reference_selected_epoch=min(reference, key=lambda r: (r["dev"]["loss"], -r["epoch"]))["epoch"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-repo", type=Path, required=True, help="git checkout of vos-memory-translator-nonlinear")
    p.add_argument("--commit", help="reference commit; default: affine_comparable_v1.json reference_revision")
    p.add_argument("--translator-repo", type=Path, help="translator source for test10 (Affine model revision)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--work-dir", type=Path)
    p.add_argument("--output", type=Path, help="write the comparison JSON here")
    p.add_argument("--weight-tolerance", type=float, default=0.)
    args = p.parse_args()
    commit = args.commit or json.loads((ROOT / "configs/affine_comparable_v1.json").read_text())["reference_revision"]
    work = (args.work_dir or Path(tempfile.mkdtemp(prefix="affine-reference-diff-"))).resolve()
    work.mkdir(parents=True, exist_ok=True)
    reference_src = work / "reference_source"
    if not reference_src.exists():
        reference_src.mkdir()
        archive = subprocess.run(["git", "-C", str(args.reference_repo), "archive", commit, "src"], capture_output=True, check=True)
        subprocess.run(["tar", "-x", "-C", str(reference_src)], input=archive.stdout, check=True)
    base = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONHASHSEED="0")
    base.pop("PYTHONPATH", None)
    if not (work / "reference_run/STATUS.json").exists():
        run([sys.executable, "-c", REFERENCE_SCRIPT, str(work), str(args.epochs)],
            dict(base, PYTHONPATH=str(reference_src / "src")), work / "reference.log")
    env = dict(base)
    if args.translator_repo:
        env["TEST10_TRANSLATOR_REPO"] = str(args.translator_repo.resolve())
    if not (work / "test10_run/train_report.json").exists() or json.loads((work / "test10_run/train_report.json").read_text()).get("status") != "complete":
        run([sys.executable, "-c", TEST10_SCRIPT, str(ROOT), str(work / "index/raw_index.json"), str(work / "test10_run"), str(args.epochs)],
            env, work / "test10.log")
    result = dict(reference_commit=commit, work_dir=str(work), **compare(work, args.epochs))
    text = json.dumps(result, indent=2)
    (args.output or work / "comparison.json").write_text(text + "\n", encoding="utf-8")
    print(text)
    weights = result["max_weight_abs_difference"]
    ok = (result["exact_counts"] and result["selected_epoch"] == result["reference_selected_epoch"] and
          (weights if weights is not None else result["selected_weight_max_abs_difference"]) <= args.weight_tolerance)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
