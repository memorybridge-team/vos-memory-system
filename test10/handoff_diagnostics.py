#!/usr/bin/env python3
"""Offline post-switch diagnostics from saved prediction caches (no model load, no GPU).

1. early_*: GT J/F/J&F over the first K post-switch frames (default K=10),
   with an individual score/status for every offset +1..+K. This horizon is
   independent of the SAM 2.1 recent-memory lifetime.
   Scores reuse the test10 metric path (``mvp_scoring.score``) on a truncated suffix.
2. agree_*: per-frame mask IoU against the Base+-native continuation (no GT), over the
   same early window and over the whole suffix. Both-empty frames score 1.

``switch_iou`` (Small vs Base+ prefix mask at the switch frame) separates cases where
the handed-over Small state already disagreed with Base+ before any translation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

from core import WORKSPACE, METHODS, GATE, SWITCH_WINDOW, Artifacts, read, write
from report import bootstrap

PLACEHOLDER = "${WORKSPACE_ROOT}"
REFERENCE = "base_native"
FIELDS = ("early_JF", "early_J", "early_F", "agree_early", "agree_full")
PAIRS = (("residual_mlp", "affine"), ("transformer", "affine"),
         ("affine", "last_mask"), ("affine", "small_only"))


def restore(value, workspace=WORKSPACE):
    """Undo the public-copy path placeholder; Artifacts.load then verifies the keys."""
    escaped = json.dumps(str(workspace), ensure_ascii=False)[1:-1]
    return json.loads(json.dumps(value, ensure_ascii=False).replace(PLACEHOLDER, escaped))


def iou(a, b):
    import numpy as np
    a, b = np.asarray(a).astype(bool), np.asarray(b).astype(bool)
    if a.shape != b.shape:
        raise ValueError(f"mask shapes differ: {a.shape} vs {b.shape}")
    union = np.count_nonzero(a | b)
    return 1. if union == 0 else float(np.count_nonzero(a & b) / union)


def _bits(order):
    def unpack(packed):
        import numpy as np
        shape, data = packed
        shape = tuple(int(x) for x in shape)
        bits = np.unpackbits(np.frombuffer(data, np.uint8), count=int(np.prod(shape)), bitorder=order)
        return bits.reshape(shape).astype(bool)
    return unpack


def mask_decoder(module=None):
    """Return a decoder proven to invert ``mvp_scoring.pack`` on random probe masks."""
    import numpy as np
    if module is None:
        from manifest import imports
        imports()
        import mvp_scoring as module
    rng = np.random.default_rng(0)
    probes = [rng.random((1, 7, 13)) > .5, rng.random((5, 3)) > .5, np.zeros((2, 9), bool)]
    packed = [module.pack(p) for p in probes]
    own = [getattr(module, n) for n in sorted(dir(module))
           if ("unpack" in n.lower() or "decode" in n.lower()) and callable(getattr(module, n))]
    for decode in (*own, lambda x: x, _bits("big"), _bits("little")):
        try:
            if all(np.array_equal(np.asarray(decode(x)).astype(bool), p) for x, p in zip(packed, probes)):
                return decode
        except Exception:
            continue
    raise RuntimeError("no decoder reproduces mvp_scoring.pack; add one to handoff_diagnostics.mask_decoder")


def gt_scorer():
    """Same scorer as the main pipeline (runtime.score_positions) on the truncated suffix."""
    from runtime import score_positions
    return score_positions


def blob_loader(store):
    from runtime import load_blob  # sha/key-verified cache reader; imports torch lazily
    return lambda case, name: load_blob(store, case, name)


def complete_cases(selection, store, methods):
    """Same inclusion rule as report.build_report: every method plus a passed gate."""
    cases = []
    for case in selection["cases"]:
        gate = store.load(case, GATE)
        if gate and gate.get("gate_passed") and all(store.load(case, m) for m in methods):
            cases.append(case)
    return cases


def case_rows(case, methods, load, decode, score, early_frames):
    source, native = load(case, "source_prefix"), load(case, "base_prefix")
    if source is None or native is None:
        raise FileNotFoundError(f"missing prefix cache for {case['case_id']} (public copies omit .pt blobs)")
    masks = {"small_only": source["masks"], REFERENCE: native["masks"]}
    for method in methods:
        if method not in masks:
            blob = load(case, method + "_predictions")
            if blob is None:
                raise FileNotFoundError(f"missing prediction cache: {case['case_id']} {method}")
            masks[method] = blob["masks"]
    suffix = list(range(case["switch"] + 1, case["end"] + 1))
    early = suffix[:early_frames]
    reference = {p: decode(masks[REFERENCE][p]) for p in suffix}
    switch_iou = iou(source["last_mask"], native["last_mask"])
    rows = []
    for method in methods:
        predicted = masks[method]
        if set(predicted) != set(suffix):
            raise ValueError(f"suffix frame mismatch: {case['case_id']} {method}")
        scores = score(case, method, {p: predicted[p] for p in early})["post_switch"]
        row = dict(case_id=case["case_id"], dataset=case["dataset"], video_id=case["video_id"],
                   method=method, switch_iou=switch_iou, early_frames=len(early),
                   early_annotated=scores["frames"], early_J=scores["J"], early_F=scores["F"],
                   early_JF=scores["J_and_F"], agree_early=None, agree_full=None)
        row["switch_frames"] = {}
        for offset in range(1, early_frames + 1):
            pos = case["switch"] + offset
            frame = dict(position=pos, frame_stem=None, status="outside_suffix", frames=0,
                         J=None, F=None, J_and_F=None, agree_base=None)
            if pos <= case["end"]:
                metric = score(case, method, {pos: predicted[pos]})["post_switch"]
                frame.update(metric, frame_stem=case["frame_stems"][pos],
                             status="scored" if metric["frames"] else "missing_annotation",
                             agree_base=iou(decode(predicted[pos]), reference[pos]))
            row["switch_frames"][f"+{offset}"] = frame
        if method != REFERENCE:
            ious = [iou(decode(predicted[p]), reference[p]) for p in suffix]
            row.update(agree_early=mean(ious[:len(early)]), agree_full=mean(ious))
        rows.append(row)
    return rows


def per_video(rows, field):
    buckets = defaultdict(list)
    for row in rows:
        if row.get(field) is not None:
            buckets[(row["dataset"], row["video_id"])].append(row[field])
    return {k: mean(v) for k, v in buckets.items()}


def paired(rows, a, b, field, seed):
    by_case = defaultdict(dict)
    for row in rows:
        by_case[row["case_id"]][row["method"]] = row
    diffs = defaultdict(list)
    for entry in by_case.values():
        if a in entry and b in entry and entry[a][field] is not None and entry[b][field] is not None:
            diffs[(entry[a]["dataset"], entry[a]["video_id"])].append(entry[a][field] - entry[b][field])
    return bootstrap([mean(v) for v in diffs.values()], seed)


def summarize(rows, methods, seed, threshold):
    groups = {"all": rows, f"switch_iou>={threshold}": [r for r in rows if r["switch_iou"] >= threshold]}
    datasets = tuple(dict.fromkeys(r["dataset"] for r in rows))
    out = {}
    for group, sub in groups.items():
        out[group] = {}
        for dataset in datasets:
            ds = [r for r in sub if r["dataset"] == dataset]
            out[group][dataset] = dict(
                cases=len({r["case_id"] for r in ds}), videos=len({r["video_id"] for r in ds}),
                methods={m: {f: bootstrap(list(per_video([r for r in ds if r["method"] == m], f).values()), seed)
                             for f in FIELDS} for m in methods},
                paired={f"{a}-minus-{b}": {f: paired(ds, a, b, f, seed) for f in ("early_JF", "agree_early")}
                        for a, b in PAIRS if a in methods and b in methods})
    return out


def markdown(report):
    k = report["early_frames"]

    def cell(metric):
        return "—" if metric is None else f"{100 * metric['mean']:.2f} [{100 * metric['ci95'][0]:.2f}, {100 * metric['ci95'][1]:.2f}]"

    lines = ["# test10 전환 직후·Base+ 일치도 진단", "",
             f"완료 {report['cases']} cases. 영상 단위 평균 ×100, 대괄호는 영상 bootstrap 95% CI.",
             f"- Early J&F: 전환 후 +1..+{k} 프레임의 GT J&F (test10 metric 경로와 동일).",
             f"- IoU vs Base+: GT 없이 Base+-native 예측 mask와의 프레임별 IoU 평균 (+1..+{k}, 전체 suffix).",
             "- switch_iou: 전환 프레임에서 Small prefix mask와 Base+ prefix mask의 IoU.", ""]
    for group, datasets in report["groups"].items():
        lines += [f"## {group}", "", f"| Dataset | Cases | Method | Early J&F +1..+{k} | IoU vs Base+ +1..+{k} | IoU vs Base+ suffix |",
                  "|---|---:|---|---|---|---|"]
        for dataset, entry in datasets.items():
            for method, metrics in entry["methods"].items():
                lines.append(f"| {dataset} | {entry['cases']} | {method} | {cell(metrics['early_JF'])} | "
                             f"{cell(metrics['agree_early'])} | {cell(metrics['agree_full'])} |")
        lines += ["", "| Dataset | Paired | Early J&F | IoU vs Base+ early |", "|---|---|---|---|"]
        for dataset, entry in datasets.items():
            for name, metrics in entry["paired"].items():
                lines.append(f"| {dataset} | {name} | {cell(metrics['early_JF'])} | {cell(metrics['agree_early'])} |")
        lines.append("")
    return "\n".join(lines)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True, help="original run directory containing the .pt caches")
    p.add_argument("--early-frames", type=int, default=SWITCH_WINDOW)
    p.add_argument("--switch-agree", type=float, default=.9, help="switch_iou threshold for the subgroup table")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--workspace-root", type=Path, default=WORKSPACE,
                   help=f"value substituted for {PLACEHOLDER} in public-copy manifests")
    return p


def main(args, *, load=None, decode=None, score=None):
    if args.early_frames < 1:
        raise ValueError("--early-frames must be positive")
    root = getattr(args, "workspace_root", WORKSPACE)
    selection = restore(read(args.run_dir / "selection.json"), root)
    store = Artifacts(args.run_dir, restore(read(args.run_dir / "provenance.json"), root))
    methods = tuple(selection.get("methods", METHODS))
    cases = complete_cases(selection, store, methods)
    if not cases:
        raise ValueError("no complete gated cases found. Artifact keys include provenance paths: run this on the "
                         "original run directory, or pass the original --workspace-root for a copied run")
    load = load or blob_loader(store)
    decode = decode or mask_decoder()
    score = score or gt_scorer()
    rows = []
    for i, case in enumerate(cases, 1):
        rows.extend(case_rows(case, methods, load, decode, score, args.early_frames))
        print(f"[{i}/{len(cases)}] {case['case_id']}", flush=True)
    report = dict(protocol="test10.handoff_diagnostics.v1", early_frames=args.early_frames,
                  reference=REFERENCE, switch_agree_threshold=args.switch_agree, cases=len(cases),
                  groups=summarize(rows, methods, args.seed, args.switch_agree), rows=rows)
    write(args.run_dir / "handoff_diagnostics.json", report)
    (args.run_dir / "handoff_diagnostics.md").write_text(markdown(report) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    main(parser().parse_args())
