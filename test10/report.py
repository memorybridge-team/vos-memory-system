"""Video-averaged, paired reports on complete, gated cases only."""
from collections import defaultdict
from statistics import mean, median
import random

from core import METHODS, GATE, DATASETS, SWITCH_WINDOW, write

METRICS = {
    "switch_window": ("switch_window", "J_and_F"),
    "J&F": ("post_switch", "J_and_F"), "J": ("post_switch", "J"), "F": ("post_switch", "F"),
    "+1": ("checkpoints", "+1", "J_and_F"), "+5": ("checkpoints", "+5", "J_and_F"),
    "+20": ("checkpoints", "+20", "J_and_F"), "remaining": ("remaining", "J_and_F"),
    "visible_J&F": ("gt_visible", "J_and_F"), "absent_FP": ("false_positives", "rate"),
    "reappearance_failure": ("reappearance", "no_recovery_rate"),
    "reappearance_delay": ("reappearance", "mean_recovery_length")}


def value(row, path):
    node = row.get("scores", {})
    for p in path:
        if not isinstance(node, dict): return None
        node = node.get(p)
    return node


def per_video(rows, path):
    buckets = defaultdict(list)
    for row in rows:
        v = value(row, path)
        if v is not None: buckets[(row["dataset"], row["video_id"])].append(v)
    return {k: mean(v) for k, v in buckets.items()}


def bootstrap(values, seed=7, reps=2000):
    if not values: return None
    rng = random.Random(seed)
    draws = sorted(mean(rng.choices(values, k=len(values))) for _ in range(reps))
    return dict(mean=mean(values), ci95=[draws[int(.025*reps)], draws[int(.975*reps)-1]], videos=len(values))


def paired(rows, a, b, seed=7, metric="J&F"):
    entries = defaultdict(dict)
    for r in rows: entries[r["case_id"]][r["method"]] = r
    diffs = defaultdict(list)
    for case in entries.values():
        if a not in case or b not in case: continue
        av, bv = value(case[a], METRICS[metric]), value(case[b], METRICS[metric])
        if av is not None and bv is not None:
            diffs[(case[a]["dataset"], case[a]["video_id"])].append(av-bv)
    return bootstrap([mean(v) for v in diffs.values()], seed)


def build_report(selection, store, seed=7, scheduled=None):
    methods = tuple(selection.get("methods", METHODS))
    datasets_in_run = (tuple(dict.fromkeys(c["dataset"] for c in selection["cases"]))
                       if selection.get("design") == "heldout.v1" else DATASETS)
    rows, missing, gates, costs = [], [], [], []
    for case in selection["cases"]:
        found = {m: store.load(case, m) for m in (*methods, GATE)}
        absent = [m for m, row in found.items() if row is None]
        gate = found[GATE]
        if absent or not gate or not gate.get("gate_passed"):
            missing.append(dict(case_id=case["case_id"], cohort=case["cohort"], missing=absent,
                                scheduled=scheduled is not None and case["case_id"] in scheduled))
        else:
            gates.append(gate); rows.extend(found[m] for m in methods)
        cost = store.load(case, "costs")
        if cost: costs.append(cost)
    windows = [value(r, ("switch_window", "k")) for r in rows]
    if len(set(windows) - {None}) > 1 or (None in windows and set(windows) != {None}):
        raise ValueError("mixed or partial switch-window scores; run --stage rescore to completion")
    window_k = windows[0] if windows else None
    report = dict(protocol="test10.v1", design=selection.get("design", "original"),
                  methods=methods, candidate_cases=len(selection["cases"]), complete_cases=len(gates),
                  incomplete=missing, gate_cases=len(gates), groups={}, timing={},
                  switch_window=dict(k=window_k, primary=window_k is not None,
                                     definition=None if window_k is None else
                                     f"J&F over annotated processed frames switch+1..switch+{window_k}; "
                                     "the switch frame itself is the last Small output and is not scored",
                                     note=None if window_k is not None else
                                     f"rows predate the switch-window metric; run --stage rescore (k={SWITCH_WINDOW})"),
                  sampling_note="offsets use processed frames; LVOS raw stride=5; VOST raw stride=6; independent single-object runs",
                  davis_note="heldout DAVIS uses train split and is exploratory" if selection.get("design") == "heldout.v1" else
                             "prior selection use unknown: exploratory validation",
                  reused_rows=sum(r.get("origin") == "reused" for r in rows))
    groups = {"all": rows,
              "legacy_dev": [r for r in rows if r["cohort"] == "legacy_dev"],
              "additional": [r for r in rows if r["cohort"] == "additional"],
              "exclude_checkpoint_videos": [r for r in rows if not r["checkpoint_video"]]}
    if selection.get("design") == "heldout.v1":
        groups = {"all": rows,
                  "heldout_core": [r for r in rows if r["cohort"] == "heldout_core"],
                  "heldout_extension": [r for r in rows if r["cohort"] == "heldout_extension"]}
    comparisons = [("affine", "direct"), ("residual_mlp", "affine"), ("transformer", "affine"),
                   ("affine_spatial", "affine"), ("affine_pointer", "affine")]
    comparisons += [(a,b) for a in ("affine", "residual_mlp", "transformer")
                    for b in ("last_mask", "anchor_replay_4", "anchor_replay_8", "anchor_replay_16")]
    comparisons = [(a, b) for a, b in comparisons if a in methods and b in methods]
    for group, group_rows in groups.items():
        datasets = {}
        for dataset in datasets_in_run:
            sub = [r for r in group_rows if r["dataset"] == dataset]
            table = {}
            for method in methods:
                method_rows = [r for r in sub if r["method"] == method]
                table[method] = {name: bootstrap(list(per_video(method_rows, path).values()), seed)
                                 for name, path in METRICS.items()}
                table[method]["cases"] = len(method_rows)
                table[method]["objects"] = len({(r["video_id"], r["object_id"]) for r in method_rows})
                for event, field in (("absent", ("false_positives", "gt_absent_frames")),
                                     ("reappearance", ("reappearance", "count"))):
                    table[method][event + "_videos"] = len({r["video_id"] for r in method_rows if (value(r, field) or 0) > 0})
            datasets[dataset] = dict(methods=table, paired={f"{a}-minus-{b}": paired(sub,a,b,seed) for a,b in comparisons},
                                     paired_switch_window={f"{a}-minus-{b}": paired(sub,a,b,seed,"switch_window")
                                                           for a,b in comparisons})
        macro, macro_window = {}, {}
        for method in methods:
            for target, name in ((macro, "J&F"), (macro_window, "switch_window")):
                vals = [datasets[d]["methods"][method][name] for d in datasets_in_run]
                target[method] = mean(v["mean"] for v in vals) if all(vals) else None
        report["groups"][group] = dict(datasets=datasets, equal_dataset_mean_JF=macro,
                                       equal_dataset_mean_switch_window=macro_window)
    for method in methods:
        measured = [r["timings"][method] for r in costs if method in r["timings"]]
        report["timing"][method] = dict(cases=len(measured), fields={})
        for field in ("prefix_s", "export_s", "transfer_s", "translate_s", "inject_s", "replay_s", "handoff_s", "first_output_s", "peak_vram_bytes"):
            vals = [r[field] for r in measured if isinstance(r.get(field), (int,float))]
            report["timing"][method]["fields"][field] = median(vals) if vals else None
    write(store.root / "summary.json", report)
    k = window_k or SWITCH_WINDOW
    def cell(metric):
        return "—" if metric is None else f"{100*metric['mean']:.2f} [{100*metric['ci95'][0]:.2f}, {100*metric['ci95'][1]:.2f}]"
    lines = ["# test10 결과", "", f"완료 {len(gates)} / 후보 {len(selection['cases'])} cases. 재사용 row {report['reused_rows']}.",
             "", "모든 방법 및 self-injection gate를 완료한 공통 case만 집계합니다. 부분 실행은 summary.json의 incomplete에 보존합니다.",
             report["davis_note"], "",
             f"주 지표는 전환 직후 J&F입니다: 처리 프레임 switch+1..switch+{k} (Base+의 첫 {k}개 출력) 평균. "
             "switch 프레임은 Small의 마지막 출력이라 채점하지 않습니다. +1은 Base+ 첫 출력 한 프레임, "
             "Suffix J&F는 switch+1..end 전체 평균입니다. LVOS/VOST offset은 처리 프레임(원본 5/6 프레임 간격)입니다.", ""]
    if window_k is None:
        lines += ["전환 직후 J&F가 없는 row입니다. `--stage rescore`로 저장된 예측을 다시 채점하세요.", ""]
    lines += [f"| Dataset | Method | Cases | Videos | 전환 직후 J&F +1..+{k} (95% CI) | +1 J&F | Suffix J&F (95% CI) |",
              "|---|---|---:|---:|---|---:|---|"]
    for d in datasets_in_run:
        for m in methods:
            row = report["groups"]["all"]["datasets"][d]["methods"][m]; metric = row["J&F"]
            first = "—" if row["+1"] is None else f"{100*row['+1']['mean']:.2f}"
            lines.append(f"| {d} | {m} | {row['cases']} | {metric['videos'] if metric else 0} | "
                         f"{cell(row['switch_window'])} | {first} | {cell(metric)} |")
    lines += ["", "세부 cohort, component ablation, visible/absent/reappearance, paired CI(`paired_switch_window`, `paired`) 및 비용은 summary.json을 참조하세요."]
    (store.root / "summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return report
