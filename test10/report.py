"""Video-averaged, paired reports on complete, gated cases only."""
from collections import defaultdict
from statistics import mean, median
import random

from core import METHODS, GATE, DATASETS, write

METRICS = {
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


def paired(rows, a, b, seed=7):
    entries = defaultdict(dict)
    for r in rows: entries[r["case_id"]][r["method"]] = r
    diffs = defaultdict(list)
    for case in entries.values():
        if a not in case or b not in case: continue
        av, bv = value(case[a], METRICS["J&F"]), value(case[b], METRICS["J&F"])
        if av is not None and bv is not None:
            diffs[(case[a]["dataset"], case[a]["video_id"])].append(av-bv)
    return bootstrap([mean(v) for v in diffs.values()], seed)


def build_report(selection, store, seed=7, scheduled=None):
    rows, missing, gates, costs = [], [], [], []
    for case in selection["cases"]:
        found = {m: store.load(case, m) for m in (*METHODS, GATE)}
        absent = [m for m, row in found.items() if row is None]
        gate = found[GATE]
        if absent or not gate or not gate.get("gate_passed"):
            missing.append(dict(case_id=case["case_id"], cohort=case["cohort"], missing=absent,
                                scheduled=scheduled is not None and case["case_id"] in scheduled))
        else:
            gates.append(gate); rows.extend(found[m] for m in METHODS)
        cost = store.load(case, "costs")
        if cost: costs.append(cost)
    report = dict(protocol="test10.v1", candidate_cases=len(selection["cases"]), complete_cases=len(gates),
                  incomplete=missing, gate_cases=len(gates), groups={}, timing={},
                  sampling_note="offsets use processed frames; LVOS raw stride=5; independent single-object runs",
                  davis_note="prior selection use unknown: exploratory validation",
                  reused_rows=sum(r.get("origin") == "reused" for r in rows))
    groups = {"all": rows,
              "legacy_dev": [r for r in rows if r["cohort"] == "legacy_dev"],
              "additional": [r for r in rows if r["cohort"] == "additional"],
              "exclude_checkpoint_videos": [r for r in rows if not r["checkpoint_video"]]}
    comparisons = [("affine", "direct"), ("residual_mlp", "affine"), ("transformer", "affine"),
                   ("affine_spatial", "affine"), ("affine_pointer", "affine")]
    comparisons += [(a,b) for a in ("affine", "residual_mlp", "transformer")
                    for b in ("last_mask", "anchor_replay_4", "anchor_replay_8", "anchor_replay_16")]
    for group, group_rows in groups.items():
        datasets = {}
        for dataset in DATASETS:
            sub = [r for r in group_rows if r["dataset"] == dataset]
            table = {}
            for method in METHODS:
                method_rows = [r for r in sub if r["method"] == method]
                table[method] = {name: bootstrap(list(per_video(method_rows, path).values()), seed)
                                 for name, path in METRICS.items()}
                table[method]["cases"] = len(method_rows)
                table[method]["objects"] = len({(r["video_id"], r["object_id"]) for r in method_rows})
                for event, field in (("absent", ("false_positives", "gt_absent_frames")),
                                     ("reappearance", ("reappearance", "count"))):
                    table[method][event + "_videos"] = len({r["video_id"] for r in method_rows if (value(r, field) or 0) > 0})
            datasets[dataset] = dict(methods=table, paired={f"{a}-minus-{b}": paired(sub,a,b,seed) for a,b in comparisons})
        macro = {}
        for method in METHODS:
            vals = [datasets[d]["methods"][method]["J&F"] for d in DATASETS]
            macro[method] = mean(v["mean"] for v in vals) if all(vals) else None
        report["groups"][group] = dict(datasets=datasets, equal_dataset_mean_JF=macro)
    for method in METHODS:
        measured = [r["timings"][method] for r in costs if method in r["timings"]]
        report["timing"][method] = dict(cases=len(measured), fields={})
        for field in ("prefix_s", "export_s", "transfer_s", "translate_s", "inject_s", "replay_s", "handoff_s", "first_output_s", "peak_vram_bytes"):
            vals = [r[field] for r in measured if isinstance(r.get(field), (int,float))]
            report["timing"][method]["fields"][field] = median(vals) if vals else None
    write(store.root / "summary.json", report)
    lines = ["# test10 결과", "", f"완료 {len(gates)} / 후보 {len(selection['cases'])} cases. 재사용 row {report['reused_rows']}.",
             "", "모든 방법 및 self-injection gate를 완료한 공통 case만 집계합니다. 부분 실행은 summary.json의 incomplete에 보존합니다.",
             "DAVIS는 과거 모델 선택 사용 여부가 확인되지 않아 탐색적 평가로 표시합니다.", "",
             "| Dataset | Method | Cases | Videos | J&F ×100 (95% CI) |", "|---|---|---:|---:|---|"]
    for d in DATASETS:
        for m in METHODS:
            row = report["groups"]["all"]["datasets"][d]["methods"][m]; metric = row["J&F"]
            cell = "—" if metric is None else f"{100*metric['mean']:.2f} [{100*metric['ci95'][0]:.2f}, {100*metric['ci95'][1]:.2f}]"
            lines.append(f"| {d} | {m} | {row['cases']} | {metric['videos'] if metric else 0} | {cell} |")
    lines += ["", "세부 cohort, component ablation, visible/absent/reappearance, paired CI 및 비용은 summary.json을 참조하세요."]
    (store.root / "summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return report
