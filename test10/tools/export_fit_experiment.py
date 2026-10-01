#!/usr/bin/env python3
"""CPU-only, audited export of a completed fit-pair training/evaluation run.

Does not change frozen inputs, schedules, scores, or model artifacts. The output
contains no RGB/annotation/state tensors and can be shared separately from them.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path
from statistics import mean, median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import Artifacts, GATE, ROOT, WORKSPACE, REPO, read, sha, verify_snapshot, write
from report import bootstrap
from run import restore_selection_paths

DATASETS = ("MOSEv2", "LVOSv2", "DAVIS2017", "VOST")
MODELS = ("affine", "residual_mlp", "transformer")


def portable(value):
    replacements = ((str(ROOT), "${TEST10_ROOT}"),
                    (str(REPO), "${TRANSLATOR_ROOT}"),
                    ("/mnt/c/Users/Home/runpod-state-pairs", "${STATE_PAIR_ROOT}"),
                    (str(WORKSPACE), "${WORKSPACE_ROOT}"))
    if isinstance(value, str):
        for old, new in replacements:
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [portable(v) for v in value]
    if isinstance(value, dict):
        return {portable(k): portable(v) for k, v in value.items()}
    return value


def csv_file(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_training(histories, training, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for m, history in histories.items():
        for ax, field in zip(axes, ("train", "validation")):
            values = [r["train_loss"] if field == "train" else r["validation"]["loss"] for r in history]
            ax.plot(range(1, 9), values, marker="o", label=m)
        epoch = training["models"][m]["epoch"]
        axes[1].scatter([epoch], [history[epoch - 1]["validation"]["loss"]], s=90, facecolors="none", edgecolors="black")
    for ax, title in zip(axes, ("Training state loss", "Video-weighted validation state loss")):
        ax.set(title=title, xlabel="Epoch", ylabel="Spatial MSE + pointer MSE")
        ax.set_xticks(range(1, 9)); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def overview(summary, training, audit, costs):
    def metric(v):
        return "—" if v is None else f"{100*v['mean']:.2f} [{100*v['ci95'][0]:.2f}, {100*v['ci95'][1]:.2f}]"
    groups = summary["groups"]["all"]["datasets"]
    lines = ["# Fit 1,000 pair 학습 후 Small→Base+ 평가", "",
        "주 지표: 전환 후 처리 프레임 +1..+10 중 annotation이 있는 프레임의 J&F. "
        "영상당 객체 하나·switch 하나이며 모든 8개 방법이 완료된 공통 영상만 사용한다. "
        "전체 suffix는 보조 지표다. 점수·CI는 point 단위(0–100)다.", "",
        "## 실제 규모와 split", "",
        "| Dataset | 학습 pair / 영상 | Checkpoint 선택 pair / 영상 | 최종 평가 영상 | 이전 평가에 사용된 영상 |",
        "|---|---:|---:|---:|---:|"]
    for d in DATASETS:
        parts = [audit["training_counts"].get(f"{d}:{split}") for split in ("train", "validation")]
        cells = ["—" if v is None else f"{v['pairs']} / {v['videos']}" for v in parts]
        lines.append(f"| {d} | {cells[0]} | {cells[1]} | {audit['evaluation_counts'][d]} | {audit['previously_evaluated_counts'].get(d,0)} |")
    lines += ["", "학습 1,000 pair(750영상), validation 200 pair(175영상)는 기존 fit pool만 사용했다. "
              f"유효 memory record는 학습 {sum(v['valid_records'] for k,v in audit['training_counts'].items() if k.endswith(':train')):,}개, "
              f"validation {sum(v['valid_records'] for k,v in audit['training_counts'].items() if k.endswith(':validation')):,}개다. "
              "MOSE 영상당 1 pair, LVOS 영상당 최대 2 pair다. 학습·validation·최종 평가는 영상 단위로 분리했다. "
              "MOSE train/development, LVOS valid, DAVIS 2017 train, VOST val을 사용한다. "
              "DAVIS train은 탐색적 평가다. 이전 실험에서 평가한 영상도 포함하므로 전체를 untouched test라고 부르지 않는다. "
              "영상별 이전 평가 이력과 길이 구간은 evaluation_cases.csv에 기록했다.", "",
              "## 학습과 checkpoint", "",
              "동일 seed 7, batch_records=4, 8 epoch, AdamW learning rate 0.001. "
              "Spatial MSE + pointer MSE를 학습하며 validation loss는 영상별 균등 평균이다. "
              "세 모델을 새로 초기화했고 과거 translator checkpoint는 사용하지 않았다.", "",
              "| 모델 | 선택 epoch | Validation loss | 파라미터 | 마지막 epoch가 최적 |",
              "|---|---:|---:|---:|---|"]
    for m in MODELS:
        info = training["models"][m]
        lines.append(f"| {m} | {info['epoch']} | {info['validation']['loss']:.6f} | {info['parameters']:,} | {'예' if info['epoch']==8 else '아니오'} |")
    lines += ["", "MLP와 Transformer는 epoch 8이 최적이므로 수렴이 확인되지 않았다. "
              "State loss가 낮다는 것만으로 segmentation 성능 우위를 뜻하지 않는다. "
              "Affine은 bias가 있는 Wx+b(69,952 parameters)이며 최소제곱 해가 아니라 AdamW 학습이다.", "",
              "![Training curves](training_curves.svg)", "",
              "선택 가중치는 ../../training/run_fit1000_val200/{affine,residual_mlp,transformer}.pt에 보존하며 "
              "SHA-256은 training.json에 기록한다.", "",
              "## 전환 직후 J&F 및 paired 차이", "",
              "| Dataset | Small-only | Base+-native | Direct | Affine | MLP | Transformer | Last-mask | Replay-16 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for d in DATASETS:
        lines.append("| " + d + " | " + " | ".join(metric(groups[d]["methods"][m]["switch_window"]) for m in summary["methods"]) + " |")
    comparisons = ("affine-minus-direct", "residual_mlp-minus-affine", "transformer-minus-affine", "affine-minus-last_mask", "affine-minus-anchor_replay_16")
    lines += ["", "같은 영상에서의 방법 차이를 video-paired bootstrap(2,000 draws, seed 7)으로 계산했다. "
              "CI가 0을 포함하면 해당 데이터셋의 우위가 확인된 것으로 해석하지 않는다.", "",
              "| Dataset | Affine−Direct | MLP−Affine | Transformer−Affine | Affine−Last-mask | Affine−Replay-16 |",
              "|---|---|---|---|---|---|"]
    for d in DATASETS:
        lines.append("| " + d + " | " + " | ".join(metric(groups[d]["paired_switch_window"][c]) for c in comparisons) + " |")
    lines += ["", "+1..+10 각각의 J/F/J&F와 GT 누락 상태는 switch_frames.csv, "
              "첫 10프레임 평균 J/F/J&F는 first10_J_F_JF.json, "
              "전체 suffix·visible/absent/reappearance·사건 영상 수·CI는 summary.json에 제공한다. "
              "LVOS offset은 원본 5프레임, VOST는 6프레임 간격의 처리 프레임이다.", "",
              "## 전환 비용과 검증", "",
              "고정 smoke 12영상(데이터셋당 3영상)의 warm-model/cold-RGB 중앙값이다. "
              "모델 load는 제외하며 video 초기화·export·이동·변환·주입·replay는 costs.json에 분리해 기록한다. "
              "첫 출력 시간은 translator/restart/replay에서는 준비된 Small 상태부터이고 "
              "Base+-native/Full Replay에서는 전체 prefix 재처리부터다. "
              "VRAM은 PyTorch peak allocated memory이며 nvidia-smi 전체 사용량이 아니다.", "",
              "| 방법 | Handoff ms | Export + 첫 출력 ms | Replay ms | Peak allocated MiB |",
              "|---|---:|---:|---:|---:|"]
    for m in summary["methods"]:
        records = [c["timings"][m] for c in costs]
        def med(field, scale=1000):
            values = [r[field] for r in records if field in r]
            return "—" if not values else f"{median(values)*scale:.2f}"
        total = median(r.get("export_s", 0) + r["first_output_s"] for r in records) * 1000
        lines.append(f"| {m} | {med('handoff_s')} | {total:.2f} | {med('replay_s')} | {med('peak_vram_bytes',1/1024**2)} |")
    lines += ["", f"Self-injection {audit['gates']}/{audit['gates']} 통과: native full-logit hash 동일, max error 0, binary IoU 1. "
              "Direct/Affine/MLP/Transformer의 전환 준비 중 과거 backbone 호출 0, "
              "Replay-16은 원래 anchor + 최근 16프레임임을 모든 완료 case에서 확인했다.", "",
              f"본 실행의 실패 {len(audit['failed'])}건, 미완료 후보 {len(audit['incomplete'])}건. "
              f"준비 단계 smoke 실패 {audit['excluded_setup_failures']}건은 수정 전 provenance로 분리하고 "
              "setup_failures.json에 보존했으며 최종 점수에 포함하지 않았다. "
              "SAM2 _C 확장 누락으로 optional hole-fill 후처리를 생략한 환경이며 모든 방법에 동일하게 적용됐다.", "",
              f"세 모델 학습+validation 합계 {sum(training['models'][m]['seconds'] for m in MODELS)/60:.2f}분, "
              f"단일 GPU 추가 평가 계측 {audit['evaluation_seconds']/60:.2f}분. "
              "여기에 초기 audit와 CPU 최종 집계 시간이 별도로 있다. "
              "영상·방법·suffix를 줄여 사례 수를 늘리지 않았다. "
              "확정 영상·pair 목록, 실행 순서, 원본 파일 hash, checkpoint hash와 결과 row는 이 폴더에 함께 보존한다.", ""]
    reuse = audit.get("downloaded_pair_reuse")
    if reuse:
        lines += ["## 기존 다운로드 state pair 재사용 점검", "",
                  f"영상·객체·처리 switch가 일치한 후보 {reuse['candidates']}개 중 "
                  f"현재 native continuation과 수치적으로 동일한 것으로 검증된 pair는 {reuse['verified_reusable']}개다. "
                  "상태 계약 및 full-suffix logit hash 점검은 downloaded_pair_reuse.json에 보존한다. "
                  "불일치는 이번 실행의 exact gate와 호환되지 않는다는 뜻이며, 모델 방향이 틀렸다는 결론이나 "
                  "학습용 fit bank 자체가 잘못됐다는 결론은 아니다. 차이의 원인은 별도 검증하지 않았다.", ""]
    return "\n".join(lines)


def validate_training(training_dir, selection):
    training = read(training_dir / "train_report.json")
    assert training["status"] == "complete"
    assert training["seed"] == 7
    assert training["config"]["epochs"] == 8
    assert training["config"]["batch_records"] == 4
    pairs = training["collection"]["pairs"]
    assert len(pairs) == 1200
    splits = {}
    counts = {}
    for dataset in DATASETS[:2]:
        for split, expected_pairs, expected_videos in (
                ("train", 500, 500 if dataset == "MOSEv2" else 250),
                ("validation", 100, 100 if dataset == "MOSEv2" else 75)):
            rows = [r for r in pairs if (r["dataset"], r["split"]) == (dataset, split)]
            videos = Counter(r["video_id"] for r in rows)
            assert len(rows) == expected_pairs and len(videos) == expected_videos
            assert max(videos.values()) <= (1 if dataset == "MOSEv2" else 2)
            assert all("/fit/" in r["path"] for r in rows)
            splits[dataset, split] = set(videos)
            counts[f"{dataset}:{split}"] = dict(pairs=len(rows), videos=len(videos),
                valid_records=sum(r["valid_records"] for r in rows))
        assert not (splits[dataset, "train"] & splits[dataset, "validation"])
        evaluated = {c["video_id"] for c in selection["cases"] if c["dataset"] == dataset}
        assert not (evaluated & (splits[dataset, "train"] | splits[dataset, "validation"]))
    histories = {}
    for model in MODELS:
        info = training["models"][model]
        assert info["sha256"] == sha(training_dir / info["checkpoint"])
        history = read(training_dir / f"{model}.history.json")
        assert [r["epoch"] for r in history] == list(range(1, 9))
        best = min(history, key=lambda r: r["validation"]["loss"])
        assert best["epoch"] == info["epoch"]
        assert best["validation"] == info["validation"]
        histories[model] = history
    return training, counts, histories


def export(run_dir, output, training_dir, expected_per_dataset):
    assert not output.exists(), f"refusing to overwrite export: {output}"
    prov = read(run_dir / "provenance.json")
    verify_snapshot(prov["files"])
    selection_raw = read(run_dir / "selection.json")
    selection = restore_selection_paths(selection_raw)
    verify_snapshot(selection.get("source_hashes", {}))
    training, training_counts, histories = validate_training(training_dir, selection)
    methods = selection["methods"]
    assert methods == ["small_only", "base_native", "direct", "affine", "residual_mlp",
                       "transformer", "last_mask", "anchor_replay_16"]
    store = Artifacts(run_dir, prov)
    prior_videos = {}
    prior_sources = {}
    for name in ("run01", "heldout01"):
        previous = ROOT / "runs" / name
        if not (previous / "budget.json").exists():
            continue
        old = read(previous / "selection.json")
        completed = {a["case_id"] for a in read(previous / "budget.json")["attempts"] if a["status"] == "complete"}
        prior_videos[name] = {(c["dataset"], c["video_id"]) for c in old["cases"] if c["case_id"] in completed}
        prior_sources[name] = {f: sha(previous / f) for f in ("selection.json", "budget.json")}
    cases, metrics, gates, missing, cost_rows = [], [], [], [], []
    seen = set()
    for c in selection["cases"]:
        assert (c["dataset"], c["video_id"]) not in seen
        seen.add((c["dataset"], c["video_id"]))
        gate = store.load(c, GATE)
        rows = [store.load(c, m) for m in methods]
        absent = [m for m, r in zip(methods, rows) if r is None]
        if gate is None or absent:
            missing.append(dict(case_id=c["case_id"], missing=absent + ([GATE] if gate is None else [])))
            continue
        assert gate["gate_passed"] and gate["binary_iou"] == 1 and gate["max_logit_error"] == 0
        assert gate["runtime"]["past_backbone_frames"] == []
        gates.append(gate)
        suffix_count = c["end"] - c["switch"]
        assert len({r["scores"]["post_switch"]["frames"] for r in rows}) == 1
        assert len({r["scores"]["switch_window"]["frames"] for r in rows}) == 1
        for r in rows:
            scores = r["scores"]
            assert scores["switch_window"]["k"] == 10
            assert scores["switch_window"]["offsets"] == list(range(1, 11))
            assert set(scores["switch_frames"]) == {f"+{i}" for i in range(1, 11)}
            for i in range(1, 11):
                f = scores["switch_frames"][f"+{i}"]
                assert f["position"] == c["switch"] + i
                assert f["frame_stem"] == c["frame_stems"][c["switch"] + i]
                assert f["status"] in ("scored", "missing_annotation")
            annotated = [f for f in scores["switch_frames"].values() if f["status"] == "scored"]
            assert len(annotated) == scores["switch_window"]["frames"]
            for field in ("J", "F", "J_and_F"):
                expected = mean(f[field] for f in annotated) if annotated else None
                actual = scores["switch_window"][field]
                assert actual is None if expected is None else abs(actual - expected) < 1e-12
            if r["method"] in ("direct", *MODELS):
                assert r["runtime"]["past_backbone_frames"] == []
            if r["method"] in ("small_only", "base_native"):
                assert r["runtime"]["backbone_frames"] == list(range(c["first"], c["end"] + 1))
            if r["method"] == "last_mask":
                assert r["runtime"]["past_backbone_frames"] == [c["switch"]]
            if r["method"] == "anchor_replay_16":
                assert r["runtime"]["past_backbone_frames"] == [c["first"], *range(c["switch"] - 15, c["switch"] + 1)]
            assert scores["post_switch"]["frames"] <= suffix_count
            metrics.append(r)
        costs = store.load(c, "costs")
        if costs:
            assert set(costs["timings"]) == set(methods)
            cost_rows.append(costs)
        used = [name for name, videos in prior_videos.items() if (c["dataset"], c["video_id"]) in videos]
        cases.append(dict(c, completed=True, prior_evaluation_runs=";".join(used),
                          previously_evaluated=bool(used)))
    counts = Counter(c["dataset"] for c in cases)
    assert counts == Counter({d: expected_per_dataset for d in DATASETS}), counts
    assert len(cost_rows) == 12
    budget = read(run_dir / "budget.json")
    assert "active_since" not in budget, "evaluation is still running"
    failed = [a for a in budget["attempts"] if a["status"] != "complete"]
    assert not failed, failed
    complete_ids = {c["case_id"] for c in cases}
    completed_attempts = [a["case_id"] for a in budget["attempts"] if a["status"] == "complete"]
    assert len(completed_attempts) == len(set(completed_attempts))
    assert set(completed_attempts) == complete_ids
    summary = read(run_dir / "summary.json")
    assert summary["complete_cases"] == len(cases) and summary["gate_cases"] == len(gates)
    setup_failures = []
    for name in ("fit1_eval40", "fit1_eval40_optimized"):
        setup_path = run_dir.parent / name / "budget.json"
        if setup_path.exists():
            setup_failures.extend(dict(a, run=name, excluded_from_final_scores=True)
                                  for a in read(setup_path)["attempts"] if a["status"] != "complete")
    window_jf = {d: {m: {f: bootstrap([r["scores"]["switch_window"][f] for r in metrics
                    if r["dataset"] == d and r["method"] == m and r["scores"]["switch_window"][f] is not None])
                    for f in ("J", "F", "J_and_F")} for m in methods} for d in DATASETS}
    audit = dict(training_counts=training_counts, evaluation_counts=dict(counts),
                 length_bins={d: dict(Counter(c["length_bin"] for c in cases if c["dataset"] == d)) for d in DATASETS},
                 no_training_validation_evaluation_video_overlap=True,
                 fit_only=True, gates=len(gates), no_replay_contract_checked=True,
                 anchor_replay_16_contract_checked=True, scored_frame_contract_checked=True,
                 checkpoint_and_input_provenance_verified=True, cost_cases=len(cost_rows),
                 failed=failed, incomplete=missing, evaluation_seconds=budget["elapsed_seconds"],
                 excluded_setup_failures=len(setup_failures),
                 prior_evaluation_sources=prior_sources,
                 previously_evaluated_counts=dict(Counter(c["dataset"] for c in cases if c["previously_evaluated"])),
                 last_mask_empty_prompts=dict(Counter(r["dataset"] for r in metrics
                     if r["method"] == "last_mask" and r["runtime"].get("empty_prompt"))),
                 checkpoint_at_epoch_limit={m: training["models"][m]["epoch"] == 8 for m in MODELS})
    reuse_path = run_dir / "downloaded_pair_reuse.json"
    reuse = read(reuse_path) if reuse_path.exists() else None
    if reuse:
        assert all(r.get("verified_reusable") is not None for r in reuse["candidate_pairs"])
        audit["downloaded_pair_reuse"] = dict(candidates=len(reuse["candidate_pairs"]),
            verified_reusable=sum(r["verified_reusable"] is True for r in reuse["candidate_pairs"]))
    output.mkdir(parents=True)
    write(output / "audit.json", audit)
    write(output / "summary.json", summary)
    write(output / "first10_J_F_JF.json", window_jf)
    write(output / "selection.json", portable(selection_raw))
    write(output / "training.json", portable(dict(training, histories=histories)))
    write(output / "provenance.json", portable(prov))
    write(output / "budget.json", budget)
    write(output / "schedule.json", read(run_dir / "schedule.json"))
    if (run_dir / "extension_decision.json").exists():
        write(output / "extension_decision.json", read(run_dir / "extension_decision.json"))
    write(output / "costs.json", cost_rows)
    write(output / "gates.json", gates)
    write(output / "setup_failures.json", portable(setup_failures))
    if reuse:
        write(output / "downloaded_pair_reuse.json", reuse)
    write(output / "case_scores.json", metrics)
    write(output / "device.json", read(run_dir / "device.json"))
    (output / "summary.md").write_text((run_dir / "summary.md").read_text(), encoding="utf-8")
    (output / "REPORT.md").write_text(overview(summary, training, audit, cost_rows), encoding="utf-8")
    (output / "switch_frames.csv").write_bytes((run_dir / "switch_frames.csv").read_bytes())
    csv_file(output / "training_pairs.csv", ("dataset", "video_id", "split", "path", "sha256", "valid_records", "upstream_commit"), portable(training["collection"]["pairs"]))
    csv_file(output / "evaluation_cases.csv", ("case_id", "dataset", "video_id", "object_id", "cohort", "length_bin", "slot", "first", "switch", "end", "first_prompt_stem", "switch_stem", "end_stem", "data_split", "prior_selection_status", "checkpoint_video", "previously_evaluated", "prior_evaluation_runs", "input_sha256", "annotation_sha256"), cases)
    plot_training(histories, training, output / "training_curves.svg")
    write(output / "files.sha256.json", {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    return audit


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--training-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-per-dataset", type=int, default=40)
    a = p.parse_args()
    print(export(a.run_dir.resolve(), a.output.resolve(), a.training_dir.resolve(), a.expected_per_dataset))
