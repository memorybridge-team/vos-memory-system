#!/usr/bin/env python3
"""Static scientific figure of already-computed, matched visible-frame drift."""
import argparse
import json
import math
from html import escape
from pathlib import Path


def svg_fallback(directory, report):
    """Dependency-free standard SVG scientific chart, no generated observations."""
    rows = [r for r in report["drift"] if r["cohort"] == "native159_prompt0"]
    bins = ["1-5", "6-10", "11-20", "21-50", "51-100", "101+"]
    methods = [("affine", "Affine", "#0072b2"), ("residual_mlp", "MLP", "#e69f00"),
               ("transformer", "Transformer", "#009e73"), ("small_only", "Source Only", "#cc79a7")]
    elements = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="800" viewBox="0 0 1200 800">',
                '<rect width="1200" height="800" fill="white"/>', '<g font-family="sans-serif" fill="#222">']
    def text(x, y, value, size=12, anchor="start"):
        elements.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}">{escape(str(value))}</text>')
    text(600, 25, "Existing native-prefix evaluation: GT-visible drift", 20, "middle")
    text(600, 47, "Full Replay minus method: positive = method worse; point estimates, changing bin support", 13, "middle")
    for index, (_, label, color) in enumerate(methods):
        x = 290 + index * 165
        elements.append(f'<line x1="{x}" y1="70" x2="{x+25}" y2="70" stroke="{color}" stroke-width="3"/>')
        text(x + 32, 74, label)
    for index, dataset in enumerate(("MOSEv2", "LVOSv2", "DAVIS2017", "VOST")):
        left, top = 80 + (index % 2) * 590, 135 + (index // 2) * 330
        metric = "native_minus_method_J" if dataset == "VOST" else "native_minus_method_J_and_F"
        selected = [r for r in rows if r["dataset"] == dataset and r["method"] in {m[0] for m in methods}]
        values = [r[metric] for r in selected]
        low, high = 5*math.floor(min([0, *values])/5), 5*math.ceil(max([0, *values])/5)
        if low == high:
            high = low + 5
        def xp(i): return left + i*87
        def yp(v): return top + 220*(high-v)/(high-low)
        note = " (development, not official val)" if dataset == "MOSEv2" else " (train, exploratory)" if dataset == "DAVIS2017" else ""
        text(left + 210, top - 20, dataset + note, 14, "middle")
        for tick in range(low, high+1, 5):
            y = yp(tick)
            elements.append(f'<line x1="{left}" y1="{y}" x2="{left+435}" y2="{y}" stroke="#ddd"/>')
            text(left-8, y+4, tick, 11, "end")
        y = yp(0)
        elements.append(f'<line x1="{left}" y1="{y}" x2="{left+435}" y2="{y}" stroke="#555"/>')
        support = {r["offset_bin"]: r["videos"] for r in selected if r["method"] == "affine"}
        for i, bucket in enumerate(bins):
            text(xp(i), top+241, bucket, 11, "middle")
            text(xp(i), top+257, "n=" + str(support.get(bucket, 0)), 10, "middle")
        for method, _, color in methods:
            points = {r["offset_bin"]: r[metric] for r in selected if r["method"] == method}
            line = [(xp(i), yp(points[b])) for i, b in enumerate(bins) if b in points]
            dashed = ' stroke-dasharray="6 4"' if method == "small_only" else ""
            coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in line)
            elements.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"{dashed}/>')
            elements.extend(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"/>' for x, y in line)
        text(left, top+280, "Gap in " + ("J" if dataset == "VOST" else "J&F") + " points; x = processed post-switch offsets", 11)
    text(600, 770, "LVOS stride=5; VOST stride=6. n counts videos with visible GT in each bin.", 12, "middle")
    elements.append('</g></svg>')
    (directory / "drift.svg").write_text("\n".join(elements) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dir", type=Path)
    args = parser.parse_args()
    report = json.loads((args.report_dir / "metrics.json").read_text())
    try:
        import matplotlib
    except ModuleNotFoundError:
        svg_fallback(args.report_dir, report)
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in report["drift"] if r["cohort"] == "native159_prompt0"]
    bins = ["1-5", "6-10", "11-20", "21-50", "51-100", "101+"]
    labels = {"affine": "Affine", "residual_mlp": "MLP", "transformer": "Transformer", "small_only": "Source Only"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for ax, dataset in zip(axes.flat, ("MOSEv2", "LVOSv2", "DAVIS2017", "VOST")):
        metric = "native_minus_method_J" if dataset == "VOST" else "native_minus_method_J_and_F"
        for method, label in labels.items():
            by_bin = {r["offset_bin"]: r for r in rows if r["dataset"] == dataset and r["method"] == method}
            values = [by_bin[b][metric] if b in by_bin else float("nan") for b in bins]
            ax.plot(range(len(bins)), values, marker="o", linestyle="--" if method == "small_only" else "-", label=label)
        support = {r["offset_bin"]: r["videos"] for r in rows if r["dataset"] == dataset and r["method"] == "affine"}
        ax.set_xticks(range(len(bins)), [b + f"\nn={support.get(b, 0)}" for b in bins])
        note = " (train/development; NOT official val)" if dataset == "MOSEv2" else " (train/exploratory)" if dataset == "DAVIS2017" else ""
        ax.set_title(dataset + note, fontsize=10)
        ax.set_ylabel("Full Replay - method (" + ("J" if dataset == "VOST" else "J&F") + " points)")
        ax.set_xlabel("Processed post-switch offset; n = visible-support videos")
        ax.axhline(0, color="black", linewidth=.8)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Existing native-prefix evaluation: GT-visible drift, one independent object/video\nBin support changes; this is not a fixed-cohort deterioration test", fontsize=11)
    fig.savefig(args.report_dir / "drift.png", dpi=160)
    fig.savefig(args.report_dir / "drift.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
