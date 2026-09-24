#!/usr/bin/env python3
"""Build a compact interactive GitHub Pages gallery from one baseline suite."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


METHODS = [
    ("direct_copy", "Direct Copy", (225, 55, 190)),
    ("target_reset", "Target Reset", (120, 120, 120)),
    ("last_mask", "Last-Mask", (45, 135, 255)),
    ("replay_1", "Replay-1", (30, 190, 210)),
    ("replay_2", "Replay-2", (245, 180, 35)),
    ("replay_4", "Replay-4", (145, 90, 230)),
    ("full_replay", "Full Replay / Target-native", (30, 210, 80)),
]


def _mask(path: Path, *, object_id: int | None = None) -> np.ndarray:
    array = np.asarray(Image.open(path))
    return array == object_id if object_id is not None else array > 0


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(image.resize(size, Image.Resampling.NEAREST)) > 0


def _overlay(
    rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]
) -> Image.Image:
    result = rgb.astype(np.float32).copy()
    result[mask] = result[mask] * 0.42 + np.asarray(color) * 0.58
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def _panel(image: Image.Image, title: str, score: float | None) -> Image.Image:
    result = Image.new("RGB", (image.width, image.height + 34), "white")
    result.paste(image, (0, 34))
    label = title if score is None else f"{title}   GT J&F {score:.3f}"
    ImageDraw.Draw(result).text((8, 10), label, fill="black")
    return result


def _scores(path: Path) -> dict[int, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["frame"]): float(row["J_and_F"]) for row in payload["frames"]}


def _html(records: list[dict[str, object]], summary: dict[str, object]) -> str:
    method_options = "".join(
        f'<option value="{method_id}">{label}</option>'
        for method_id, label, _color in METHODS
    )
    rows = []
    for row in summary["methods"]:
        visible_jf = row.get("mean_visible_J_and_F")
        visible_jf_text = "n/a" if visible_jf is None else f"{visible_jf:.6f}"
        rows.append(
            "<tr><td>{label}</td><td>{prompt_source}</td><td>{history_frames_reprocessed}</td>"
            "<td>{mean_J_and_F:.6f}</td><td>{visible_jf_text}</td><td>{mean_binary_iou_to_target_native:.6f}</td>"
            "<td>{wall_time_seconds:.3f}</td><td>{backbone_calls_before_or_at_switch}</td></tr>".format(
                visible_jf_text=visible_jf_text, **row
            )
        )
    sequence = str(summary["sequence"])
    object_id = int(summary["object_id"])
    switch_frame = int(summary["switch_frame"])
    start_frame = int(summary["start_frame"])
    end_frame = int(summary["end_frame"])
    template = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CMMT cached baseline suite</title>
<style>
body{margin:0;background:#0d1117;color:#e6edf3;font:15px/1.55 system-ui,sans-serif}main{max-width:1420px;margin:36px auto;padding:0 20px}a{color:#58a6ff}.card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:18px;margin:18px 0}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.grid img{width:100%;border-radius:8px;background:#fff}.controls{display:grid;grid-template-columns:1fr 1fr;gap:12px}.controls label{display:flex;flex-direction:column;gap:5px}select,input{width:100%}table{border-collapse:collapse;width:100%;overflow:auto;display:block}th,td{padding:8px 10px;border-bottom:1px solid #30363d;text-align:right;white-space:nowrap}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}.note{color:#9da7b3}.score{font-family:ui-monospace,monospace}canvas{width:100%;height:360px;background:white;border-radius:8px}video{width:100%;border-radius:8px}@media(max-width:850px){.grid,.controls{grid-template-columns:1fr}}
</style></head><body><main>
<p class="note">Cross-Model Memory Translator · Phase 1 engineering result</p>
<h1>DAVIS __SEQUENCE__ baseline 비교</h1>
<p>Object __OBJECT_ID__, switch frame __SWITCH_FRAME__ 이후 frames __START_FRAME__–__END_FRAME__의 단일 사례 결과입니다. 전체 DAVIS benchmark가 아닙니다.</p>
<div class="card"><h2>핵심 결과</h2><div style="overflow:auto"><table><thead><tr><th>방법</th><th>입력</th><th>과거 처리 frame</th><th>전체 J&amp;F</th><th>GT-visible J&amp;F</th><th>Native IoU</th><th>시간(s)</th><th>과거 backbone</th></tr></thead><tbody>__TABLE_ROWS__</tbody></table></div>
<p class="note">Visible J&amp;F는 GT에 대상이 존재하는 프레임만 평균냅니다. Target Reset은 SAM 2 객체 슬롯 등록을 위해 switch frame에 빈 mask만 넣은 proxy입니다. Last-Mask와 Replay-1은 정의상 같은 결과여야 하며 실제로 일치했습니다.</p></div>
<div class="card"><h2>전체 결과 영상</h2><video controls loop muted src="baseline_suite.mp4"></video><p class="note">영상 패널: GT / Target-native / Direct / Last-Mask / Replay-2 / Replay-4.</p></div>
<div class="card"><h2>Frame별 J&amp;F</h2><canvas id="chart" width="1320" height="360"></canvas></div>
<div class="card"><h2 id="frame-title"></h2><input id="slider" type="range" min="0" value="0"><div class="controls"><label>비교 방법 A<select id="method-a">__METHOD_OPTIONS__</select></label><label>비교 방법 B<select id="method-b">__METHOD_OPTIONS__</select></label></div><p id="scores" class="score"></p><div class="grid"><img id="gt"><img id="native"><img id="a"><img id="b"></div></div>
<p class="note">Dataset: <a href="https://davischallenge.org/">DAVIS 2017</a>, <a href="https://creativecommons.org/licenses/by-nc/4.0/">CC BY-NC 4.0</a>. 비상업적 연구 평가 목적으로 사용했으며 DAVIS benchmark 논문을 인용합니다.</p>
<script>
const records=__RECORDS__;const methods=__METHODS__;const colors=__COLORS__;
const slider=document.querySelector('#slider'),aSel=document.querySelector('#method-a'),bSel=document.querySelector('#method-b');slider.max=records.length-1;aSel.value='last_mask';bSel.value='replay_2';
function fmt(x){return x==null?'n/a':x.toFixed(3)}
function show(){const r=records[+slider.value],a=aSel.value,b=bSel.value;document.querySelector('#frame-title').textContent=`Frame ${r.frame}`;document.querySelector('#gt').src=r.images.target_native;document.querySelector('#native').src=r.images.target_native;document.querySelector('#a').src=r.images[a];document.querySelector('#b').src=r.images[b];document.querySelector('#scores').textContent=`Target-native ${fmt(r.scores.full_replay)} | ${methods[a]} ${fmt(r.scores[a])} | ${methods[b]} ${fmt(r.scores[b])}`}
slider.oninput=show;aSel.onchange=show;bSel.onchange=show;show();
const canvas=document.querySelector('#chart'),ctx=canvas.getContext('2d'),W=canvas.width,H=canvas.height,m=48;ctx.strokeStyle='#ddd';ctx.fillStyle='#222';ctx.font='12px system-ui';for(let t=0;t<=4;t++){const y=H-m-t*(H-2*m)/4;ctx.beginPath();ctx.moveTo(m,y);ctx.lineTo(W-m,y);ctx.stroke();ctx.fillText((t/4).toFixed(2),8,y+4)};for(const [id,label] of Object.entries(methods)){ctx.strokeStyle=colors[id];ctx.lineWidth=id==='full_replay'?4:2;ctx.beginPath();records.forEach((r,i)=>{const x=m+i*(W-2*m)/(records.length-1),y=H-m-r.scores[id]*(H-2*m);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke()}let lx=m;for(const [id,label] of Object.entries(methods)){ctx.fillStyle=colors[id];ctx.fillRect(lx,10,14,10);ctx.fillStyle='#222';ctx.fillText(label,lx+18,20);lx+=Math.max(118,label.length*8+35)}
</script></main></body></html>"""
    labels = {method_id: label for method_id, label, _color in METHODS}
    colors = {
        method_id: "#%02x%02x%02x" % color
        for method_id, _label, color in METHODS
    }
    return (
        template.replace("__TABLE_ROWS__", "".join(rows))
        .replace("__SEQUENCE__", sequence)
        .replace("__OBJECT_ID__", str(object_id))
        .replace("__SWITCH_FRAME__", str(switch_frame))
        .replace("__START_FRAME__", str(start_frame))
        .replace("__END_FRAME__", str(end_frame))
        .replace("__METHOD_OPTIONS__", method_options)
        .replace("__RECORDS__", json.dumps(records, separators=(",", ":")))
        .replace("__METHODS__", json.dumps(labels, separators=(",", ":")))
        .replace("__COLORS__", json.dumps(colors, separators=(",", ":")))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", required=True, type=Path)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--annotation-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--jpeg-quality", type=int, default=72)
    args = parser.parse_args()

    suite_dir = args.suite_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    summary = json.loads((suite_dir / "summary.json").read_text(encoding="utf-8"))
    start_frame = int(summary["start_frame"])
    end_frame = int(summary["end_frame"])
    scores = {
        method_id: _scores(suite_dir / method_id / "davis.json")
        for method_id, _label, _color in METHODS
    }
    method_meta = {
        method_id: (label, color) for method_id, label, color in METHODS
    }
    native_mask_dir = suite_dir / "full_replay" / "candidate_masks"
    records: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="cmmt-baseline-video-") as temp:
        video_frames = Path(temp)
        for frame in range(start_frame, end_frame + 1):
            stem = f"{frame:05d}"
            with Image.open(args.video_dir / f"{stem}.jpg") as image:
                rgb_image = image.convert("RGB")
            height = round(rgb_image.height * args.width / rgb_image.width)
            rgb_image = rgb_image.resize((args.width, height), Image.Resampling.LANCZOS)
            rgb = np.asarray(rgb_image)
            gt = _resize_mask(
                _mask(args.annotation_dir / f"{stem}.png", object_id=args.object_id),
                rgb_image.size,
            )
            native = _resize_mask(_mask(native_mask_dir / f"{stem}.png"), rgb_image.size)
            images: dict[str, str] = {}
            gt_name = f"gt_{stem}.jpg"
            _panel(_overlay(rgb, gt, (255, 45, 35)), "DAVIS GT", None).save(
                frames_dir / gt_name, quality=args.jpeg_quality, optimize=True
            )
            images["gt"] = f"frames/{gt_name}"
            native_name = f"target_native_{stem}.jpg"
            native_panel = _panel(
                _overlay(rgb, native, (30, 210, 80)),
                "Target-native",
                scores["full_replay"][frame],
            )
            native_panel.save(
                frames_dir / native_name, quality=args.jpeg_quality, optimize=True
            )
            images["target_native"] = f"frames/{native_name}"
            for method_id, (label, color) in method_meta.items():
                if method_id == "full_replay":
                    images[method_id] = images["target_native"]
                    continue
                candidate = _resize_mask(
                    _mask(suite_dir / method_id / "candidate_masks" / f"{stem}.png"),
                    rgb_image.size,
                )
                name = f"{method_id}_{stem}.jpg"
                _panel(
                    _overlay(rgb, candidate, color), label, scores[method_id][frame]
                ).save(frames_dir / name, quality=args.jpeg_quality, optimize=True)
                images[method_id] = f"frames/{name}"
            records.append(
                {
                    "frame": frame,
                    "images": images,
                    "scores": {
                        method_id: values[frame] for method_id, values in scores.items()
                    },
                }
            )

            video_panels = [
                Image.open(frames_dir / gt_name),
                Image.open(frames_dir / native_name),
                Image.open(frames_dir / f"direct_copy_{stem}.jpg"),
                Image.open(frames_dir / f"last_mask_{stem}.jpg"),
                Image.open(frames_dir / f"replay_2_{stem}.jpg"),
                Image.open(frames_dir / f"replay_4_{stem}.jpg"),
            ]
            canvas = Image.new(
                "RGB", (sum(panel.width for panel in video_panels), video_panels[0].height)
            )
            x = 0
            for panel_image in video_panels:
                canvas.paste(panel_image, (x, 0))
                x += panel_image.width
                panel_image.close()
            canvas.save(
                video_frames / f"frame_{stem}.jpg", quality=args.jpeg_quality, optimize=True
            )

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                "6",
                "-start_number",
                str(start_frame),
                "-i",
                str(video_frames / "frame_%05d.jpg"),
                "-vf",
                "scale=1920:-2",
                "-c:v",
                "libx264",
                "-crf",
                "26",
                "-pix_fmt",
                "yuv420p",
                str(output_dir / "baseline_suite.mp4"),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    (output_dir / "frames.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "index.html").write_text(
        _html(records, summary), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "frames": len(records),
                "methods": len(METHODS),
                "output": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
