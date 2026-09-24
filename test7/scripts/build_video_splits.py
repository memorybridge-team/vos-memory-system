"""Create deterministic video-level fit/development manifests from a train manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _bucket(video_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{video_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def build_splits(source: dict, *, seed: int = 7, fit_fraction: float = 0.8) -> tuple[dict, dict]:
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("fit_fraction must be between 0 and 1")
    video_key = "video_id" if any("video_id" in case for case in source["cases"]) else "sequence"
    videos = sorted({str(case[video_key]) for case in source["cases"]})
    fit_videos = {video for video in videos if _bucket(video, seed) < fit_fraction}
    # Avoid an empty side for tiny smoke manifests.
    if not fit_videos:
        fit_videos.add(videos[0])
    if len(fit_videos) == len(videos):
        fit_videos.remove(videos[-1])

    def make(name: str, selected: set[str]) -> dict:
        cases = [case for case in source["cases"] if str(case[video_key]) in selected]
        result = {
            "schema_version": "cmmt.video_split_manifest.v1",
            "dataset": source["dataset"],
            "release": source.get("release"),
            "source_split": source["split"],
            "split": name,
            "seed": seed,
            "fit_fraction": fit_fraction,
            "video_key": video_key,
            "videos": sorted(selected),
            "video_count": len(selected),
            "case_count": len(cases),
            "source_manifest_content_sha256": source["content_sha256"],
        }
        canonical = json.dumps(result, sort_keys=True, separators=(",", ":"))
        result["content_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
        return result

    return make("fit", fit_videos), make("development", set(videos) - fit_videos)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fit-fraction", type=float, default=0.8)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    fit, development = build_splits(source, seed=args.seed, fit_fraction=args.fit_fraction)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_name(args.output_prefix.name + "_fit.json").write_text(json.dumps(fit, indent=2), encoding="utf-8")
    args.output_prefix.with_name(args.output_prefix.name + "_development.json").write_text(json.dumps(development, indent=2), encoding="utf-8")
    print(json.dumps({"fit_videos": fit["video_count"], "development_videos": development["video_count"], "fit_cases": fit["case_count"], "development_cases": development["case_count"]}, indent=2))


if __name__ == "__main__":
    main()
