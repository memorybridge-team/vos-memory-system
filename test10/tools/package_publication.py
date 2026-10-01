#!/usr/bin/env python3
"""Lossless publication of existing experiment outputs; never runs inference.

Large prediction/state tensor caches are inventoried, not silently uploaded.
Archives retain each member's original bytes and relative path. Existing result
exports, training reports and weights are never rewritten.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parents[1]
SECRET = re.compile(rb"AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{20}|github_pat_[A-Za-z0-9_]{20}|-----BEGIN .*PRIVATE KEY-----")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def clean_files(folder):
    return sorted(p for p in folder.rglob("*") if p.is_file()
                  and "__pycache__" not in p.parts and p.suffix != ".pyc")


def archive(path, members, entries):
    print(f"Packaging {path.name}: {len(members)} original files", flush=True)
    with path.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0, compresslevel=6) as gz:
        with tarfile.open(fileobj=gz, mode="w|", format=tarfile.PAX_FORMAT) as tar:
            for source, member in sorted(members, key=lambda item: item[1]):
                if source.suffix in (".json", ".csv", ".md", ".py", ".log", ".toml", ".yaml"):
                    if SECRET.search(source.read_bytes()):
                        raise ValueError(f"Potential secret; publication stopped: {source}")
                before = source.stat()
                checksum = sha(source)
                info = tar.gettarinfo(str(source), arcname=member)
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                info.mode = 0o644
                with source.open("rb") as stream:
                    tar.addfile(info, stream)
                after = source.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError(f"Input changed during publication: {source}")
                entries.append(dict(archive=str(path.relative_to(ROOT.parent)), member=member,
                                    original_path=str(source), bytes=before.st_size, sha256=checksum))
    if path.stat().st_size >= 50_000_000:
        raise ValueError(f"Archive requires splitting before GitHub publication: {path}")


def verify_members(output):
    with (output / "archive_members.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    expected = {}
    for row in rows:
        expected.setdefault(row["archive"], {})[row["member"]] = row
    for relative, members in expected.items():
        seen = set()
        with tarfile.open(ROOT.parent / relative, "r|gz") as tar:
            for info in tar:
                if info.name not in members or info.name in seen or not info.isfile():
                    raise ValueError(f"Unexpected/duplicate archive member: {info.name}")
                h = hashlib.sha256()
                with tar.extractfile(info) as stream:
                    for block in iter(lambda: stream.read(4 << 20), b""):
                        h.update(block)
                row = members[info.name]
                if h.hexdigest() != row["sha256"] or info.size != int(row["bytes"]):
                    raise ValueError(f"Archive bytes differ from original: {info.name}")
                seen.add(info.name)
        if seen != set(members):
            raise ValueError(f"Missing archive members: {relative}")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--translator-repo", type=Path, default=Path("/home/home/test/test9/vos-memory-translator-nonlinear"))
    parser.add_argument("--sam2-repo", type=Path, default=Path("/home/home/test/sam2"))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("refuse to overwrite an existing publication")
    if not output.is_relative_to(ROOT / "results"):
        parser.error("publication must be inside test10/results")
    output.mkdir(parents=True)
    raw_dir = output / "raw_runs"
    raw_dir.mkdir()
    entries = []
    for name in ("fit1_eval40", "fit1_eval40_optimized", "fit1_eval40_fast", "downloaded_bank_first5"):
        run = ROOT / "runs" / name
        files = sorted(p for p in run.glob("*") if p.is_file() and not p.name.startswith("."))
        if (run / "artifacts").is_dir():
            files += sorted(p for p in (run / "artifacts").iterdir()
                            if p.is_file() and p.suffix in (".json", ".sha256"))
        archive(raw_dir / (name + "_metadata.tar.gz"),
                [(p, str(p.relative_to(ROOT.parent))) for p in files], entries)
    bank = ROOT / "runs/downloaded_bank_first5/scores"
    for dataset in sorted(p for p in bank.iterdir() if p.is_dir()):
        files = sorted(dataset.glob("*.json"))
        archive(raw_dir / ("downloaded_bank_first5_scores_" + dataset.name + ".tar.gz"),
                [(p, str(p.relative_to(ROOT.parent))) for p in files], entries)
    translator = args.translator_repo.resolve()
    source_files = clean_files(translator / "src") + clean_files(translator / "scripts")
    source_files += [translator / name for name in ("LICENSE", "pyproject.toml") if (translator / name).is_file()]
    archive(raw_dir / "translator_source.tar.gz",
            [(p, "translator/" + str(p.relative_to(translator))) for p in source_files], entries)
    sam2 = args.sam2_repo.resolve()
    source_files = [p for p in clean_files(sam2 / "sam2") if p.suffix in (".py", ".yaml", ".yml", ".json")]
    source_files += [sam2 / name for name in ("LICENSE", "LICENSE_cctorch", "setup.py", "pyproject.toml") if (sam2 / name).is_file()]
    archive(raw_dir / "sam2_source.tar.gz",
            [(p, "sam2/" + str(p.relative_to(sam2))) for p in source_files], entries)
    write_csv(output / "archive_members.csv", entries)
    report = json.loads((ROOT / "results/paper_metrics_20261001/metrics.json").read_text())
    excluded = [dict(original_path=p, bytes=Path(p).stat().st_size, sha256=h,
                     publication_status="local only; user approved inventory-and-SHA256 publication on 2026-10-02",
                     hash_source="existing CPU scoring report")
                for p, h in sorted(report["existing_mask_files_sha256"].items())]
    write_csv(output / "unpublished_prediction_caches.csv", excluded)
    files = clean_files(ROOT / "results") + clean_files(ROOT / "training/run_fit1000_val200")
    files += clean_files(ROOT / "tools") + clean_files(ROOT / "tests")
    files += sorted(p for p in ROOT.iterdir() if p.is_file())
    file_rows = [dict(path=str(p.relative_to(ROOT.parent)), bytes=p.stat().st_size, sha256=sha(p))
                 for p in sorted(set(files))]
    write_csv(output / "published_files.csv", file_rows)
    verified = verify_members(output)
    write_json(output / "bundle.json", dict(
        schema="test10.publication.v1", publication_date_kst="2026-10-02",
        source_export_dates_preserved=True, inference_performed=False, training_performed=False,
        original_result_files_rewritten=False, archived_member_count=verified,
        archived_original_bytes=sum(r["bytes"] for r in entries),
        listed_publication_files=len(file_rows), listed_publication_bytes=sum(r["bytes"] for r in file_rows),
        archived_members_sha256=sha(output / "archive_members.csv"),
        published_files_sha256=sha(output / "published_files.csv"),
        excluded_prediction_cache_files=len(excluded), excluded_prediction_cache_bytes=sum(r["bytes"] for r in excluded),
        exclusions=["Raw RGB/GT datasets and original downloaded training state-pair bank are existing inputs, not new results",
                    "User approved retaining prediction caches locally; each original path, size and recorded SHA-256 is inventoried",
                    "Other intermediate binary state caches, virtual environments, __pycache__ and lock files are not published"],
        verification="Every archived member was read back and matched to the original-file SHA-256 and size"))
    print(json.dumps(dict(output=str(output), archive_members_verified=verified,
                          files=len(file_rows), bytes=sum(r["bytes"] for r in file_rows))), flush=True)


if __name__ == "__main__":
    main()
