from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALID_STATUSES = {"todo", "in_progress", "blocked", "done"}
CANONICAL_ISSUE_RE = re.compile(
    r"^https://github\.com/memorybridge-team/"
    r"vos-memory-translator-nonlinear/issues/[1-9][0-9]*/?$"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class Evidence:
    artifact: str | None = None
    command: str | None = None
    note: str | None = None


@dataclass
class TaskRecord:
    task_id: str
    title: str
    issue_url: str
    owner: str
    acceptance_criteria: list[str]
    status: str = "todo"
    checked_criteria: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    result_summary: str | None = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TaskRecord":
        value = dict(value)
        value["evidence"] = [Evidence(**item) for item in value.get("evidence", [])]
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate_schema(self) -> list[str]:
        errors: list[str] = []
        if not self.task_id.strip():
            errors.append("task_id is required")
        if not self.title.strip():
            errors.append("title is required")
        if not self.owner.strip():
            errors.append("owner is required")
        if self.status not in VALID_STATUSES:
            errors.append(f"invalid status: {self.status}")
        if not CANONICAL_ISSUE_RE.fullmatch(self.issue_url):
            errors.append("issue_url must point to the canonical nonlinear repository")
        if not self.acceptance_criteria:
            errors.append("at least one acceptance criterion is required")
        unknown = set(self.checked_criteria) - set(self.acceptance_criteria)
        if unknown:
            errors.append(f"checked criteria are not declared: {sorted(unknown)}")
        return errors

    def validate_completion(self, repository_root: Path) -> list[str]:
        errors = self.validate_schema()
        missing = set(self.acceptance_criteria) - set(self.checked_criteria)
        if missing:
            errors.append(f"unchecked acceptance criteria: {sorted(missing)}")
        if not self.result_summary or not self.result_summary.strip():
            errors.append("result_summary is required before completion")
        if not any(item.command for item in self.evidence):
            errors.append("at least one reproduction or validation command is required")
        artifacts = [item.artifact for item in self.evidence if item.artifact]
        if not artifacts:
            errors.append("at least one artifact is required")
        for artifact in artifacts:
            path = repository_root / artifact
            if not path.exists():
                errors.append(f"artifact does not exist: {artifact}")
        return errors


def load_task(path: Path) -> TaskRecord:
    return TaskRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))


def write_task(path: Path, task: TaskRecord) -> None:
    errors = task.validate_schema()
    if errors:
        raise ValueError("; ".join(errors))
    task.updated_at = utc_now()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(task.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(path.with_name("report.md"), task)


def write_report(path: Path, task: TaskRecord) -> None:
    checked = set(task.checked_criteria)
    lines = [
        f"# {task.title}",
        "",
        f"- Task ID: `{task.task_id}`",
        f"- Issue: {task.issue_url}",
        f"- Owner: `{task.owner}`",
        f"- Status: `{task.status}`",
        "",
        "## 완료 기준",
        "",
    ]
    for criterion in task.acceptance_criteria:
        mark = "x" if criterion in checked else " "
        lines.append(f"- [{mark}] {criterion}")
    lines.extend(["", "## 증거", ""])
    if task.evidence:
        for item in task.evidence:
            parts = []
            if item.artifact:
                parts.append(f"artifact: `{item.artifact}`")
            if item.command:
                parts.append(f"command: `{item.command}`")
            if item.note:
                parts.append(item.note)
            lines.append(f"- {'; '.join(parts)}")
    else:
        lines.append("- 아직 기록된 증거가 없습니다.")
    lines.extend(
        [
            "",
            "## 결과",
            "",
            task.result_summary or "아직 완료되지 않았습니다.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def repository_root(task_path: Path) -> Path:
    current = task_path.resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ValueError("could not find repository root containing pyproject.toml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage evidence-gated CMMT research task records."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="Create a local task record.")
    init.add_argument("--task", required=True, type=Path)
    init.add_argument("--task-id", required=True)
    init.add_argument("--title", required=True)
    init.add_argument("--issue-url", required=True)
    init.add_argument("--owner", required=True)
    init.add_argument("--criterion", action="append", required=True)

    for name in ("start", "show"):
        command = subparsers.add_parser(name)
        command.add_argument("--task", required=True, type=Path)

    record = subparsers.add_parser("record")
    record.add_argument("--task", required=True, type=Path)
    record.add_argument("--artifact")
    record.add_argument("--command-line")
    record.add_argument("--note")

    check = subparsers.add_parser("check")
    check.add_argument("--task", required=True, type=Path)
    check.add_argument("--criterion", required=True)

    block = subparsers.add_parser("block")
    block.add_argument("--task", required=True, type=Path)
    block.add_argument("--reason", required=True)

    finish = subparsers.add_parser("finish")
    finish.add_argument("--task", required=True, type=Path)
    finish.add_argument("--summary", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    path: Path = args.task
    if args.command == "init":
        task = TaskRecord(
            task_id=args.task_id,
            title=args.title,
            issue_url=args.issue_url,
            owner=args.owner,
            acceptance_criteria=args.criterion,
        )
        write_task(path, task)
    else:
        task = load_task(path)
        if args.command == "start":
            task.status = "in_progress"
            write_task(path, task)
        elif args.command == "record":
            if not any((args.artifact, args.command_line, args.note)):
                raise SystemExit("record requires --artifact, --command-line, or --note")
            task.evidence.append(
                Evidence(
                    artifact=args.artifact,
                    command=args.command_line,
                    note=args.note,
                )
            )
            write_task(path, task)
        elif args.command == "check":
            if args.criterion not in task.acceptance_criteria:
                raise SystemExit("criterion was not declared by this task")
            if args.criterion not in task.checked_criteria:
                task.checked_criteria.append(args.criterion)
            write_task(path, task)
        elif args.command == "block":
            task.status = "blocked"
            task.evidence.append(Evidence(note=f"Blocker: {args.reason}"))
            write_task(path, task)
        elif args.command == "finish":
            task.result_summary = args.summary
            errors = task.validate_completion(repository_root(path))
            if errors:
                raise SystemExit("cannot finish task:\n- " + "\n- ".join(errors))
            task.status = "done"
            write_task(path, task)
        elif args.command == "show":
            pass
    print(json.dumps(task.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
