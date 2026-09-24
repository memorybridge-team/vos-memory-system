from __future__ import annotations

from pathlib import Path

from vos_memory_inspector.research_task import (
    Evidence,
    TaskRecord,
    load_task,
    main,
    write_task,
)


ISSUE_URL = (
    "https://github.com/memorybridge-team/"
    "vos-memory-translator-nonlinear/issues/7"
)


def make_repository(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
    return tmp_path


def test_write_task_creates_json_and_report(tmp_path: Path) -> None:
    root = make_repository(tmp_path)
    path = root / "tasks/model-state/task.json"
    task = TaskRecord(
        task_id="model-state",
        title="Map state",
        issue_url=ISSUE_URL,
        owner="KIMKYUDO",
        acceptance_criteria=["shape table exists"],
    )
    write_task(path, task)
    assert load_task(path).status == "todo"
    report = path.with_name("report.md").read_text(encoding="utf-8")
    assert "- [ ] shape table exists" in report


def test_completion_rejects_missing_evidence(tmp_path: Path) -> None:
    root = make_repository(tmp_path)
    task = TaskRecord(
        task_id="model-state",
        title="Map state",
        issue_url=ISSUE_URL,
        owner="KIMKYUDO",
        acceptance_criteria=["shape table exists"],
        checked_criteria=["shape table exists"],
        result_summary="done",
    )
    errors = task.validate_completion(root)
    assert "at least one reproduction or validation command is required" in errors
    assert "at least one artifact is required" in errors


def test_noncanonical_or_malformed_issue_is_rejected() -> None:
    invalid_urls = [
        "https://github.com/memorybridge-team/unrelated-repository/issues/1",
        "https://github.com/memorybridge-team/vos-memory-translator-nonlinear/issues/",
        "https://github.com/memorybridge-team/vos-memory-translator-nonlinear/issues/7extra",
    ]
    for issue_url in invalid_urls:
        task = TaskRecord(
            task_id="wrong-issue",
            title="Wrong issue",
            issue_url=issue_url,
            owner="KIMKYUDO",
            acceptance_criteria=["use an exact canonical Issue URL"],
        )
        assert (
            "issue_url must point to the canonical nonlinear repository"
            in task.validate_schema()
        )


def test_completion_accepts_checked_existing_artifact(tmp_path: Path) -> None:
    root = make_repository(tmp_path)
    artifact = root / "reports/state.md"
    artifact.parent.mkdir()
    artifact.write_text("verified", encoding="utf-8")
    task = TaskRecord(
        task_id="model-state",
        title="Map state",
        issue_url=ISSUE_URL,
        owner="KIMKYUDO",
        acceptance_criteria=["shape table exists"],
        checked_criteria=["shape table exists"],
        evidence=[Evidence(artifact="reports/state.md", command="pytest -q")],
        result_summary="state contract verified",
    )
    assert task.validate_completion(root) == []


def test_cli_finish_is_evidence_gated(tmp_path: Path) -> None:
    root = make_repository(tmp_path)
    task_path = root / "tasks/state/task.json"
    criterion = "shape table exists"
    main(
        [
            "init",
            "--task",
            str(task_path),
            "--task-id",
            "state",
            "--title",
            "Map state",
            "--issue-url",
            ISSUE_URL,
            "--owner",
            "KIMKYUDO",
            "--criterion",
            criterion,
        ]
    )
    main(["start", "--task", str(task_path)])
    artifact = root / "reports/state.md"
    artifact.parent.mkdir()
    artifact.write_text("verified", encoding="utf-8")
    main(
        [
            "record",
            "--task",
            str(task_path),
            "--artifact",
            "reports/state.md",
            "--command-line",
            "pytest -q",
        ]
    )
    main(["check", "--task", str(task_path), "--criterion", criterion])
    main(["finish", "--task", str(task_path), "--summary", "verified"])
    assert load_task(task_path).status == "done"
