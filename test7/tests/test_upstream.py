from __future__ import annotations

from vos_memory_inspector.upstream import (
    DEFAULT_SAM2_UPSTREAM_COMMIT,
    sam2_upstream_commit,
)


def test_sam2_commit_uses_environment_then_dotenv_then_default(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("SAM2_UPSTREAM_COMMIT", raising=False)
    monkeypatch.setattr(
        "vos_memory_inspector.upstream._dotenv_candidates",
        lambda: [tmp_path / ".env"],
    )

    assert sam2_upstream_commit() == DEFAULT_SAM2_UPSTREAM_COMMIT
    assert DEFAULT_SAM2_UPSTREAM_COMMIT == "2b90b9f5ceec907a1c18123530e92e794ad901a4"

    (tmp_path / ".env").write_text(
        "SAM2_UPSTREAM_COMMIT=abc123\n",
        encoding="utf-8",
    )
    assert sam2_upstream_commit() == "abc123"

    monkeypatch.setenv("SAM2_UPSTREAM_COMMIT", "from-env")
    assert sam2_upstream_commit() == "from-env"
