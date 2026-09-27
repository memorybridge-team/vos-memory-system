"""Small filesystem helpers shared by experiment artifact writers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _partial_path(path: Path) -> Path:
    return path.with_name(path.name + ".partial")


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    """Write text through a sibling ``.partial`` file and atomically replace ``path``."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = _partial_path(path)
    partial.write_text(text, encoding=encoding)
    partial.replace(path)
    return path


def atomic_torch_save(payload: Any, path: str | Path) -> Path:
    """``torch.save`` through a sibling ``.partial`` file and atomically replace ``path``."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = _partial_path(path)
    torch.save(payload, partial)
    partial.replace(path)
    return path


def portable_path(path: str | Path, *, base: str | Path | None = None) -> str:
    """Return a machine-independent POSIX path for reports.

    Paths inside ``base`` (default: the current working directory) are stored
    relative to it. Paths outside it are reduced to their file name so local
    filesystem layouts never leak into committed artifacts.
    """

    resolved = Path(path).resolve()
    root = (Path.cwd() if base is None else Path(base)).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.name
