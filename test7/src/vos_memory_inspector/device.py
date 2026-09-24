"""Pick a torch device: CUDA, then Apple MPS, otherwise CPU."""

from __future__ import annotations

import torch


def mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def resolve_device(
    requested: str | torch.device | None = None,
    *,
    allow_mps: bool = True,
) -> str:
    """Return the device string a caller should actually use.

    ``None`` and ``"auto"`` select ``cuda`` when CUDA is available. On a Mac
    they then select ``mps`` when ``allow_mps`` is true. Otherwise they select
    ``cpu``. ``cpu`` stays on CPU. An explicit accelerator is returned only
    when that backend exists.
    """

    if requested is None:
        text = "auto"
    else:
        text = str(requested).strip()
    key = text.lower()
    if key in {"", "auto"}:
        if torch.cuda.is_available():
            return "cuda"
        if allow_mps and mps_available():
            return "mps"
        return "cpu"
    if key == "cpu":
        return "cpu"
    if key.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"device {text!r} was requested but CUDA is not available"
            )
        return text
    if key == "mps":
        if not mps_available():
            raise RuntimeError("device 'mps' was requested but MPS is not available")
        return "mps"
    return text
