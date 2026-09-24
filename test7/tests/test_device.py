"""Device selection for CUDA, Apple MPS, and CPU."""

import pytest
import torch

from vos_memory_inspector.device import mps_available, resolve_device


def test_auto_prefers_cuda_then_mps_then_cpu() -> None:
    if torch.cuda.is_available():
        expected = "cuda"
    elif mps_available():
        expected = "mps"
    else:
        expected = "cpu"
    assert resolve_device(None) == expected
    assert resolve_device("auto") == expected
    assert resolve_device("cpu") == "cpu"


def test_sam2_auto_skips_mps() -> None:
    if torch.cuda.is_available():
        assert resolve_device("auto", allow_mps=False) == "cuda"
    else:
        assert resolve_device("auto", allow_mps=False) == "cpu"


def test_explicit_cuda_requires_a_gpu() -> None:
    if torch.cuda.is_available():
        assert resolve_device("cuda") == "cuda"
        assert resolve_device("cuda:0") == "cuda:0"
        return
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_device("cuda")


def test_explicit_mps_requires_the_backend() -> None:
    if mps_available():
        assert resolve_device("mps") == "mps"
        return
    with pytest.raises(RuntimeError, match="MPS is not available"):
        resolve_device("mps")
