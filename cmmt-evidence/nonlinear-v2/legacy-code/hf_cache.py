"""Duck-typed Hugging Face KV-cache normalization for inspection experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch


KV_AXES = ("batch", "kv_head", "sequence", "head_dim")


@dataclass(frozen=True)
class KVLayer:
    index: int
    key: torch.Tensor
    value: torch.Tensor

    def validate(self) -> "KVLayer":
        for name, tensor in (("key", self.key), ("value", self.value)):
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
                raise ValueError(
                    f"layer {self.index} {name} must be [B,H_kv,T,D], got "
                    f"{type(tensor).__name__} {getattr(tensor, 'shape', None)}"
                )
        if self.key.shape[:3] != self.value.shape[:3]:
            raise ValueError(
                f"layer {self.index} K/V batch, head and sequence axes differ: "
                f"{tuple(self.key.shape)} vs {tuple(self.value.shape)}"
            )
        return self


@dataclass(frozen=True)
class KVCacheView:
    layers: tuple[KVLayer, ...]
    source_layout: str

    def validate(self) -> "KVCacheView":
        if not self.layers:
            raise ValueError("KV cache contains no layers")
        for layer in self.layers:
            layer.validate()
        return self

    def contract_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": "cmmt.hf_kv_cache.v1",
            "source_layout": self.source_layout,
            "axes": list(KV_AXES),
            "dynamic_axes": ["batch", "sequence"],
            "layers": [
                {
                    "index": layer.index,
                    "key_shape": list(layer.key.shape),
                    "value_shape": list(layer.value.shape),
                    "key_dtype": str(layer.key.dtype),
                    "value_dtype": str(layer.value.dtype),
                    "device": str(layer.key.device),
                    "bytes": (
                        layer.key.numel() * layer.key.element_size()
                        + layer.value.numel() * layer.value.element_size()
                    ),
                }
                for layer in self.layers
            ],
        }


def _pairs_from_layers(layers: Iterable[Any]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for index, layer in enumerate(layers):
        if isinstance(layer, (tuple, list)) and len(layer) >= 2:
            key, value = layer[0], layer[1]
        elif hasattr(layer, "keys") and hasattr(layer, "values"):
            key, value = layer.keys, layer.values
        elif hasattr(layer, "key") and hasattr(layer, "value"):
            key, value = layer.key, layer.value
        else:
            raise TypeError(f"Unsupported cache layer {index}: {type(layer).__name__}")
        pairs.append((key, value))
    return pairs


def normalize_hf_cache(cache: Any) -> KVCacheView:
    """Normalize legacy tuples and current Transformers cache objects.

    No Transformers import is required.  This makes the probe usable against
    multiple library versions while keeping tensor ownership unchanged.
    """

    source_layout = type(cache).__name__
    if hasattr(cache, "to_legacy_cache"):
        legacy = cache.to_legacy_cache()
        pairs = _pairs_from_layers(legacy)
        source_layout += ".to_legacy_cache"
    elif hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        keys = list(cache.key_cache)
        values = list(cache.value_cache)
        if len(keys) != len(values):
            raise ValueError("key_cache and value_cache have different layer counts")
        pairs = list(zip(keys, values, strict=True))
        source_layout += ".key_cache/value_cache"
    elif hasattr(cache, "layers"):
        pairs = _pairs_from_layers(cache.layers)
        source_layout += ".layers"
    elif isinstance(cache, (tuple, list)):
        pairs = _pairs_from_layers(cache)
        source_layout = "legacy_tuple" if isinstance(cache, tuple) else "legacy_list"
    else:
        raise TypeError(f"Unsupported HF cache container: {type(cache).__name__}")

    return KVCacheView(
        tuple(KVLayer(index, key, value).validate() for index, (key, value) in enumerate(pairs)),
        source_layout,
    ).validate()


def flatten_kv_tokens(tensor: torch.Tensor) -> torch.Tensor:
    """Convert external ``[B,H,T,D]`` to mapper matrix ``[B*T,H*D]``."""

    if tensor.ndim != 4:
        raise ValueError(f"expected [B,H,T,D], got {tuple(tensor.shape)}")
    batch, heads, sequence, head_dim = tensor.shape
    return tensor.permute(0, 2, 1, 3).reshape(batch * sequence, heads * head_dim)


def unflatten_kv_tokens(
    matrix: torch.Tensor,
    *,
    batch: int,
    heads: int,
    sequence: int,
    head_dim: int,
) -> torch.Tensor:
    """Restore mapper output ``[B*T,H*D]`` to external ``[B,H,T,D]``."""

    if matrix.shape != (batch * sequence, heads * head_dim):
        raise ValueError(
            "matrix shape does not match requested cache axes: "
            f"{tuple(matrix.shape)} vs {(batch * sequence, heads * head_dim)}"
        )
    return matrix.reshape(batch, sequence, heads, head_dim).permute(0, 2, 1, 3)


def to_legacy_cache(view: KVCacheView) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    view.validate()
    return tuple((layer.key, layer.value) for layer in view.layers)
