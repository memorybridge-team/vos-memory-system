"""Same-query diagnostics for target-native memory-attention consumption."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch.nn import functional as F


def _comparison(predicted: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    predicted = predicted.detach().float()
    reference = reference.detach().float()
    if predicted.shape != reference.shape:
        raise ValueError(
            f"consumer diagnostic shape mismatch: {tuple(predicted.shape)} != "
            f"{tuple(reference.shape)}"
        )
    return {
        "mse": float(F.mse_loss(predicted, reference).cpu()),
        "cosine": float(
            F.cosine_similarity(
                predicted.reshape(1, -1), reference.reshape(1, -1), eps=1e-12
            ).cpu()
        ),
    }


def _run_and_capture(
    memory_attention: torch.nn.Module,
    *,
    curr: torch.Tensor | list[torch.Tensor],
    memory: torch.Tensor,
    curr_pos: torch.Tensor | list[torch.Tensor],
    memory_pos: torch.Tensor,
    num_obj_ptr_tokens: int,
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    captures: list[dict[str, torch.Tensor]] = [
        {} for _ in range(len(memory_attention.layers))
    ]
    handles = []

    for layer_index, layer in enumerate(memory_attention.layers):
        attention = layer.cross_attn_image
        if not hasattr(attention, "k_proj") or not hasattr(attention, "v_proj"):
            raise RuntimeError(
                f"cross-attention layer {layer_index} lacks k_proj/v_proj"
            )

        def pre_hook(
            module: torch.nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            *,
            index: int = layer_index,
        ) -> None:
            del args
            key = kwargs.get("k")
            value = kwargs.get("v")
            if key is None or value is None:
                raise RuntimeError("cross-attention did not expose keyword k/v inputs")
            captures[index]["k_projection"] = module.k_proj(key).detach().cpu()
            captures[index]["v_projection"] = module.v_proj(value).detach().cpu()

        def post_hook(
            module: torch.nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: torch.Tensor,
            *,
            index: int = layer_index,
        ) -> None:
            del module, args, kwargs
            captures[index]["cross_attention_output"] = output.detach().cpu()

        handles.append(
            attention.register_forward_pre_hook(pre_hook, with_kwargs=True)
        )
        handles.append(
            attention.register_forward_hook(post_hook, with_kwargs=True)
        )
    try:
        with torch.no_grad():
            output = memory_attention(
                curr=curr,
                memory=memory,
                curr_pos=curr_pos,
                memory_pos=memory_pos,
                num_obj_ptr_tokens=num_obj_ptr_tokens,
            )
    finally:
        for handle in handles:
            handle.remove()
    required = {"k_projection", "v_projection", "cross_attention_output"}
    if any(set(row) != required for row in captures):
        raise RuntimeError("one or more memory-attention layers were not executed")
    return output.detach().cpu(), captures


def compare_same_query_memory_attention(
    memory_attention: torch.nn.Module,
    *,
    curr: torch.Tensor | list[torch.Tensor],
    curr_pos: torch.Tensor | list[torch.Tensor],
    native_memory: torch.Tensor,
    native_memory_pos: torch.Tensor,
    translated_memory: torch.Tensor,
    translated_memory_pos: torch.Tensor,
    num_obj_ptr_tokens: int,
) -> Mapping[str, Any]:
    """Compare two memories using the same frozen target query and target PE."""

    if native_memory.shape != translated_memory.shape:
        raise ValueError("native and translated assembled memory shapes must match")
    if native_memory_pos.shape != translated_memory_pos.shape:
        raise ValueError("native and translated memory PE shapes must match")
    native_output, native_layers = _run_and_capture(
        memory_attention,
        curr=curr,
        memory=native_memory,
        curr_pos=curr_pos,
        memory_pos=native_memory_pos,
        num_obj_ptr_tokens=num_obj_ptr_tokens,
    )
    translated_output, translated_layers = _run_and_capture(
        memory_attention,
        curr=curr,
        memory=translated_memory,
        curr_pos=curr_pos,
        memory_pos=translated_memory_pos,
        num_obj_ptr_tokens=num_obj_ptr_tokens,
    )
    layers = []
    for index, (translated, native) in enumerate(
        zip(translated_layers, native_layers, strict=True)
    ):
        layers.append(
            {
                "layer": index,
                "k_projection": _comparison(
                    translated["k_projection"], native["k_projection"]
                ),
                "v_projection": _comparison(
                    translated["v_projection"], native["v_projection"]
                ),
                "cross_attention_output": _comparison(
                    translated["cross_attention_output"],
                    native["cross_attention_output"],
                ),
            }
        )
    return {
        "schema_version": "cmmt.consumer_diagnostic.v1",
        "same_query": True,
        "num_obj_ptr_tokens": int(num_obj_ptr_tokens),
        "layers": layers,
        "full_memory_attention_output": _comparison(
            translated_output, native_output
        ),
    }
