from __future__ import annotations

import inspect
import types
from dataclasses import dataclass
from typing import Any, Mapping

from .inventory import ATTENTION_INPUT_ROUTES
from .manifest import DumpPolicy, ManifestWriter, TensorChangeTracker, tensor_stats
from .probe import ProbeConfig


@dataclass
class _FrameContext:
    frame_idx: int
    object_id: int | str
    cond_non_cond: str


class MemoryAttentionProbe:
    """Capture the actual assembled memory inputs at MemoryAttention.forward.

    The upstream pre-hook does not receive frame identifiers. A narrow wrapper
    around the pinned private preparation method supplies that context and is
    restored by ``detach``.
    """

    def __init__(
        self,
        predictor: Any,
        inference_state: Mapping[str, Any],
        config: ProbeConfig,
        writer: ManifestWriter,
        *,
        dump_policy: DumpPolicy | None = None,
        change_tracker: TensorChangeTracker | None = None,
    ) -> None:
        required = ("_prepare_memory_conditioned_features", "memory_attention")
        missing = [name for name in required if not hasattr(predictor, name)]
        if missing:
            raise RuntimeError(
                "Unsupported SAM 2 predictor for pinned probe; missing " + ", ".join(missing)
            )
        self.predictor = predictor
        self.inference_state = inference_state
        self.config = config
        self.writer = writer
        self.dump_policy = dump_policy or DumpPolicy()
        self.change_tracker = change_tracker or TensorChangeTracker()
        self._context: _FrameContext | None = None
        self._original_prepare = predictor._prepare_memory_conditioned_features
        self._hook_handle: Any = None
        self._attached = False

    def attach(self) -> "MemoryAttentionProbe":
        if self._attached:
            return self
        original = self._original_prepare
        signature = inspect.signature(original)

        def wrapped_prepare(bound_predictor: Any, *args: Any, **kwargs: Any) -> Any:
            arguments = signature.bind_partial(*args, **kwargs).arguments
            frame_idx = int(arguments["frame_idx"])
            is_init = bool(arguments["is_init_cond_frame"])
            output_dict = arguments["output_dict"]
            object_id = self._resolve_object_id(output_dict)
            prior = self._context
            self._context = _FrameContext(
                frame_idx=frame_idx,
                object_id=object_id,
                cond_non_cond="cond" if is_init else "non_cond",
            )
            try:
                return original(*args, **kwargs)
            finally:
                self._context = prior

        self.predictor._prepare_memory_conditioned_features = types.MethodType(
            wrapped_prepare, self.predictor
        )
        try:
            self._hook_handle = self.predictor.memory_attention.register_forward_pre_hook(
                self._pre_hook, with_kwargs=True
            )
        except TypeError as exc:
            self.predictor._prepare_memory_conditioned_features = self._original_prepare
            raise RuntimeError(
                "This PyTorch runtime lacks keyword-aware forward pre-hooks"
            ) from exc
        self._attached = True
        return self

    def detach(self) -> None:
        if not self._attached:
            return
        self.predictor._prepare_memory_conditioned_features = self._original_prepare
        if self._hook_handle is not None:
            self._hook_handle.remove()
        self._hook_handle = None
        self._attached = False
        self._context = None

    def __enter__(self) -> "MemoryAttentionProbe":
        return self.attach()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.detach()

    def _resolve_object_id(self, output_dict: Any) -> int | str:
        for object_idx, candidate in self.inference_state[
            "output_dict_per_obj"
        ].items():
            if candidate is output_dict:
                return self.inference_state["obj_idx_to_id"][object_idx]
        raise RuntimeError(
            "Could not associate memory preparation output_dict with an object"
        )

    def _pre_hook(
        self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        del module, args
        if self._context is None:
            raise RuntimeError(
                "MemoryAttention ran without frame context; pinned private API likely changed"
            )
        for tensor_name, kwarg_name in (
            ("memory_attention.memory", "memory"),
            ("memory_attention.memory_pos", "memory_pos"),
        ):
            tensor = kwargs.get(kwarg_name)
            if tensor is None:
                raise RuntimeError(
                    f"MemoryAttention pre-hook did not receive keyword {kwarg_name!r}"
                )
            route = ATTENTION_INPUT_ROUTES[tensor_name]
            changed, max_abs_delta = self.change_tracker.compare(
                (self._context.object_id, tensor_name), tensor
            )
            dump_path = self.dump_policy.dump(
                tensor=tensor,
                model_id=self.config.model_id,
                video_id=self.config.video_id,
                object_id=self._context.object_id,
                frame_idx=self._context.frame_idx,
                tensor_name=tensor_name,
            )
            self.writer.write(
                {
                    "model_id": self.config.model_id,
                    "upstream_commit": self.config.upstream_commit,
                    "video_id": self.config.video_id,
                    "frame_idx": self._context.frame_idx,
                    "switch_frame": self.config.switch_frame,
                    "object_id": self._context.object_id,
                    "cond_non_cond": self._context.cond_non_cond,
                    "tensor_name": tensor_name,
                    "producer": route.producer,
                    "consumer": route.consumer,
                    "state_path": route.state_path,
                    "temporal_class": route.temporal_class,
                    **tensor_stats(tensor),
                    "changed_from_previous": changed,
                    "max_abs_delta_from_previous": max_abs_delta,
                    "dump_path": dump_path,
                }
            )

