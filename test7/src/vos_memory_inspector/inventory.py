from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TensorRoute:
    producer: str
    consumer: str
    state_path: str
    temporal_class: str


FRAME_TENSOR_ROUTES: dict[str, TensorRoute] = {
    "maskmem_features": TensorRoute(
        producer=(
            "sam2.modeling.sam2_base.SAM2Base._encode_new_memory:"
            'memory_encoder()["vision_features"]'
        ),
        consumer=(
            "sam2.modeling.sam2_base.SAM2Base."
            '_prepare_memory_conditioned_features:prev["maskmem_features"]'
        ),
        state_path=(
            'inference_state["output_dict_per_obj"][object_idx]'
            '[cond_or_non_cond][frame_idx]["maskmem_features"]'
        ),
        temporal_class="spatial_memory",
    ),
    "maskmem_pos_enc": TensorRoute(
        producer=(
            "sam2.modeling.sam2_base.SAM2Base._encode_new_memory:"
            'memory_encoder()["vision_pos_enc"]'
        ),
        consumer=(
            "sam2.modeling.sam2_base.SAM2Base."
            '_prepare_memory_conditioned_features:prev["maskmem_pos_enc"][-1]'
        ),
        state_path=(
            'inference_state["output_dict_per_obj"][object_idx]'
            '[cond_or_non_cond][frame_idx]["maskmem_pos_enc"]'
            ' -> inference_state["constants"]["maskmem_pos_enc"]'
        ),
        temporal_class="spatial_position_constant",
    ),
    "pred_masks": TensorRoute(
        producer="sam2.modeling.sam2_base.SAM2Base.track_step:pred_masks",
        consumer=(
            "sam2.sam2_video_predictor.SAM2VideoPredictor."
            "add_new_points_or_box:previous same-frame correction logits; "
            "not read by next-frame memory attention"
        ),
        state_path=(
            'inference_state["output_dict_per_obj"][object_idx]'
            '[cond_or_non_cond][frame_idx]["pred_masks"]'
        ),
        temporal_class="supporting_prediction_state",
    ),
    "obj_ptr": TensorRoute(
        producer=(
            "sam2.modeling.sam2_base.SAM2Base._forward_sam_heads:"
            "obj_ptr_proj(sam_output_token)"
        ),
        consumer=(
            "sam2.modeling.sam2_base.SAM2Base."
            '_prepare_memory_conditioned_features:out["obj_ptr"]'
        ),
        state_path=(
            'inference_state["output_dict_per_obj"][object_idx]'
            '[cond_or_non_cond][frame_idx]["obj_ptr"]'
        ),
        temporal_class="object_pointer_memory",
    ),
    "object_score_logits": TensorRoute(
        producer=(
            "sam2.modeling.sam.mask_decoder.MaskDecoder.predict_masks:"
            "pred_obj_score_head"
        ),
        consumer=(
            "sam2.modeling.sam2_base.SAM2Base._encode_new_memory:"
            "current-frame no-object spatial gating; not read from prior frame"
        ),
        state_path=(
            'inference_state["output_dict_per_obj"][object_idx]'
            '[cond_or_non_cond][frame_idx]["object_score_logits"]'
        ),
        temporal_class="supporting_gating_state",
    ),
}


ATTENTION_INPUT_ROUTES: dict[str, TensorRoute] = {
    "memory_attention.memory": TensorRoute(
        producer=(
            "sam2.modeling.sam2_base.SAM2Base."
            "_prepare_memory_conditioned_features:torch.cat(to_cat_memory)"
        ),
        consumer=(
            "sam2.modeling.memory_attention.MemoryAttention.forward:memory"
        ),
        state_path="transient (not stored in inference_state)",
        temporal_class="assembled_attention_input",
    ),
    "memory_attention.memory_pos": TensorRoute(
        producer=(
            "sam2.modeling.sam2_base.SAM2Base."
            "_prepare_memory_conditioned_features:torch.cat(to_cat_memory_pos_embed)"
        ),
        consumer=(
            "sam2.modeling.memory_attention.MemoryAttention.forward:memory_pos"
        ),
        state_path="transient (not stored in inference_state)",
        temporal_class="assembled_attention_position_input",
    ),
}


FRAME_STATE_KEYS = tuple(FRAME_TENSOR_ROUTES)

