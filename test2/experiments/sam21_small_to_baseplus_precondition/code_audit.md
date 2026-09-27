# Verified SAM 2.1 memory path

- Upstream commit: `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- Producer path: `_track_step`/decoder → `_encode_new_memory` → `MemoryEncoder` → output dictionary.
- Consumer path: `_prepare_memory_conditioned_features` → target-native temporal assembly → `MemoryAttention`.
- `maskmem_features` and `maskmem_pos_enc` are stored separately.
- `no_obj_embed_spatial` is added after memory encoding.
- Stored `obj_ptr` has no temporal PE; PE and 4×64 tokenization happen at consumption.
- Pointer tokens are passed as `num_k_exclude_rope`, so their suffix is excluded from spatial RoPE.
- Source PE/RoPE stripping is therefore not part of this experiment.

## Verified source locations

Paths are relative to the SAM 2 checkout at the upstream commit above.

- `memory_consumer`: `sam2/modeling/sam2_base.py` line(s) 497
- `memory_producer`: `sam2/modeling/sam2_base.py` line(s) 678
- `memory_encoder_pix_projection`: `sam2/modeling/memory_encoder.py` line(s) 174
- `memory_encoder_mask_addition`: `sam2/modeling/memory_encoder.py` line(s) 175
- `memory_encoder_fuser`: `sam2/modeling/memory_encoder.py` line(s) 176
- `memory_encoder_output_projection`: `sam2/modeling/memory_encoder.py` line(s) 177
- `memory_encoder_positional_output`: `sam2/modeling/memory_encoder.py` line(s) 179
- `pointer_projection`: `sam2/modeling/sam2_base.py` line(s) 393
- `pointer_token_split`: `sam2/modeling/sam2_base.py` line(s) 641
- `no_object_spatial_addition`: `sam2/modeling/sam2_base.py` line(s) 720
- `stored_memory_assignment`: `sam2/modeling/sam2_base.py` line(s) 808
- `stored_positional_assignment`: `sam2/modeling/sam2_base.py` line(s) 809
- `pointer_rope_exclusion`: `sam2/modeling/memory_attention.py` line(s) 153
- `rope_suffix_exclusion`: `sam2/modeling/sam/transformer.py` line(s) 296
