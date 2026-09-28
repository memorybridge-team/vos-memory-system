# Tensor state inspection

- Root type: `CanonicalState`
- Tensor count: `30`
- Tensor bytes: `24390898`

| Path | Shape | Axes | Dtype | Device | Bytes |
|---|---:|---|---|---|---:|
| `state.spatial_memory` | `[1, 1, 11, 64, 64, 64]` | `['batch', 'object', 'record', 'channel', 'height', 'width']` | `torch.bfloat16` | `cpu` | 5767168 |
| `state.object_pointer` | `[1, 1, 11, 256]` | `['batch', 'object', 'record', 'feature']` | `torch.float32` | `cpu` | 11264 |
| `state.presence_logits` | `[1, 1, 11, 1]` | `['batch', 'object', 'record', 'scalar']` | `torch.float32` | `cpu` | 44 |
| `state.frame_indices` | `[1, 1, 11]` | `['batch', 'object', 'record']` | `torch.int64` | `cpu` | 88 |
| `state.slot_order` | `[1, 1, 11]` | `['batch', 'object', 'record']` | `torch.int64` | `cpu` | 88 |
| `state.is_conditioning` | `[1, 1, 11]` | `['batch', 'object', 'record']` | `torch.bool` | `cpu` | 11 |
| `state.validity` | `[1, 1, 11]` | `['batch', 'object', 'record']` | `torch.bool` | `cpu` | 11 |
| `state.positional_information.records.object=0/record=0[0]` | `[1, 64, 64, 64]` | `[]` | `torch.float32` | `cpu` | 1048576 |
| `state.positional_information.records.object=0/record=1[0]` | `[1, 64, 64, 64]` | `[]` | `torch.float32` | `cpu` | 1048576 |
| `state.positional_information.records.object=0/record=2[0]` | `[1, 64, 64, 64]` | `[]` | `torch.float32` | `cpu` | 1048576 |
| `state.positional_information.records.object=0/record=3[0]` | `[1, 64, 64, 64]` | `[]` | `torch.float32` | `cpu` | 1048576 |
| `state.positional_information.records.object=0/record=4[0]` | `[1, 64, 64, 64]` | `[]` | `torch.float32` | `cpu` | 1048576 |
| `state.positional_information.records.object=0/record=5[0]` | `[1, 64, 64, 64]` | `[]` | `torch.float32` | `cpu` | 1048576 |
| `state.positional_information.records.object=0/record=6[0]` | `[1, 64, 64, 64]` | `[]` | `torch.float32` | `cpu` | 1048576 |
| `state.positional_information.records.object=0/record=7[0]` | `[1, 64, 64, 64]` | `[]` | `torch.float32` | `cpu` | 1048576 |
| `state.positional_information.records.object=0/record=8[0]` | `[1, 64, 64, 64]` | `[]` | `torch.float32` | `cpu` | 1048576 |
| `state.positional_information.records.object=0/record=9[0]` | `[1, 64, 64, 64]` | `[]` | `torch.float32` | `cpu` | 1048576 |
| `state.positional_information.records.object=0/record=10[0]` | `[1, 64, 64, 64]` | `[]` | `torch.float32` | `cpu` | 1048576 |
| `state.metadata.preserved_inputs.mask_inputs_per_obj.0.0` | `[1, 1, 1024, 1024]` | `[]` | `torch.float32` | `cpu` | 4194304 |
| `state.metadata.preserved_pred_masks.object=0/record=0` | `[1, 1, 256, 256]` | `['object_or_batch', 'mask_channel', 'height', 'width']` | `torch.float32` | `cpu` | 262144 |
| `state.metadata.preserved_pred_masks.object=0/record=1` | `[1, 1, 256, 256]` | `['object_or_batch', 'mask_channel', 'height', 'width']` | `torch.float32` | `cpu` | 262144 |
| `state.metadata.preserved_pred_masks.object=0/record=2` | `[1, 1, 256, 256]` | `['object_or_batch', 'mask_channel', 'height', 'width']` | `torch.float32` | `cpu` | 262144 |
| `state.metadata.preserved_pred_masks.object=0/record=3` | `[1, 1, 256, 256]` | `['object_or_batch', 'mask_channel', 'height', 'width']` | `torch.float32` | `cpu` | 262144 |
| `state.metadata.preserved_pred_masks.object=0/record=4` | `[1, 1, 256, 256]` | `['object_or_batch', 'mask_channel', 'height', 'width']` | `torch.float32` | `cpu` | 262144 |
| `state.metadata.preserved_pred_masks.object=0/record=5` | `[1, 1, 256, 256]` | `['object_or_batch', 'mask_channel', 'height', 'width']` | `torch.float32` | `cpu` | 262144 |
| `state.metadata.preserved_pred_masks.object=0/record=6` | `[1, 1, 256, 256]` | `['object_or_batch', 'mask_channel', 'height', 'width']` | `torch.float32` | `cpu` | 262144 |
| `state.metadata.preserved_pred_masks.object=0/record=7` | `[1, 1, 256, 256]` | `['object_or_batch', 'mask_channel', 'height', 'width']` | `torch.float32` | `cpu` | 262144 |
| `state.metadata.preserved_pred_masks.object=0/record=8` | `[1, 1, 256, 256]` | `['object_or_batch', 'mask_channel', 'height', 'width']` | `torch.float32` | `cpu` | 262144 |
| `state.metadata.preserved_pred_masks.object=0/record=9` | `[1, 1, 256, 256]` | `['object_or_batch', 'mask_channel', 'height', 'width']` | `torch.float32` | `cpu` | 262144 |
| `state.metadata.preserved_pred_masks.object=0/record=10` | `[1, 1, 256, 256]` | `['object_or_batch', 'mask_channel', 'height', 'width']` | `torch.float32` | `cpu` | 262144 |
