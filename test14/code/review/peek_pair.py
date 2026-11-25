#!/usr/bin/env python3
"""What one cached pair stores: prompt, conditioning frame and any Small masks."""
import torch

from vos_memory_inspector.state_schema import CanonicalState, StateSpec

ROOT = '/workspace/cmmt-design-review-20261104/data/'
for split in ('val', 'dev'):
    meta = torch.load(ROOT + split + '/meta.pt', map_location='cpu', weights_only=False)
    row = meta['cases'][0]
    print(split, {k: (v if not hasattr(v, 'shape') else v.shape) for k, v in row.items()})
    with torch.serialization.safe_globals([CanonicalState, StateSpec]):
        value = torch.load(row['path'], map_location='cpu', weights_only=True)
    print(' top keys', list(value))
    print(' metadata', {k: (v if not isinstance(v, (dict, list)) or len(str(v)) < 300 else type(v).__name__ + str(list(v)[:8] if isinstance(v, dict) else len(v))) for k, v in value['metadata'].items()})
    for name in ('source_canonical', 'target_canonical'):
        s = value[name]
        print(' ', name, 'switch', int(s.switch_frame), 'valid', int(s.validity.sum()), 'cond', s.is_conditioning[0, 0].nonzero().flatten().tolist(),
              'frames', s.frame_indices[0, 0][s.validity[0, 0]].tolist())
        print('   metadata keys', {k: (type(v).__name__, (v.shape if hasattr(v, 'shape') else (list(v)[:6] if isinstance(v, dict) else None))) for k, v in s.metadata.items()})
