#!/usr/bin/env python3
"""Convert the v2 pair caches into flat record arrays for fast random access.

Original caches are only read. Splits: fit, dev (the repository's development cases, kept
out of training here so MOSE stays measurable) and val (official LVOS validation).
"""
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from vos_memory_inspector.state_schema import CanonicalState, StateSpec

ROOT = Path(os.environ['REVIEW_ROOT'])
OUT = ROOT / 'data'
INDEX = Path('/workspace/j700qtcgrg/CMMT-official-isolated/official-20261005T012256KST/index/raw_index.json')
OLD, NEW = '/workspace/CMMT-task07-artifacts/production', '/workspace/j700qtcgrg/CMMT-task07-artifacts/production'


def split_of(case):
    if case['training_role'] == 'validation':
        return 'val'
    return 'fit' if case.get('origin_paired_split') == 'fit' else 'dev'


def open_arrays(split, count, mode):
    folder = OUT / split
    return (np.memmap(folder / 'src_sp.i16', dtype=np.int16, mode=mode, shape=(count, 64, 64, 64)),
            np.memmap(folder / 'tgt_sp.i16', dtype=np.int16, mode=mode, shape=(count, 64, 64, 64)),
            np.memmap(folder / 'src_p.f32', dtype=np.float32, mode=mode, shape=(count, 256)),
            np.memmap(folder / 'tgt_p.f32', dtype=np.float32, mode=mode, shape=(count, 256)))


def worker(job):
    split, count, offset, path, expected = job
    torch.set_num_threads(1)
    with torch.serialization.safe_globals([CanonicalState, StateSpec]):
        value = torch.load(path, map_location='cpu', weights_only=True)
    s, t = value['source_canonical'], value['target_canonical']
    valid = s.validity[0, 0]
    if not torch.equal(valid, t.validity[0, 0]) or int(valid.sum()) != expected:
        raise ValueError(f'validity mismatch {path}')
    arrays = open_arrays(split, count, 'r+')
    index = valid.nonzero().flatten()
    arrays[0][offset:offset + expected] = s.spatial_memory[0, 0, index].view(torch.int16).numpy()
    arrays[1][offset:offset + expected] = t.spatial_memory[0, 0, index].view(torch.int16).numpy()
    arrays[2][offset:offset + expected] = s.object_pointer[0, 0, index].numpy()
    arrays[3][offset:offset + expected] = t.object_pointer[0, 0, index].numpy()
    return dict(frame=s.frame_indices[0, 0, index].numpy().astype(np.int32),
                cond=s.is_conditioning[0, 0, index].numpy().astype(np.bool_),
                src_pres=s.presence_logits[0, 0, index, 0].numpy().astype(np.float32),
                tgt_pres=t.presence_logits[0, 0, index, 0].numpy().astype(np.float32),
                switch=int(s.switch_frame), offset=offset)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = json.loads(INDEX.read_text())
    entries = {'fit': [], 'dev': [], 'val': []}
    for entry in raw['entries']:
        entries[split_of(entry['case'])].append(entry)
    (OUT / 'normalization.json').write_text(json.dumps(raw['normalization']))
    pool = mp.get_context('spawn').Pool(int(os.environ.get('BUILD_WORKERS', 16)))
    for split, rows in entries.items():
        folder = OUT / split
        folder.mkdir(exist_ok=True)
        if (folder / 'meta.pt').exists():
            print(split, 'exists', flush=True)
            continue
        count = sum(e['records'] for e in rows)
        print(split, 'cases', len(rows), 'records', count, flush=True)
        for name, unit in (('src_sp.i16', 524288), ('tgt_sp.i16', 524288), ('src_p.f32', 1024), ('tgt_p.f32', 1024)):
            with open(folder / name, 'wb') as stream:
                stream.truncate(count * unit)
        jobs, offset = [], 0
        for e in rows:
            path = e['path'].replace(OLD, NEW)
            jobs.append((split, count, offset, path, e['records']))
            offset += e['records']
        started = time.time()
        results = []
        for i, item in enumerate(pool.imap(worker, jobs, chunksize=8), 1):
            results.append(item)
            if i % 2000 == 0 or i == len(jobs):
                print(f'{split} {i}/{len(jobs)} {time.time() - started:.0f}s', flush=True)
        case_id = np.zeros(count, dtype=np.int32)
        position = np.zeros(count, dtype=np.int16)
        frame = np.zeros(count, dtype=np.int32)
        switch = np.zeros(count, dtype=np.int32)
        cond = np.zeros(count, dtype=np.bool_)
        src_pres = np.zeros(count, dtype=np.float32)
        tgt_pres = np.zeros(count, dtype=np.float32)
        for c, item in enumerate(results):
            a, n = item['offset'], len(item['frame'])
            case_id[a:a + n] = c
            position[a:a + n] = np.arange(n)
            frame[a:a + n] = item['frame']
            switch[a:a + n] = item['switch']
            cond[a:a + n] = item['cond']
            src_pres[a:a + n] = item['src_pres']
            tgt_pres[a:a + n] = item['tgt_pres']
        cases = [dict(case_id=e['case']['case_id'], dataset=e['case']['dataset'], video_id=e['case']['video_id'],
                      object_id=e['case']['object_id'], switch_frame=e['case']['switch_frame'],
                      runtime_switch=e['runtime_switch_frame'], path=e['path'].replace(OLD, NEW),
                      origin=e['case'].get('origin_paired_split'), records=e['records']) for e in rows]
        torch.save(dict(count=count, case_id=torch.from_numpy(case_id), position=torch.from_numpy(position),
                        frame=torch.from_numpy(frame), switch=torch.from_numpy(switch), cond=torch.from_numpy(cond),
                        src_pres=torch.from_numpy(src_pres), tgt_pres=torch.from_numpy(tgt_pres), cases=cases),
                   folder / 'meta.pt')
        print(split, 'done', f'{time.time() - started:.0f}s', flush=True)
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
