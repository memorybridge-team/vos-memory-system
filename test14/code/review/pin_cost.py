#!/usr/bin/env python3
"""Per-frame cost of one extra conditioning record: memory attention and mask decoder, 7 records against 8."""
import statistics
import time

import numpy as np
import torch

from train_e2e import Store, Teacher, held_out, bf16


def timed(teacher, view, item, spatial, pointer):
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        teacher.read(view, item, spatial, pointer)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000


def main():
    device = torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(.08)
    teacher = Teacher(device)
    items = [i for i in held_out(Store('dev'), 16) if len(i['frames']) >= 8]
    item = items[0]
    y = bf16(item['y'], device)
    q = item['q'].to(device)
    view = teacher.image(item['image'])
    start = time.perf_counter()
    for _ in range(10):
        teacher.image(item['image'])
    torch.cuda.synchronize()
    encode = (time.perf_counter() - start) * 100
    extra = dict(item)
    extra['frames'] = np.r_[item['frames'], item['switch'] + 1]
    extra['cond'] = np.r_[item['cond'], True]
    plain = item
    y2 = torch.cat((y, y[-1:]))
    q2 = torch.cat((q, q[-1:]))
    a, b = [], []
    for k in range(70):
        a.append(timed(teacher, view, plain, y, q))
        b.append(timed(teacher, view, extra, y2, q2))
    a, b = a[10:], b[10:]
    print(f'records in item {len(item["frames"])}; image encoder {encode:.1f} ms')
    print(f'memory attention + decoder: 7 records {statistics.median(a):.2f} ms, 8 records {statistics.median(b):.2f} ms, extra {statistics.median(b) - statistics.median(a):.2f} ms')
    print('GPU busy with other work, so read the difference rather than the absolute values')


if __name__ == '__main__':
    main()
