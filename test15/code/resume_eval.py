#!/usr/bin/env python3
"""Base+ resume vs restart without memory (memory-resume-20261120).

Cases and rules are unseen_eval's, as in every earlier PUMaVOS / M3-VOS table: the object starts at its first
GT-visible frame with the GT mask as prompt, switch s = start + fraction * (end - start), and scores are taken on
GT-visible frames after s. Small and Base+ each run from the start. At s the Base+ session ends and a new Base+
session continues from s+1 with one of:

  resume          Base+'s own exported state at s (conditioning + recent records, 16 pointers); must equal Base+ alone
  orig            the original GT prompt only, no memory                          (restart_p0)
  selfmask        Base+'s own last visible mask (frame <= s) only, no memory      (restart_mask)
  orig_selfmask   original prompt + Base+'s own last visible mask, no memory      (restart_p0_mask)
  orig_lastvis    original prompt + Small's last visible mask                     (OLV, as in unseen_eval)
  lastvis         Small's last visible mask only                                  (as in unseen_more)
  anything else   a translated Small state, resolved as in experiments_v5/large_vos_eval.py (paper, strip:..., lin:...)

Per case and switch the output keeps per-frame J, F, GT visibility and predicted area for every method and for Small
and Base+ alone; the last visible frame of Small, Base+ and the GT; how many of each model's recent records and
pointers at s predict the object; and, for each continuation, the records it read at s+1 and whether its
original-prompt record equals Base+'s own one.

  . env.sh && python3 code/resume_eval.py --dataset pumavos --tag pumavos --fractions 0.25,0.5,0.75 \\
      --method resume=resume --method restart_p0=orig ...
"""
import gc
import json
import time

import torch

import large_vos_eval  # noqa: F401  installs the paper / strip: / lin: sources; LIN_RUN becomes the output root
import unseen_eval as live
from jf_eval import assemble

PROMPT_TOKENS = ('orig', 'selfmask', 'orig_selfmask', 'orig_lastvis', 'lastvis')
_score = live.score


def score(video, frame, mask, obj, metrics):
    """unseen_eval.score plus the predicted area as a fraction of the frame (false positives on GT-empty frames)."""
    value = _score(video, frame, mask, obj, metrics)
    if value is None:
        return None
    return value + (float(mask.mean()),)


live.score = score


def present(state, window):
    """Of the latest `window` non-conditioning records in an exported state, how many predict the object."""
    valid = state.validity[0, 0].bool()
    cond = state.is_conditioning[0, 0].bool()
    idx = (valid & ~cond).nonzero().flatten().tolist()
    idx.sort(key=lambda i: int(state.frame_indices[0, 0, i]))
    idx = idx[-window:]
    logits = state.presence_logits[0, 0].float()
    return dict(n=len(idx), present=sum(1 for i in idx if float(logits[i].max()) > 0))


def probe(records, start, target):
    """continuations hook: on the first continued frame, log the records it read and compare the prompt record."""
    cond = target.is_conditioning[0, 0].nonzero().flatten().tolist()
    reference = None
    if len(cond) == 1 and int(target.frame_indices[0, 0, cond[0]]) == start:
        reference = target.spatial_memory[0, 0, cond[0]].float(), target.object_pointer[0, 0, cond[0]].float()

    def after(method, state, position, mask):
        if method in records:
            return
        store = state['output_dict_per_obj'][0]
        past = [f for f in store['non_cond_frame_outputs'] if f < position]
        row = dict(at=position, cond=sorted(store['cond_frame_outputs']),
                   recent6=sum(1 for f in past if f >= position - 6), ptr15=sum(1 for f in past if f >= position - 15))
        out = store['cond_frame_outputs'].get(start)
        if out is not None and reference is not None:
            row['p0_spatial_diff'] = float((out['maskmem_features'][0].float().cpu() - reference[0]).abs().max())
            row['p0_pointer_diff'] = float((out['obj_ptr'][0].float().cpu() - reference[1]).abs().max())
        records[method] = row
    return after


def trace(values, frames):
    rows = [values.get(f) for f in frames]
    return {key: [r[i] if r is not None else None for r in rows] for i, key in enumerate(('J', 'F', 'visible', 'area'))}


def run_case(args, video, name, obj, start, end, path, fractions, methods, sources, small, base, metrics):
    started = time.monotonic()
    points = {}
    for fraction in fractions:
        frame = min(max(start + round(fraction * (end - start)), start + 1), end - 1)
        points[str(round(fraction * 100))] = frame
    small_scores, source_states, small_anchors = live.rollout(small, video, obj, start, end, set(points.values()), metrics)
    base_scores, target_states, base_anchors = live.rollout(base, video, obj, start, end, set(points.values()), metrics)
    prompt = video.labels(start)[0] == obj
    result = dict(dataset=args.dataset, video=name, object=obj, start=start, end=end, switches={})
    for label, switch in points.items():
        s, t = source_states[switch], target_states[switch]
        if not torch.equal(s.validity, t.validity) or not torch.equal(s.frame_indices, t.frame_indices):
            raise RuntimeError('source/target record mismatch')
        lv_s, lv_b = small_anchors[switch], base_anchors[switch]
        first = (start, prompt)
        with torch.inference_mode(), torch.autocast('cuda', enabled=False):
            canonicals = []
            for m, spec in methods.items():
                token = spec[0]
                if token == 'resume':
                    canonicals.append((m, assemble(m, ('target',) * 4, sources, s, t), []))
                elif token == 'orig':
                    canonicals.append((m, None, [first]))
                elif token == 'selfmask':
                    canonicals.append((m, None, [lv_b]))
                elif token == 'orig_selfmask':
                    canonicals.append((m, None, [first] + ([lv_b] if lv_b[0] != start else [])))
                elif token == 'orig_lastvis':
                    canonicals.append((m, None, [first] + ([lv_s] if lv_s[0] != start else [])))
                elif token == 'lastvis':
                    canonicals.append((m, None, [lv_s]))
                else:
                    canonicals.append((m, assemble(m, spec, sources, s, t), []))
        records = {}
        scored = live.continuations(base, video, canonicals, switch, obj, end, metrics, probe(records, start, t))
        frames = list(range(switch + 1, end + 1))
        visible_before = [f for f in range(start, switch + 1) if f in base_scores and base_scores[f][2]]
        row = dict(switch=switch, last_visible_small=lv_s[0], last_visible_base=lv_b[0],
                   last_visible_gt=max(visible_before) if visible_before else None,
                   base_recent6=present(t, 6), base_ptr15=present(t, 15), small_recent6=present(s, 6), small_ptr15=present(s, 15),
                   reads=records, prompts={m: [f for f, _ in masks] for m, _, masks in canonicals},
                   small=live.summarize(small_scores, frames), base=live.summarize(base_scores, frames))
        for method, values in scored.items():
            row[method] = live.summarize(values, frames)
            if methods[method][0] == 'resume':
                diffs = [abs(a - b) for f in frames if f in base_scores for a, b in zip(values[f][:2], base_scores[f][:2])]
                row[method + '_max_abs_diff_vs_base'] = max(diffs) if diffs else 0.0
        row['trace'] = {m: trace(v, frames) for m, v in dict(scored, small=small_scores, base=base_scores).items()}
        result['switches'][label] = row
    result['seconds'] = time.monotonic() - started
    path.write_text(json.dumps(result))
    line = result['switches'][next(iter(points))]
    gc.collect()
    torch.cuda.empty_cache()
    print(name, obj, {k: round(v['JF'], 3) for k, v in line.items() if isinstance(v, dict) and v.get('JF') is not None},
          f"{result['seconds']:.0f}s", flush=True)


live.run_case = run_case

if __name__ == '__main__':
    live.main()
