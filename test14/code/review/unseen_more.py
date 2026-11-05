#!/usr/bin/env python3
"""More live compositions on PUMaVOS/M3-VOS: same cases and rules as unseen_eval.

Extra tokens: orig (prompt only), lastvis (Small's last visible mask only),
recent:<src> (recent records from <src> + last visible mask, no prompt record) and
pin:<k>:<src> (full translation from <src>; once Base+ has run k frames past the switch, its own record
also becomes a conditioning record, as in more_eval).
"""
import gc
import json
import time

import torch

import unseen_eval as live
from handoff_eval import subset


def pinner(rules, switch):
    def after(method, state, position, mask):
        rule = rules.get(method)
        if rule is None or rule['done'] or position < switch + rule['k'] or not mask.any():
            return
        store = state['output_dict_per_obj'][0]
        record = store['non_cond_frame_outputs'].get(position)
        if record is None or float(record['object_score_logits'].max()) <= 0:
            return
        store['cond_frame_outputs'][position] = store['non_cond_frame_outputs'].pop(position)
        rule.update(done=True, frame=position)
    return after


def run_case(args, video, name, obj, start, end, path, fractions, methods, sources, small, base, metrics):
    started = time.monotonic()
    points = {}
    for fraction in fractions:
        frame = min(max(start + round(fraction * (end - start)), start + 1), end - 1)
        points[str(round(fraction * 100))] = frame
    small_scores, source_states, anchors = live.rollout(small, video, obj, start, end, set(points.values()), metrics)
    base_scores, target_states, _ = live.rollout(base, video, obj, start, end, set(points.values()), metrics)
    result = dict(dataset=args.dataset, video=name, object=obj, start=start, end=end, switches={})
    prompt = video.labels(start)[0] == obj
    for label, switch in points.items():
        s, t = source_states[switch], target_states[switch]
        if not torch.equal(s.validity, t.validity) or not torch.equal(s.frame_indices, t.frame_indices):
            raise RuntimeError('source/target record mismatch')
        last = anchors[switch]
        with torch.inference_mode(), torch.autocast('cuda', enabled=False):
            canonicals, rules = [], {}
            valid, cond = s.validity[0, 0], s.is_conditioning[0, 0]
            for m, spec in methods.items():
                token = spec[0]
                if token == 'orig':
                    canonicals.append((m, None, [(start, prompt)]))
                elif token == 'lastvis':
                    canonicals.append((m, None, [last]))
                elif token == 'orig_lastvis':
                    canonicals.append((m, None, [(start, prompt)] + ([last] if last[0] != start else [])))
                elif token.startswith('pin:'):
                    _, k, source = token.split(':', 2)
                    canonicals.append((m, live.assemble(m, (source,) * 4, sources, s, t), []))
                    rules[m] = dict(k=int(k), done=False, frame=None)
                elif token.startswith('recent:'):
                    keep = (valid & ~cond).nonzero().flatten().tolist()
                    canonicals.append((m, subset(live.assemble(m, (token[7:],) * 4, sources, s, t), keep), [last]))
                else:
                    canonicals.append((m, live.assemble(m, spec, sources, s, t), []))
        scored = live.continuations(base, video, canonicals, switch, obj, end, metrics, pinner(rules, switch))
        frames = list(range(switch + 1, end + 1))
        row = dict(switch=switch, last_visible=last[0], small=live.summarize(small_scores, frames), base=live.summarize(base_scores, frames),
                   pinned={m: r['frame'] for m, r in rules.items()})
        first30 = frames[:30]
        row['small_first30'] = live.summarize(small_scores, first30)
        row['base_first30'] = live.summarize(base_scores, first30)
        for method, values in scored.items():
            row[method] = live.summarize(values, frames)
            row[method + '_first30'] = live.summarize(values, first30)
        result['switches'][label] = row
    result['seconds'] = time.monotonic() - started
    path.write_text(json.dumps(result))
    line = result['switches'][next(iter(points))]
    gc.collect()
    torch.cuda.empty_cache()
    print(name, obj, {k: round(v['JF'], 3) for k, v in line.items() if isinstance(v, dict) and v.get('JF') is not None and 'first30' not in k}, f"{result['seconds']:.0f}s", flush=True)


live.run_case = run_case
if __name__ == '__main__':
    live.main()
