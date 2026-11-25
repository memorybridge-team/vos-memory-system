#!/usr/bin/env python3
"""Per-case scores of the external datasets from the design review, as one JSON. Reads finished cases only; no model runs."""
import glob
import json

ROOT = '/workspace/cmmt-design-review-20261104/runs/unseen/'
OUT = '/workspace/cmmt-design-review-20261104/runs/analysis/external_cases.json'
TAGS = {'pumavos': ('pumavos', 'pumavos2', 'pumavos3', 'pumavos4', 'pumavos5'), 'm3vos': ('m3vos', 'm3vos2', 'm3vos3', 'm3vos4', 'm3vos5')}


def main():
    out = {}
    for key, tags in TAGS.items():
        table = {}
        for tag in tags:
            for path in glob.glob(f'{ROOT}{tag}/cases/*.json'):
                case = json.load(open(path))
                for name, row in case['switches'].items():
                    entry = table.setdefault(f"{case['video']}|{case['object']}|{name}", dict(video=case['video'], object=case['object'], name=name, switch=row.get('switch'), start=case.get('start'), end=case.get('end'), scores={}, tags={}))
                    for method, score in row.items():
                        if isinstance(score, dict) and 'JF' in score:
                            if method in entry['scores'] and entry['scores'][method]['JF'] is not None and score['JF'] is not None and abs(entry['scores'][method]['JF'] - score['JF']) > 1e-9:
                                entry.setdefault('conflicts', []).append((method, tag, entry['tags'][method], entry['scores'][method]['JF'], score['JF']))
                            entry['scores'].setdefault(method, score)
                            entry['tags'].setdefault(method, tag)
        out[key] = list(table.values())
        print(key, len(table), 'switch cases', sum(1 for v in table.values() if 'conflicts' in v), 'with conflicts')
    json.dump(out, open(OUT, 'w'))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
