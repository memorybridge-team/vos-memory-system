"""Check whether PyTorch's deterministic flag fixes same-Native MPS repetition."""
import json
import sys
import argparse
from pathlib import Path
from types import SimpleNamespace
from compare_multiple_videos import (SAM, HERE, flatten, comparison, seed, torch,
                                     np, Image, settings, SAM2Runner, MEMORY_FIELDS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deterministic', action='store_true')
    args = parser.parse_args()
    (HERE/'reruns').mkdir(exist_ok=True)
    output = HERE/'reruns'/('mps_deterministic_probe.json' if args.deterministic else 'mps_default_probe.json')
    report = {'device': 'mps', 'deterministic_algorithms': args.deterministic,
              'video': 'bmx-bumps', 'frames': 9, 'complete': False}
    try:
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS unavailable')
        torch.use_deterministic_algorithms(args.deterministic)
        torch.set_num_threads(4)
        settings.DEVICE = 'mps'
        settings.USE_BF16 = False
        settings.SAM2_CHECKPOINT_DIR = str(SAM/'checkpoints')
        paths = sorted((SAM/'data/DAVIS/JPEGImages/480p/bmx-bumps').glob('*.jpg'))[:9]
        labels = np.array(Image.open(SAM/'data/DAVIS/Annotations/480p/bmx-bumps/00000.png'))
        video = SimpleNamespace(frame_paths=paths, size=labels.shape)
        seed()
        runner = SAM2Runner('base_plus')
        reference = {}
        rows = []
        for repeat in range(2):
            seed()
            session = runner.start(video)
            try:
                session.configure_benchmark()
                session.add_prompt(0, labels == 1)
                session.encode_prompts()
                for out in session.track(0, 8):
                    current = flatten({'binary_mask': out.mask,
                                       **{k: session.memory_of(out.frame)[k] for k in MEMORY_FIELDS}})
                    if not repeat:
                        reference[out.frame] = current
                    else:
                        rows.append({'frame': out.frame, 'comparison': comparison(reference[out.frame], current)})
                    print(f'deterministic_flag={args.deterministic} native repeat={repeat} frame={out.frame}', flush=True)
            finally:
                session.close()
        report.update(complete=True, comparisons=rows,
                      all_fields_exact=all(r['comparison']['exact'] for r in rows),
                      exact_mask_frames=sum(r['comparison']['fields']['/binary_mask']['exact'] for r in rows))
    except RuntimeError as exc:
        report['runtime_error'] = str(exc)
    output.write_text(json.dumps(report,indent=2)+'\n')
    if not report['complete']:
        sys.exit(1)


if __name__ == '__main__':
    main()
