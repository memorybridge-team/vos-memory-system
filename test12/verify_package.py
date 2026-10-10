"""Verify published evidence and the original checkpoint-backed source snapshots."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    validation = json.loads((HERE/'extended/validation.json').read_text())
    assert validation['validation_passed']
    for relative, expected in validation['evidence_sha256'].items():
        assert digest(HERE/relative) == expected, relative
    manifest = json.loads((HERE/'provenance/source_manifest.json').read_text())
    for relative, expected in manifest['recorded_results_sha256'].items():
        assert digest(HERE/relative) == expected, relative
    snapshots = {row['original_path']: row for row in manifest['source_snapshots']}
    for row in snapshots.values():
        assert digest(HERE/row['snapshot_path']) == row['sha256'], row['snapshot_path']
    for name, expected in manifest['original_scripts'].items():
        assert digest(HERE/'provenance/original_scripts'/name) == expected, name
    config = json.loads((HERE/'extended/configuration_manifest.json').read_text())
    for original, expected in config['source_files'].items():
        assert snapshots[original]['sha256'] == expected, original
    for device in ['cpu', 'mps']:
        env = json.loads((HERE/'extended'/device/'environment.json').read_text())
        for key, relative in [('test_script_sha256', 'provenance/original_scripts/compare_multiple_videos.py'),
                              ('session_sha256', 'runtime/model/sam2_runner.py'),
                              ('full_replay_sha256', 'runtime/baseline/no_handoff.py'),
                              ('predictor_sha256', 'provenance/sam2/sam2/sam2_video_predictor.py')]:
            assert digest(HERE/relative) == env[key], (device, key)
    for name in ['results_cpu.json', 'results_mps.json', 'control_mps.json']:
        result = json.loads((HERE/name).read_text())
        assert digest(HERE/'fixtures/fixture_cpu.npz') == result['prompt_fixture_sha256'], name
    print('PASS: 8 raw evidence files, source snapshots, original scripts and prompt fixture')

if __name__ == '__main__':
    main()
