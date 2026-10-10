"""Validate complete coverage, independent evidence and saved metric/hash consistency."""
import json
import math
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    results = {}
    environments = {}
    input_signatures = {}
    for device in ['cpu', 'mps']:
        folder = HERE/'extended'/device
        env = json.loads((folder/'environment.json').read_text())
        assert env['complete'] and env['model_weights_unchanged']
        environments[device] = env
        counts = {'videos': 0, 'inference_frames': 0, 'frame_pairs': 0,
                  'exact_frame_pairs': 0, 'mask_exact_pairs': 0,
                  'full_state_pairs': 0, 'exact_full_state_pairs': 0,
                  'metadata_equal_state_pairs': 0, 'cached_feature_exact_state_pairs': 0,
                  'exact_restart_preflights': 0}
        for video in ['bmx-bumps', 'camel', 'breakdance']:
            report = json.loads((folder/(video+'.json')).read_text())
            assert report['complete'] and report['all_runs_have_independent_states']
            assert report['frame_count'] == 33 and len(report['runs']) == 6
            signature = {k: report[k] for k in ['input_rgb_sha256', 'input_gt_sha256', 'prompt_sha256', 'original_object_id', 'resolution']}
            if video in input_signatures:
                assert input_signatures[video] == signature
            input_signatures[video] = signature
            native = {r['frame']: r for r in report['runs'][0]['frames']}
            counts['videos'] += 1
            for run in report['runs']:
                assert [r['frame'] for r in run['frames']] == list(range(33))
                assert [r['frame_idx'] for r in run['single_frame_inference_calls']] == list(range(33))
                # Predictor wrapper handles prompt preflight only. Subsequent
                # track_step calls use SAM2Base._encode_new_memory directly.
                assert run['memory_encoder_calls'] == [0]
                assert run['single_frame_inference_calls'][0]['run_mem_encoder'] is False
                assert all(r['run_mem_encoder'] is True for r in run['single_frame_inference_calls'][1:])
                counts['inference_frames'] += 33
                if run['variant'] == 'native_0':
                    continue
                assert run['inference_schedule_matches_native_0']
                for r in run['frames']:
                    for metric in ['gt_j', 'gt_f', 'gt_jf']:
                        assert math.isfinite(r[metric]) and 0 <= r[metric] <= 1
                    assert r['gt_jf'] == (r['gt_j']+r['gt_f'])/2
                    c = r['vs_native_0']
                    assert c['metadata_equal'] and c['array_keys_equal']
                    counts['frame_pairs'] += 1
                    counts['exact_frame_pairs'] += c['exact']
                    counts['mask_exact_pairs'] += c['fields']['/binary_mask']['exact']
                    for key, field in c['fields'].items():
                        assert field['finite']
                        equal_hash = r['output_hashes']['arrays'][key] == native[r['frame']]['output_hashes']['arrays'][key]
                        assert field['exact'] == equal_hash
                        assert field['exact'] == (field['different_elements'] == 0)
                        if field['exact']:
                            assert field['max_abs_error'] == field['mean_abs_error'] == field['rmse'] == 0
                        else:
                            assert field['max_abs_error'] > 0 and field['rmse'] > 0
                    if c['fields']['/binary_mask']['exact']:
                        assert r['mask_iou_vs_native_0'] == 1
                        assert r['gt_jf_difference_vs_native_0'] == 0
                for f, state in run['switch_states'].items():
                    f = int(f)
                    meta = state['hashes']['metadata']
                    assert meta['/num_frames'] == 33
                    assert meta['/obj_ids/0'] == 1
                    assert meta['/obj_id_to_idx/1'] == 0
                    assert meta['/obj_idx_to_id/0'] == 1
                    assert meta['/mask_inputs_per_obj/0/@keys'] == ['0']
                    assert meta['/point_inputs_per_obj/0/@keys'] == []
                    assert meta['/output_dict_per_obj/0/cond_frame_outputs/@keys'] == ['0']
                    retained = sorted(map(int, meta['/output_dict_per_obj/0/non_cond_frame_outputs/@keys']))
                    assert retained == list(range(max(1, f-16), f+1))
                    c = state['vs_native_0']
                    assert c['metadata_equal'] and c['array_keys_equal']
                    assert all(v['finite'] for v in c['fields'].values())
                    ref_hashes = report['runs'][0]['switch_states'][str(f)]['hashes']['arrays']
                    for key, field in c['fields'].items():
                        assert field['exact'] == (state['hashes']['arrays'][key] == ref_hashes[key])
                    counts['full_state_pairs'] += 1
                    counts['exact_full_state_pairs'] += c['exact']
                    counts['metadata_equal_state_pairs'] += c['metadata_equal']
                    counts['cached_feature_exact_state_pairs'] += all(v['exact'] for k,v in c['fields'].items() if k.startswith('/cached_features'))
                audit = run['preflight_restart_audit']
                if audit is not None:
                    assert audit['exact']
                    counts['exact_restart_preflights'] += 1
        assert counts['inference_frames'] == 594 and counts['frame_pairs'] == 495
        assert counts['full_state_pairs'] == 45 and counts['exact_restart_preflights'] == 9
        if device == 'cpu':
            assert counts['exact_frame_pairs'] == 495 and counts['exact_full_state_pairs'] == 45
        results[device] = counts
    for key in ['parameter_sha256', 'checkpoint_sha256', 'predictor_sha256', 'test_script_sha256', 'session_sha256']:
        assert environments['cpu'][key] == environments['mps'][key]
    unique_rgb = {h for sig in input_signatures.values() for h in sig['input_rgb_sha256'].values()}
    result = {'validation_passed': True, 'same_inputs_weights_and_sources_across_devices': True,
              'devices': results, 'unique_rgb_frames': len(unique_rgb), 'unique_videos': len(input_signatures),
              'scope': 'Local FP32 paths; CPU deterministic, MPS default; one object per video'}
    evidence = [HERE/'extended'/device/name for device in ['cpu','mps']
                for name in ['environment.json','bmx-bumps.json','camel.json','breakdance.json']]
    result['evidence_sha256'] = {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence}
    (HERE/'extended/validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
