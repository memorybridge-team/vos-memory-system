"""Aggregate completed multivideo audits without dropping individual frames."""
import csv
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent


def main():
    frame_rows, condition_rows, video_rows = [], [], []
    for folder in sorted((HERE/'extended').iterdir()):
        if not folder.is_dir():
            continue
        env = json.loads((folder/'environment.json').read_text())
        if not env.get('complete'):
            continue
        for path in sorted(folder.glob('*.json')):
            if path.name == 'environment.json':
                continue
            report = json.loads(path.read_text())
            if not report.get('complete'):
                continue
            if len(report['runs']) != 6:
                raise AssertionError('Run coverage')
            controls = {r['frame']: r for r in report['runs'][1]['frames']}
            for run in report['runs'][1:]:
                if [r['frame'] for r in run['frames']] != list(range(report['frame_count'])):
                    raise AssertionError('Frame coverage')
                for row in run['frames']:
                    c = row['vs_native_0']
                    mask = c['fields']['/binary_mask']
                    logits = c['fields']['/full_resolution_logits']
                    frame_rows.append({'device': folder.name, 'video': report['video'],
                                       'variant': run['variant'], 'frame': row['frame'],
                                       'all_fields_exact': c['exact'], 'mask_exact': mask['exact'],
                                       'different_pixels': mask['different_elements'],
                                       'mask_iou': row['mask_iou_vs_native_0'],
                                       'full_logits_max_abs_error': logits['max_abs_error'],
                                       'full_logits_mae': logits['mean_abs_error'],
                                       'full_logits_rmse': logits['rmse'],
                                       'gt_j': row['gt_j'], 'gt_f': row['gt_f'], 'gt_jf': row['gt_jf'],
                                       'gt_jf_difference': row['gt_jf_difference_vs_native_0']})
                for sw in report['switches']:
                    if run['switch_frame'] is not None and run['switch_frame'] != sw['frame']:
                        continue
                    rows = [r for r in run['frames'] if r['frame'] > sw['frame']]
                    noise_rows = [controls[r['frame']] for r in rows]
                    mae = mean(r['vs_native_0']['fields']['/full_resolution_logits']['mean_abs_error'] for r in rows)
                    noise_mae = mean(r['vs_native_0']['fields']['/full_resolution_logits']['mean_abs_error'] for r in noise_rows)
                    condition_rows.append({'device': folder.name, 'video': report['video'],
                                           'variant': run['variant'], 'switch_percent': int(sw['name']),
                                           'future_frames': len(rows),
                                           'exact_mask_frames': sum(r['vs_native_0']['fields']['/binary_mask']['exact'] for r in rows),
                                           'all_fields_exact_frames': sum(r['vs_native_0']['exact'] for r in rows),
                                           'max_different_pixels': max(r['vs_native_0']['fields']['/binary_mask']['different_elements'] for r in rows),
                                           'mean_mask_iou': mean(r['mask_iou_vs_native_0'] for r in rows),
                                           'min_mask_iou': min(r['mask_iou_vs_native_0'] for r in rows),
                                           'mean_full_logits_mae': mae,
                                           'native_repeat_mean_full_logits_mae': noise_mae,
                                           'mae_ratio_to_native_repeat': mae/noise_mae if noise_mae else None,
                                           'mean_gt_jf_difference_pp': 100*mean(r['gt_jf_difference_vs_native_0'] for r in rows),
                                           'max_abs_frame_gt_jf_difference_pp': 100*max(abs(r['gt_jf_difference_vs_native_0']) for r in rows)})
            audits = [run['preflight_restart_audit'] for run in report['runs'] if run['preflight_restart_audit'] is not None]
            states = [state['vs_native_0'] for run in report['runs'][1:] for state in run['switch_states'].values()]
            compared = [row['vs_native_0'] for run in report['runs'][1:] for row in run['frames']]
            video_rows.append({'device': folder.name, 'video': report['video'],
                               'runs': len(report['runs']), 'inference_frames': sum(len(r['frames']) for r in report['runs']),
                               'frame_pairs_vs_native_0': len(compared),
                               'all_fields_exact_pairs': sum(c['exact'] for c in compared),
                               'full_state_pairs': len(states), 'exact_full_state_pairs': sum(c['exact'] for c in states),
                               'state_metadata_equal_pairs': sum(c['metadata_equal'] for c in states),
                               'restart_preflight_audits': len(audits), 'exact_idempotent_preflights': sum(c['exact'] for c in audits),
                               'all_inference_schedules_match': all(r['inference_schedule_matches_native_0'] for r in report['runs'][1:]),
                               'all_states_independent': report['all_runs_have_independent_states'],
                               'model_weights_unchanged': env['model_weights_unchanged'],
                               'first_native_repeat_any_difference': next((r['frame'] for r in report['runs'][1]['frames'] if not r['vs_native_0']['exact']), None)})
    for name, rows in [('frame_comparisons',frame_rows), ('condition_summary',condition_rows), ('video_audit_summary',video_rows)]:
        if rows:
            with (HERE/'extended'/(name+'.csv')).open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    (HERE/'extended/summary.json').write_text(json.dumps({'videos': video_rows, 'conditions': condition_rows},indent=2)+'\n')
    print(json.dumps(video_rows,indent=2))


if __name__ == '__main__':
    main()
