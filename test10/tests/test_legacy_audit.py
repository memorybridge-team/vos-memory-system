"""Optional historical records must not block an independent evaluation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manifest import audit_legacy


class LegacyAudit(unittest.TestCase):
    def case(self):
        return dict(case_id='new-case', original_case_id='old-case', dataset='LVOSv2',
                    cohort='legacy_dev', changed=False, switch=4, end=20)

    def test_independent_manifest_ignores_unreadable_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy = Path(tmp)
            (legacy/'eval_dev').mkdir()
            (legacy/'eval_dev/results.jsonl').write_text('invalid JSON')
            (legacy/'eval_dev/run_meta_0.json').write_text('invalid JSON')
            with patch('manifest.LEGACY', legacy):
                self.assertEqual(audit_legacy({'cases':[{'case_id':'independent'}]}, {}), [])

    def test_missing_archive_marks_historical_rows_unusable(self):
        with tempfile.TemporaryDirectory() as tmp, patch('manifest.LEGACY', Path(tmp)/'missing'):
            rows = audit_legacy({'cases':[self.case()]}, {})
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(r['status']=='unusable' for r in rows))
            self.assertTrue(all(r['reason']=='missing legacy results file' for r in rows))
            self.assertTrue(all(r['historical_row'] is None for r in rows))

    def test_present_history_remains_diagnostic_and_unverified(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy = Path(tmp)
            (legacy/'eval_dev').mkdir()
            historical = dict(dataset='LVOSv2', case_id='old-case', method='direct',
                              switch_position=4, future_positions=16)
            (legacy/'eval_dev/results.jsonl').write_text(json.dumps(historical)+'\n')
            meta = dict(code_revision={'source_digest':'runtime'}, models={}, autocast='bfloat16')
            (legacy/'eval_dev/run_meta_0.json').write_text(json.dumps(meta))
            prov = dict(legacy_source_digest='runtime', models={}, autocast='bfloat16')
            with patch('manifest.LEGACY', legacy):
                rows = audit_legacy({'cases':[self.case()]}, prov)
            direct = next(r for r in rows if r['method']=='direct')
            self.assertEqual(direct['status'], 'unverified')
            self.assertEqual(direct['historical_row'], historical)
            self.assertIn('hashes were not recorded', direct['reason'])
            self.assertTrue(all(r['status']=='unusable' for r in rows if r['method']!='direct'))


if __name__ == '__main__':
    unittest.main()
