"""Distinguish direct publication logs from historical constrained trials."""
import json
from pathlib import Path
import tempfile
import unittest

import view_rp1_gmr_3d as view


class DirectReferenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / 'reference.jsonl'
        self.rows = [dict(mode='live', source_time_s=i*.02, time_s=10.+i*.02,
                          qpos=[0., 0., .78, 1., 0., 0., 0.]+[.2*i]*28,
                          alignment_se2=[0.]*6, blend_active=i == 0,
                          reference_pipeline='alignment_transition_only',
                          final_constraints_enabled=False) for i in range(3)]
        for row in self.rows:
            row['desired_qpos'] = row['qpos'].copy()

    def load(self):
        self.path.write_text('\n'.join(json.dumps(row) for row in self.rows)+'\n')
        return view.load_final(self.path)

    def test_direct_log_preserves_all_poses_and_processing_metadata(self):
        replay = self.load()
        self.assertEqual(replay.reference_pipeline, 'alignment_transition_only')
        self.assertEqual(replay.qpos.tolist(), [row['qpos'] for row in self.rows])

    def test_direct_log_cannot_claim_constraints_or_change_proposal(self):
        self.rows[1]['final_constraints_enabled'] = True
        with self.assertRaises(ValueError):
            self.load()
        self.rows[1]['final_constraints_enabled'] = False
        self.rows[1]['desired_qpos'][7] += .01
        with self.assertRaises(ValueError):
            self.load()

    def test_mixed_or_unknown_pipeline_is_rejected(self):
        for pipeline in ('constrained_qp', 'unknown'):
            self.rows[1]['reference_pipeline'] = pipeline
            with self.assertRaises(ValueError):
                self.load()

    def test_historical_log_defaults_to_constrained_trial(self):
        for row in self.rows:
            row.pop('reference_pipeline')
            row.pop('final_constraints_enabled')
            row.pop('desired_qpos')
        self.assertEqual(self.load().reference_pipeline, 'constrained_qp')


if __name__ == '__main__':
    unittest.main()
