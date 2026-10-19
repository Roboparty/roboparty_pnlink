"""Native SDK Euler comparisons must not inherit GMR metadata or targets."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from pn_rp1_calibration import PN_JOINT_NAMES, PN_PARENTS, ROBOT_JOINT_NAMES
import view_rp1_gmr_3d as view


class MethodComparisonTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / 'methods.json'
        self.times = np.array([15., 15.02, 15.04])
        self.qpos = np.zeros((3, 35))
        self.qpos[:, :3] = [[1., 2., .9], [1.1, 2., .8], [1.2, 2., .7]]
        self.qpos[:, 3] = 1.
        self.positions = np.random.default_rng(38).normal(size=(3, 21, 3))
        indices = {name: i for i, name in enumerate(PN_JOINT_NAMES)}
        parents = [-1 if PN_PARENTS[name] is None else indices[PN_PARENTS[name]]
                   for name in PN_JOINT_NAMES]
        self.values = dict(time_s=self.times-self.times[0], source_time_s=self.times,
                           qpos=self.qpos, joint_names=np.array(ROBOT_JOINT_NAMES),
                           source_joint_names=np.array(PN_JOINT_NAMES),
                           source_parent_index=np.asarray(parents),
                           source_position_m=self.positions, source_index=np.array([1499, 1501, 1503]))
        self.rows = []
        for i in range(3):
            reference = self.qpos[i].copy()
            reference[2] -= .146
            self.rows.append(dict(mode='live', time_s=2.+i*.02,
                                  source_time_s=float(self.times[i]),
                                  qpos=reference.tolist(), blend_active=False,
                                  alignment_se2=[0., 0., 0., 0., 0., 0.],
                                  reference_pipeline='legacy_euler_reference'))
        self.trial = dict(method='legacy_euler', label='原版欧拉角', core='core.npz',
                          reference='reference.jsonl', display_scale=.78)

    def load(self):
        np.savez(self.root / 'core.npz', **self.values)
        (self.root / 'reference.jsonl').write_text('\n'.join(json.dumps(row) for row in self.rows))
        self.path.write_text(json.dumps({'variants': [self.trial]}))
        return view.load_method_comparisons(self.path)

    def test_load_keeps_native_and_published_height_and_distinct_metadata(self):
        bundles = self.load()
        self.assertEqual(len(bundles), 2)
        core = bundles['原版欧拉角 · 原生retarget（未扣0.146m高度）']
        final = bundles['原版欧拉角 · 原版发布reference（已扣0.146m）']
        np.testing.assert_array_equal(core['replay'].qpos, self.qpos)
        np.testing.assert_allclose(final['replay'].qpos[:, 2], self.qpos[:, 2]-.146)
        self.assertEqual(core['replay'].reference_pipeline, 'legacy_euler_core')
        self.assertEqual(final['replay'].reference_pipeline, 'legacy_euler_reference')
        for bundle in bundles.values():
            replay = bundle['replay']
            self.assertEqual(replay.method, 'legacy_euler')
            self.assertEqual(replay.mapping_profile, 'sdk_euler')
            self.assertEqual(replay.display_pn_scale, .78)
            self.assertIsNone(replay.root_position_scale)
            self.assertIsNone(replay.joint_velocity_rad_s)
            self.assertIsNone(replay.pre_ik_support_enabled)
            self.assertIsNone(replay.ik_floor_enabled)
            self.assertIsNone(replay.torso_pitch_compensation_enabled)
            self.assertIsNone(bundle['target_positions'])
            self.assertEqual(bundle['target_names'], [])
            np.testing.assert_array_equal(bundle['pn'].positions, self.positions)
            self.assertIn('禁用', view.processed_target_hint(replay))
            self.assertIn('不代表 SDK 内部 FSC 缩放', view.processing_note(replay))

    def test_native_core_can_be_loaded_without_published_reference(self):
        self.trial.pop('reference')
        self.assertEqual(len(self.load()), 1)

    def test_legacy_root_scale_metadata_never_rescales_robot_or_display_pn(self):
        self.trial['root_position_scale'] = .86781609
        for bundle in self.load().values():
            self.assertEqual(bundle['replay'].root_position_scale, .86781609)
            self.assertEqual(view.pn_display_scale(bundle['replay'], {}), .78)
        self.trial['root_position_scale'] = False
        with self.assertRaisesRegex(ValueError, 'legacy SDK root scale metadata'):
            self.load()

    def test_rejects_noncanonical_joints_and_missing_original_pn(self):
        original_names = self.values['joint_names']
        self.values['joint_names'] = original_names[::-1]
        with self.assertRaisesRegex(ValueError, 'canonical RP1 order'):
            self.load()
        self.values['joint_names'] = original_names
        self.values.pop('source_position_m')
        with self.assertRaisesRegex(ValueError, 'incomplete original PN'):
            self.load()

    def test_rejects_unknown_method_and_invalid_display_scale(self):
        self.trial['method'] = 'gmr'
        with self.assertRaisesRegex(ValueError, 'unsupported comparison method'):
            self.load()
        self.trial['method'] = 'legacy_euler'
        for invalid in (True, 0., -1., np.nan, '0.78'):
            self.trial['display_scale'] = invalid
            with self.subTest(scale=invalid), self.assertRaisesRegex(ValueError, 'display-only PN scale'):
                self.load()

    def test_rejects_unknown_or_gmr_reference_processing(self):
        self.rows[0]['reference_pipeline'] = 'alignment_transition_only'
        with self.assertRaisesRegex(ValueError, 'unknown processing pipelines'):
            self.load()

    def test_rejects_nonfinite_source_pose_and_malformed_diagnostic_geometry(self):
        qpos = self.qpos.copy()
        qpos[1, 7] = np.nan
        self.values['qpos'] = qpos
        with self.assertRaisesRegex(ValueError, 'nonfinite references'):
            self.load()
        self.values['qpos'] = self.qpos
        self.values['target_names'] = np.array(['base_link'])
        self.values['target_position_m'] = np.full((3, 1, 3), np.nan)
        with self.assertRaisesRegex(ValueError, 'diagnostic target geometry'):
            self.load()
        self.values['target_position_m'] = np.zeros((3, 1, 2))
        with self.assertRaisesRegex(ValueError, 'diagnostic target geometry'):
            self.load()

    def test_legacy_overlay_preserves_pn_without_fabricated_ik_targets(self):
        bundle = next(iter(self.load().values()))
        viewer = view.RP1Viewer.__new__(view.RP1Viewer)
        viewer.active = bundle['replay']
        viewer.show_tpose = viewer.ros_mode = False
        viewer.target_times, viewer.target_positions = self.times, None
        viewer.replay_targets = {viewer.active.name: (self.times, None)}
        viewer.replay_pn = {viewer.active.name: bundle['pn']}
        viewer.pn_names = PN_JOINT_NAMES
        viewer.source_root_ref = np.array([0., 0., 1.])
        viewer.robot_root_ref = np.array([0., 0., .77])
        viewer.pelvis_compare_control = SimpleNamespace(value=False)
        processed, raw, scaled, info = viewer._skeleton_positions(1)
        self.assertIsNone(processed)
        np.testing.assert_array_equal(raw, self.positions[1])
        np.testing.assert_allclose(scaled, viewer.robot_root_ref + .78*(raw-viewer.source_root_ref))
        self.assertEqual(info['source_index'], 1501)
        original = self.positions.copy()
        viewer.pelvis_compare_control.value = True
        _, centered_raw, centered_scaled, _ = viewer._skeleton_positions(1)
        hip = PN_JOINT_NAMES.index('Hips')
        np.testing.assert_allclose(centered_raw[hip], self.qpos[1, :3])
        np.testing.assert_allclose(centered_scaled[hip], self.qpos[1, :3])
        np.testing.assert_array_equal(bundle['pn'].positions, original)


if __name__ == '__main__':
    unittest.main()
