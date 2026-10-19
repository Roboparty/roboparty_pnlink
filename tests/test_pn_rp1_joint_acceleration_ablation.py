"""Disable hinge acceleration only; retain every independent motion bound."""

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mink
import mujoco
import numpy as np
from qpsolvers import solve_qp

from pn_rp1_retarget import (DEFAULT_CONTACT, DEFAULT_OUTPUT_LIMITS, DEFAULT_SOLVER,
                            PNToRP1Retargeter, _make_motion_limit)
from test_pn_rp1_root_motion_limits import simple_floating_model

HERE = Path(__file__).resolve().parents[1] / "mocapapi_python"
sys.path.insert(0, str(HERE.parent/'tools'))
from evaluate_recording import quality_metrics


class JointAccelerationSwitchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = simple_floating_model()

    def limits(self, enabled=None):
        limits = copy.deepcopy(DEFAULT_OUTPUT_LIMITS)
        limits['joint_velocity_rad_s'] = [14.] * 28
        limits['joint_acceleration_rad_s2'] = [100.] * 28
        if enabled is not None:
            limits['joint_acceleration_enabled'] = enabled
        return limits

    def checker(self, limits, previous=None):
        solver = PNToRP1Retargeter.__new__(PNToRP1Retargeter)
        solver.config = {'output_limits': limits, 'solver': copy.deepcopy(DEFAULT_SOLVER),
                         'contact': copy.deepcopy(DEFAULT_CONTACT)}
        solver.motion_limit = SimpleNamespace(previous_world_velocity=(
            np.zeros(34) if previous is None else previous.copy()))
        solver.joint_lower = self.model.jnt_range[1:, 0]
        solver.joint_upper = self.model.jnt_range[1:, 1]
        solver.joint_names = [f'joint_{i}' for i in range(28)]
        solver.model = self.model
        return solver

    def test_absent_flag_and_true_produce_identical_bounds_in_both_root_modes(self):
        self.assertNotIn('joint_acceleration_enabled', DEFAULT_OUTPUT_LIMITS)
        cfg = mink.Configuration(self.model)
        q = self.model.qpos0.copy()
        q[7] = 1.995
        cfg.update(q)
        for polytope in (False, True):
            implicit = _make_motion_limit(mink, self.model, self.limits(), polytope)
            explicit = _make_motion_limit(mink, self.model, self.limits(True), polytope)
            implicit.previous_world_velocity[:] = np.linspace(-.1, .1, 34)
            explicit.previous_world_velocity[:] = implicit.previous_world_velocity
            first = implicit.compute_qp_inequalities(cfg, .02)
            second = explicit.compute_qp_inequalities(cfg, .02)
            np.testing.assert_array_equal(first.G, second.G)
            np.testing.assert_array_equal(first.h, second.h)

    def test_disabled_allows_velocity_reversal_but_preserves_root_and_joint_velocity(self):
        dt = .02
        previous = np.zeros(34)
        previous[6:] = 14.
        for polytope in (False, True):
            configuration = mink.Configuration(self.model)
            limit = _make_motion_limit(mink, self.model, self.limits(False), polytope)
            limit.previous_world_velocity[:] = previous
            constraints = [limit.compute_qp_inequalities(configuration, dt),
                           mink.ConfigurationLimit(self.model).compute_qp_inequalities(configuration, dt)]
            desired = np.r_[[100.] * 6, [-100.] * 28]
            delta = solve_qp(np.eye(34), -desired*dt,
                             np.vstack([item.G for item in constraints]),
                             np.concatenate([item.h for item in constraints]), solver='daqp')
            self.assertIsNotNone(delta)
            velocity = delta/dt
            np.testing.assert_allclose(velocity[6:], -14., atol=1e-9)
            self.assertGreater(float(np.max(np.abs((velocity[6:]-previous[6:])/dt))), 100.)
            self.assertLessEqual(np.linalg.norm(velocity[:3]), 1.5+1e-9)
            self.assertLessEqual(np.linalg.norm(velocity[3:6]), 3.+1e-9)
            self.assertLessEqual(np.linalg.norm(velocity[:3]/dt), 8.+1e-7)
            self.assertLessEqual(np.linalg.norm(velocity[3:6]/dt), 20.+1e-7)
            configuration.integrate_inplace(velocity, dt)
            self.checker(self.limits(False), previous)._check_motion_result(velocity, dt, configuration.q)
            with self.assertRaisesRegex(RuntimeError, 'joint acceleration'):
                self.checker(self.limits(True), previous)._check_motion_result(velocity, dt, configuration.q)

    def test_disabled_removes_joint_braking_envelope_but_external_angle_limit_remains(self):
        configuration = mink.Configuration(self.model)
        q = self.model.qpos0.copy()
        q[7] = 1.995
        configuration.update(q)
        for polytope in (False, True):
            off = _make_motion_limit(mink, self.model, self.limits(False), polytope)
            on = _make_motion_limit(mink, self.model, self.limits(True), polytope)
            off_bounds = off.compute_qp_inequalities(configuration, .02)
            on_bounds = on.compute_qp_inequalities(configuration, .02)
            hinge_row = 0 if polytope else 6
            self.assertAlmostEqual(off_bounds.h[hinge_row], 14.*.02)
            self.assertLess(on_bounds.h[hinge_row], .005)
            angle = mink.ConfigurationLimit(self.model, gain=.95).compute_qp_inequalities(configuration, .02)
            desired = np.zeros(34)
            desired[6] = 14.
            delta = solve_qp(np.eye(34), -desired*.02,
                             np.vstack([off_bounds.G, angle.G]),
                             np.r_[off_bounds.h, angle.h], solver='daqp')
            self.assertIsNotNone(delta)
            self.assertLessEqual(q[7]+delta[6], 2.+1e-9)
            self.assertGreater(delta[6], 0.)
            root_rows = slice(56, None) if polytope else np.r_[0:6, 34:40]
            np.testing.assert_array_equal(off_bounds.G[root_rows], on_bounds.G[root_rows])
            np.testing.assert_array_equal(off_bounds.h[root_rows], on_bounds.h[root_rows])

    def test_aftercheck_retains_every_other_bound_when_joint_acceleration_is_off(self):
        solver = self.checker(self.limits(False))
        q = self.model.qpos0.copy()
        for index, value, expected in ((6, 14.01, 'joint velocity'),
                                       (0, 1.51, 'root linear velocity'),
                                       (3, 3.01, 'root angular velocity'),
                                       (0, .161, 'root linear acceleration'),
                                       (3, .401, 'root angular acceleration')):
            velocity = np.zeros(34)
            velocity[index] = value
            with self.subTest(bound=expected):
                with self.assertRaisesRegex(RuntimeError, expected):
                    solver._check_motion_result(velocity, .02, q)
        q[7] = 2.001
        with self.assertRaisesRegex(RuntimeError, 'joint limits'):
            solver._check_motion_result(np.zeros(34), .02, q)

    def test_non_boolean_flags_are_rejected_before_configuration_or_qp_use(self):
        for invalid in (0, 1, None, 'false', [], {}):
            limits = self.limits()
            limits['joint_acceleration_enabled'] = invalid
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(ValueError, 'must be boolean'):
                    self.checker(limits)._check_parameters()
                with self.assertRaisesRegex(ValueError, 'must be boolean'):
                    _make_motion_limit(mink, self.model, limits)

    def test_metadata_preserves_configured_values_and_declares_whether_enforced(self):
        outputs = []
        for enabled in (None, True, False):
            with patch('pn_rp1_kinematic_model.export_kinematic_xml', return_value='xml'):
                outputs.append(self.checker(self.limits(enabled)).metadata())
        self.assertEqual(outputs[0], outputs[1])
        self.assertFalse(outputs[2]['joint_acceleration_enabled'])
        self.assertFalse(outputs[2]['joint_acceleration_braking_enabled'])
        self.assertEqual(outputs[2]['acceleration_limits'], [100.]*28)
        self.assertIn('configured hinge values only', outputs[2]['acceleration_limits_semantics'])
        self.assertIn('disabled', outputs[2]['acceleration_limits_semantics'])
        self.assertEqual(outputs[0]['velocity_limits'], outputs[2]['velocity_limits'])
        for key in DEFAULT_OUTPUT_LIMITS:
            if key.startswith('root_'):
                self.assertEqual(outputs[0][key], outputs[2][key])

    def test_evaluator_measures_unbounded_acceleration_without_fabricated_violations(self):
        qpos = np.tile(self.model.qpos0, (4, 1))
        qpos[:, 7] = [0., .2, 0., .2]
        zero = np.zeros((4, 1))
        kwargs = dict(model=self.model, qpos=qpos, position_error=zero,
                      rotation_error=zero, names=['base_link'],
                      sole_height=np.zeros((4, 2)), sole_center=np.zeros((4, 2, 3)),
                      contacts=np.zeros((4, 2), dtype=bool), dt=.02)
        enabled, _, _ = quality_metrics(**kwargs, limits=self.limits(True))
        disabled, _, _ = quality_metrics(**kwargs, limits=self.limits(False))
        self.assertGreater(enabled['output_bound_violations']['values']['joint_acceleration_rad_s2']['violations'], 0)
        self.assertEqual(enabled['joint_acceleration_rad_s2'], disabled['joint_acceleration_rad_s2'])
        self.assertGreater(disabled['joint_acceleration_rad_s2']['max'], 100.)
        bounds = disabled['output_bound_violations']
        self.assertFalse(bounds['joint_acceleration_enabled'])
        self.assertNotIn('joint_acceleration_rad_s2', bounds['values'])
        self.assertIn('not enforced', bounds['disabled_bounds']['joint_acceleration_rad_s2'])
        self.assertTrue(all(item['violations'] == 0 for item in bounds['values'].values()))

    def test_real_retargeter_keeps_floor_and_single_qp_while_flag_is_disabled(self):
        config = json.loads((HERE/'rp1/pn_axis_to_rp1_28dof_current.json').read_text())
        from pn_rp1_paths import resolve_model_path
        config['source_calibration']['robot_model_path'] = str(resolve_model_path(
            configured_path=config['source_calibration']['robot_model_path'],
            config_path=HERE/'rp1/pn_axis_to_rp1_28dof_current.json'))
        config['output_limits']['joint_acceleration_enabled'] = False
        config['output_limits']['joint_acceleration_rad_s2'] = [100.]*28
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'disabled.json'
            path.write_text(json.dumps(config))
            solver = PNToRP1Retargeter(path)
            self.assertFalse(solver.motion_limit.joint_acceleration_enabled)
            self.assertTrue(solver.config['solver']['floor_enabled'])
            self.assertTrue(any(type(limit).__name__ == 'FloorLimit' for limit in solver.ik_limits))
            self.assertTrue(any(isinstance(limit, mink.ConfigurationLimit) for limit in solver.ik_limits))
            for index in range(4):
                solver.retarget(solver.adapter.source_tpose, .02, source_time_s=index*.02)
                self.assertEqual(solver.last_stats['qp_solves'], 1)
                self.assertGreaterEqual(min(solver.sole_min_height().values()), .002-1e-4)
            metadata = solver.metadata()
            self.assertFalse(metadata['joint_acceleration_enabled'])
            self.assertEqual(metadata['acceleration_limits'], [100.]*28)


if __name__ == '__main__':
    unittest.main()
