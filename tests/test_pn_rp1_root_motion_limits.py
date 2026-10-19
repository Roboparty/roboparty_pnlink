"""Root norm bounds and legacy-limit compatibility, without GMR/mesh loads."""

import copy
import math
import unittest

import mink
import mujoco
import numpy as np
from qpsolvers import solve_qp
from scipy.spatial import HalfspaceIntersection
from scipy.spatial.transform import Rotation

from pn_rp1_retarget import DEFAULT_OUTPUT_LIMITS, _make_motion_limit, _root_motion_polytope


def simple_floating_model():
    children = "".join(
        f'<body name="body_{i}"><inertial pos="0 0 0" mass="1" diaginertia=".01 .01 .01"/>'
        f'<joint name="joint_{i}" type="hinge" axis="0 0 1" range="-2 2"/></body>'
        for i in range(28)
    )
    return mujoco.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/><worldbody><body name="base_link">'
        '<freejoint/><inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/>'
        + children + '</body></worldbody></mujoco>'
    )


class RootMotionLimitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = simple_floating_model()
        cls.limits = copy.deepcopy(DEFAULT_OUTPUT_LIMITS)

    def setup_limit(self, orientation=None, *, polytope=False, previous=None):
        configuration = mink.Configuration(self.model)
        pose = self.model.qpos0.copy()
        if orientation is not None:
            pose[3:7] = orientation.as_quat()[[3, 0, 1, 2]]
        # Include a hinge close to its bound, exercising unchanged braking.
        pose[7] = 1.995
        configuration.update(pose)
        limit = _make_motion_limit(mink, self.model, self.limits, root_norm_polytope=polytope)
        if previous is not None:
            limit.previous_world_velocity = np.array(previous, dtype=float)
        return configuration, limit

    def solve_world_velocity(self, orientation, desired, previous, *, polytope, dt=.02):
        configuration, limit = self.setup_limit(orientation, polytope=polytope, previous=previous)
        target = np.asarray(desired, dtype=float).copy()
        target[3:6] = orientation.inv().apply(target[3:6])
        constraint = limit.compute_qp_inequalities(configuration, dt)
        delta = solve_qp(np.eye(34), -target * dt, constraint.G, constraint.h, solver="daqp")
        self.assertIsNotNone(delta, "root motion constraints should admit this bounded history")
        world = delta / dt
        world[3:6] = orientation.apply(world[3:6])
        return world

    def test_all_polytope_vertices_and_random_directions_are_inside_unit_ball(self):
        normals, scale = _root_motion_polytope()
        self.assertEqual(normals.shape, (42, 3))
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1., atol=1e-15)
        self.assertGreater(scale, .93)
        self.assertLess(scale, 1.)
        # Independently intersect the primal planes rather than using the
        # exporter's dual-hull vertex enumeration to verify the guarantee.
        vertices = HalfspaceIntersection(
            np.column_stack([normals, np.full(42, -scale)]), np.zeros(3)
        ).intersections
        self.assertTrue(np.all(np.linalg.norm(vertices, axis=1) <= 1.))
        rng = np.random.default_rng(97)
        directions = rng.normal(size=(1000, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        radial_limits = scale / np.max(directions @ normals.T, axis=1)
        self.assertGreaterEqual(float(radial_limits.min()), scale - 1e-14)
        self.assertLessEqual(float(radial_limits.max()), 1. + 1e-12)

    def test_default_path_retains_exact_legacy_component_and_joint_braking_bounds(self):
        previous = np.linspace(-.05, .05, 34)
        orientation = Rotation.from_euler("xyz", [.2, -.4, .9])
        configuration, explicit = self.setup_limit(orientation, previous=previous)
        implicit = _make_motion_limit(mink, self.model, self.limits)
        implicit.previous_world_velocity = previous.copy()
        actual = implicit.compute_qp_inequalities(configuration, .02)
        explicit_constraint = explicit.compute_qp_inequalities(configuration, .02)
        np.testing.assert_array_equal(actual.G, explicit_constraint.G)
        np.testing.assert_array_equal(actual.h, explicit_constraint.h)
        np.testing.assert_array_equal(actual.G, np.vstack([np.eye(34), -np.eye(34)]))
        local_previous = previous.copy()
        local_previous[3:6] = orientation.inv().apply(previous[3:6])
        lo = np.maximum(-explicit.vmax, local_previous - explicit.amax * .02)
        hi = np.minimum(explicit.vmax, local_previous + explicit.amax * .02)
        q = configuration.q[7:]
        a = explicit.amax[6:]
        lower_distance = np.maximum(q - self.model.jnt_range[1:, 0] - 1e-5, 0.)
        upper_distance = np.maximum(self.model.jnt_range[1:, 1] - q - 1e-5, 0.)
        lo[6:] = np.maximum(lo[6:], -(np.sqrt((a*.02)**2 + 2*a*lower_distance) - a*.02))
        hi[6:] = np.minimum(hi[6:], np.sqrt((a*.02)**2 + 2*a*upper_distance) - a*.02)
        np.testing.assert_array_equal(actual.h, np.r_[hi * .02, -lo * .02])

    def test_polytope_keeps_joint_constraints_identical_to_legacy(self):
        previous = np.linspace(-.05, .05, 34)
        rotation = Rotation.from_euler("xyz", [.3, .7, -.2])
        cfg, legacy = self.setup_limit(rotation, previous=previous)
        _, poly = self.setup_limit(rotation, previous=previous, polytope=True)
        old = legacy.compute_qp_inequalities(cfg, .02)
        new = poly.compute_qp_inequalities(cfg, .02)
        np.testing.assert_array_equal(new.G[:28], old.G[6:34])
        np.testing.assert_array_equal(new.G[28:56], old.G[40:68])
        np.testing.assert_array_equal(new.h[:28], old.h[6:34])
        np.testing.assert_array_equal(new.h[28:56], old.h[40:68])

    def test_random_solved_world_velocity_and_acceleration_respect_norm_caps(self):
        rng = np.random.default_rng(591)
        previous = np.zeros(34)
        previous[:3] = [.1, .1, .1]
        previous[3:6] = [-.1, .05, .02]
        for index in range(40):
            rotation = Rotation.from_rotvec(rng.normal(size=3))
            desired = np.zeros(34)
            desired[:6] = rng.normal(size=6) * 5.
            world = self.solve_world_velocity(rotation, desired, previous, polytope=True)
            acceleration = (world - previous) / .02
            with self.subTest(direction=index):
                self.assertLessEqual(np.linalg.norm(world[:3]), 1.5 + 1e-9)
                self.assertLessEqual(np.linalg.norm(world[3:6]), 3. + 1e-9)
                self.assertLessEqual(np.linalg.norm(acceleration[:3]), 8. + 1e-7)
                self.assertLessEqual(np.linalg.norm(acceleration[3:6]), 20. + 1e-7)

    def test_world_angular_solution_is_independent_of_free_base_yaw_and_less_conservative(self):
        desired, previous = np.zeros(34), np.zeros(34)
        desired[3] = 3.
        solved = []
        for yaw in (0., .37, math.pi / 4, math.pi / 2, 2.6):
            rotation = Rotation.from_euler("z", yaw)
            solved.append(self.solve_world_velocity(rotation, desired, previous, polytope=True))
        for world in solved[1:]:
            np.testing.assert_allclose(world, solved[0], atol=1e-10, rtol=0.)
        legacy_zero = self.solve_world_velocity(Rotation.identity(), desired, previous, polytope=False)
        legacy_yaw = self.solve_world_velocity(Rotation.from_euler("z", math.pi / 4),
                                                desired, previous, polytope=False)
        self.assertGreater(abs(legacy_zero[3] - legacy_yaw[3]), .05)
        self.assertGreater(solved[0][3], legacy_zero[3] * 1.5)


if __name__ == "__main__":
    unittest.main()
