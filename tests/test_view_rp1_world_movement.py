"""Verify world-space paths and camera coverage without a running 3D server."""

import itertools
import unittest

import numpy as np

import view_rp1_gmr_3d as view


def references(positions):
    """Give the geometry helpers complete references with distinct joint values."""
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    qpos = np.zeros((len(positions), 35), dtype=float)
    qpos[:, :3] = positions
    qpos[:, 3] = 1.
    qpos[:, 7:] = np.arange(28, dtype=float) / 100.
    return qpos


class WorldMovementTests(unittest.TestCase):
    def test_ground_path_preserves_absolute_xy_and_uses_ground_height(self):
        qpos = references([[4.5, -7., .8], [-2., 3.25, .3], [-5.5, -1., 1.4]])
        original = qpos.copy()
        points = view.ground_trajectory(qpos)

        self.assertEqual(points.shape, (3, 3))
        self.assertEqual(points.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(points[:, :2], qpos[:, :2])
        np.testing.assert_array_equal(points[:, 2], np.full(3, .006, dtype=np.float32))
        np.testing.assert_array_equal(qpos, original)
        # Changing a drawn path must not rewrite the reference buffer.
        points[0, 0] = 100.
        np.testing.assert_array_equal(qpos, original)

    def test_path_prefix_uses_inclusive_index_and_never_connects_end_to_start(self):
        points = np.array([[3., -4., .006], [2., -4., .006],
                           [2., -1., .006], [-1., -2., .006]], dtype=np.float32)
        original = points.copy()
        for end_index, count in ((None, 3), (-1, 0), (0, 0), (1, 1), (2, 2),
                                 (3, 3), (100, 3)):
            with self.subTest(end_index=end_index):
                segments = view.trajectory_segments(points, end_index=end_index)
                self.assertEqual(segments.shape, (count, 2, 3))
                self.assertTrue(np.isfinite(segments).all())
                np.testing.assert_array_equal(segments[:, 0], points[:count])
                np.testing.assert_array_equal(segments[:, 1], points[1:count + 1])
        np.testing.assert_array_equal(points, original)

    def test_empty_and_single_sample_paths_are_finite_empty_segments(self):
        for count in (0, 1):
            qpos = references([[12., -8., .75]] if count else [])
            original = qpos.copy()
            points = view.ground_trajectory(qpos)
            self.assertEqual(points.shape, (count, 3))
            self.assertEqual(points.dtype, np.dtype(np.float32))
            self.assertTrue(np.isfinite(points).all())
            for end_index in (None, -1, 0, 20):
                segments = view.trajectory_segments(points, end_index=end_index)
                self.assertEqual(segments.shape, (0, 2, 3))
                self.assertTrue(np.isfinite(segments).all())
            np.testing.assert_array_equal(qpos, original)

    def assert_camera_covers_motion(self, qpos, aspect):
        original = qpos.copy()
        position, look_at = map(np.asarray, view.world_camera(qpos, aspect=aspect))
        self.assertEqual(position.shape, (3,))
        self.assertEqual(look_at.shape, (3,))
        self.assertTrue(np.isfinite(position).all())
        self.assertTrue(np.isfinite(look_at).all())
        offset = position - look_at
        self.assertGreater(float(np.linalg.norm(offset)), 0.)
        expected_direction = np.array([1.6, -1.9, 1.05])
        expected_direction /= np.linalg.norm(expected_direction)
        np.testing.assert_allclose(offset / np.linalg.norm(offset), expected_direction,
                                   rtol=1e-10, atol=1e-10)

        forward = -offset / np.linalg.norm(offset)
        right = np.cross(forward, [0., 0., 1.])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        low = qpos[:, :3].min(axis=0)
        high = qpos[:, :3].max(axis=0)
        high[2] += 1.3
        corners = np.asarray(list(itertools.product(*zip(low, high))))
        relative = corners - position
        depth = relative @ forward
        self.assertGreater(float(depth.min()), 0.)
        vertical_limit = np.tan(np.deg2rad(55.) / 2.)
        self.assertLessEqual(float(np.max(np.abs(relative @ right) / depth)),
                             aspect * vertical_limit + 1e-10)
        self.assertLessEqual(float(np.max(np.abs(relative @ up) / depth)),
                             vertical_limit + 1e-10)
        np.testing.assert_array_equal(qpos, original)
        return float(np.linalg.norm(offset))

    def test_camera_fits_extreme_motion_and_headroom_for_viewport_aspects(self):
        qpos = references([[4.5, -7., .15], [-8., 5.75, 1.6],
                           [1., -2., 4.2], [-1.5, .3, .8]])
        for aspect in (.25, .65, 1., 2.4):
            with self.subTest(aspect=aspect):
                self.assert_camera_covers_motion(qpos, aspect)

    def test_narrow_viewport_moves_camera_out_to_preserve_horizontal_coverage(self):
        qpos = references([[-12., 3., .75], [8., 3., .75]])
        narrow_distance = self.assert_camera_covers_motion(qpos, .25)
        square_distance = self.assert_camera_covers_motion(qpos, 1.)
        self.assertGreater(narrow_distance, square_distance)

    def test_stationary_reference_camera_remains_finite_and_fits_headroom(self):
        qpos = references([[-18., 9., .8]])
        self.assert_camera_covers_motion(qpos, 1.)


if __name__ == "__main__":
    unittest.main()
