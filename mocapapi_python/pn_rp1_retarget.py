#!/usr/bin/env python3
"""PN -> RP1.3.2, GMR task construction and bounded single-step mink IK.

No ROS/device imports. The worker protocol is newline-delimited JSON, with
diagnostics on stderr. The PN adapter owns standard-T-pose and segment scaling;
its final body targets must not pass through GMR offsets a second time.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
from functools import lru_cache
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from pn_rp1_calibration import DEFAULT_CONFIG, PNSourceAdapter, ROBOT_JOINT_NAMES
from pn_rp1_paths import default_gmr_root, resolve_gmr_root, resolve_model_path

GMR_ROOT = default_gmr_root()
IDLE = np.array([
    0., 0., .78, 1., 0., 0., 0.,
    -.1, 0., 0., -.3, -.2, 0., .1, 0., 0., .3, -.2, 0., 0., 0.,
    .2, .2, 0., -1.2, 0., 0., 0., -.2, -.2, 0., 1.2, 0., 0., 0.,
])
DEFAULT_OUTPUT_LIMITS = {
    'joint_velocity_rad_s': [4.] * 28,
    'joint_acceleration_rad_s2': [40.] * 28,
    'root_linear_velocity_limit_m_s': 1.5,
    'root_linear_acceleration_limit_m_s2': 8.,
    'root_angular_velocity_limit_rad_s': 3.,
    'root_angular_acceleration_limit_rad_s2': 20.,
}
DEFAULT_SOLVER = {
    'backend': 'daqp', 'damping': .01, 'posture_cost': .01,
    'knee_posture_cost': 3., 'stance_root_position_cost': 8.,
    'task_gain': .7, 'floor_enabled': True, 'floor_clearance_m': .002,
    'floor_approach_gain': .25, 'floor_braking_acceleration_m_s2': 3.,
    'floor_braking_margin_m': .003,
    'collision_enabled': False,
    'reference_task_tracking': False,
}
DEFAULT_CONTACT = {
    'enabled': True, 'enter_height_m': .035, 'exit_height_m': .075,
    'enter_speed_m_s': .20, 'exit_speed_m_s': .60,
    'anchor_release_distance_m': .065, 'enter_frames': 3,
    'exit_frames': 2, 'source_floor_height_m': 0.,
    'position_cost': 300., 'orientation_cost': 30.,
}


def _rotation(q):
    return Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]])


def _wxyz(rotation):
    return rotation.as_quat()[[3, 0, 1, 2]]


def pn_knee_flexion(poses):
    """Unsigned anatomical flexion from PN thigh/shin world directions.

    Geometric flexion avoids Euler angle discontinuities around 90 degrees.
    RP1's mirrored mechanical axes map it to left negative/right positive.
    These are soft IK posture targets, never edits to solved output angles.
    """
    result = {}
    for side in ('Left', 'Right'):
        hip, knee, ankle = [np.asarray(poses[side + joint][0], dtype=float)
                            for joint in ('UpLeg', 'Leg', 'Foot')]
        thigh, shin = knee - hip, ankle - knee
        if (not np.isfinite(thigh).all() or not np.isfinite(shin).all()
                or np.linalg.norm(thigh) < .005 or np.linalg.norm(shin) < .005):
            raise ValueError(f'PN {side} knee flexion requires finite noncollapsed leg segments')
        result[side.lower()] = math.atan2(float(np.linalg.norm(np.cross(thigh, shin))),
                                         float(np.dot(thigh, shin)))
    return result


def _validated_qpos(qpos, lower, upper):
    q = np.asarray(qpos, dtype=np.float64).copy()
    if q.shape != (35,) or not np.all(np.isfinite(q)):
        raise ValueError('RP1 reference must contain 35 finite values')
    norm = np.linalg.norm(q[3:7])
    if norm < 1e-8:
        raise ValueError('RP1 root quaternion cannot be zero')
    q[3:7] /= norm
    if np.any(q[7:] < lower - 1e-8) or np.any(q[7:] > upper + 1e-8):
        raise ValueError('RP1 initial reference is outside model joint limits')
    return q


@lru_cache(maxsize=1)
def _root_motion_polytope():
    """42 symmetric world normals and a certified inscribed-ball scale.

    The normals are the vertices of a once-subdivided icosahedron. Their
    convex hull facets enumerate vertices of the dual halfspace polytope
    ``normal @ velocity <= 1``. Shrinking by its maximum vertex norm places
    the entire convex feasible polytope inside the unit Euclidean ball.
    A small relative numerical margin makes boundary rounding conservative.
    """
    from scipy.spatial import ConvexHull

    phi = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = np.array([
        vector
        for first in (-1.0, 1.0)
        for second in (-phi, phi)
        for vector in ((0.0, first, second), (first, second, 0.0), (second, 0.0, first))
    ])
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    edges = set()
    for triangle in ConvexHull(vertices).simplices:
        for a, b in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            edges.add(tuple(sorted((int(a), int(b)))))
    midpoints = np.array([vertices[a] + vertices[b] for a, b in sorted(edges)])
    midpoints /= np.linalg.norm(midpoints, axis=1, keepdims=True)
    normals = np.vstack([vertices, midpoints])
    facets = ConvexHull(normals).simplices
    dual_vertices = np.linalg.solve(normals[facets], np.ones((len(facets), 3, 1)))[..., 0]
    max_radius = float(np.linalg.norm(dual_vertices, axis=1).max())
    scale = (1.0 - 1e-12) / max_radius
    normals.setflags(write=False)
    return normals, scale


def joint_acceleration_enabled(limits):
    """Missing flags retain the historical hinge acceleration/braking bounds."""
    enabled = limits.get('joint_acceleration_enabled', True)
    if not isinstance(enabled, bool):
        raise ValueError('joint_acceleration_enabled must be boolean')
    return enabled


def _make_motion_limit(mink, model, limits, root_norm_polytope=False):
    """Constrain displacement and acceleration in the actual output period.

    By default, root component bounds use norm_limit / sqrt(3), preserving
    the core's original constraints exactly. Optional world-frame inscribed
    root polytopes give a closer Euclidean velocity/acceleration bound for the
    final-reference projector. Optional hinge acceleration disabling also
    removes its acceleration-dependent stopping-distance envelope. Joint
    angle/velocity limits and every root bound remain active.
    Angular history is in world coordinates; QP increments use the current
    free-joint local tangent.
    """
    class MotionLimit(mink.Limit):
        def __init__(self):
            self.joint_acceleration_enabled = joint_acceleration_enabled(limits)
            self.previous_world_velocity = np.zeros(model.nv)
            self.eye = np.eye(model.nv)
            self.vmax = np.r_[
                np.full(3, limits['root_linear_velocity_limit_m_s'] / math.sqrt(3)),
                np.full(3, limits['root_angular_velocity_limit_rad_s'] / math.sqrt(3)),
                limits['joint_velocity_rad_s'],
            ]
            self.amax = np.r_[
                np.full(3, limits['root_linear_acceleration_limit_m_s2'] / math.sqrt(3)),
                np.full(3, limits['root_angular_acceleration_limit_rad_s2'] / math.sqrt(3)),
                limits['joint_acceleration_rad_s2'],
            ]
            if root_norm_polytope:
                self.root_normals, self.root_norm_scale = _root_motion_polytope()

        def compute_qp_inequalities(self, configuration, dt):
            previous = self.previous_world_velocity.copy()
            previous[3:6] = _rotation(configuration.q[3:7]).inv().apply(previous[3:6])
            lower = np.maximum(-self.vmax, previous - self.amax * dt)
            upper = np.minimum(self.vmax, previous + self.amax * dt)
            # Keep the next state inside the braking envelope. Merely clipping
            # the next joint angle permits a fast approach that cannot stop
            # inside the range under the following acceleration constraint.
            if self.joint_acceleration_enabled:
                joint_q = configuration.q[7:]
                joint_a = self.amax[6:]
                distance_lower = np.maximum(joint_q - model.jnt_range[1:, 0] - 1e-5, 0.)
                distance_upper = np.maximum(model.jnt_range[1:, 1] - joint_q - 1e-5, 0.)
                brake_lower = np.sqrt((joint_a * dt) ** 2 + 2 * joint_a * distance_lower) - joint_a * dt
                brake_upper = np.sqrt((joint_a * dt) ** 2 + 2 * joint_a * distance_upper) - joint_a * dt
                lower[6:] = np.maximum(lower[6:], -brake_lower)
                upper[6:] = np.minimum(upper[6:], brake_upper)
            else:
                lower[6:] = -self.vmax[6:]
                upper[6:] = self.vmax[6:]
            if not root_norm_polytope:
                return mink.Constraint(G=np.vstack([self.eye, -self.eye]),
                                       h=np.r_[upper * dt, -lower * dt])
            # In this branch root bounds are whole-vector world-frame bounds.
            # Every normal includes its negative. Angular rows transform a
            # local free-joint increment into world before applying normals.
            joint_rows = self.eye[6:]
            rows = [joint_rows, -joint_rows]
            bounds = [upper[6:] * dt, -lower[6:] * dt]
            rotation = _rotation(configuration.q[3:7]).as_matrix()
            for coordinate, kind, units, units_a, transform in (
                (slice(0, 3), 'root_linear', 'm_s', 'm_s2', np.eye(3)),
                (slice(3, 6), 'root_angular', 'rad_s', 'rad_s2', rotation),
            ):
                normals = self.root_normals
                block = np.zeros((len(normals), model.nv))
                block[:, coordinate] = normals @ transform
                max_speed = limits[f'{kind}_velocity_limit_{units}'] * self.root_norm_scale
                max_acceleration = limits[f'{kind}_acceleration_limit_{units_a}'] * self.root_norm_scale
                rows.extend([block, block.copy()])
                bounds.extend([
                    np.full(len(normals), max_speed * dt),
                    max_acceleration * dt * dt
                    + (normals @ self.previous_world_velocity[coordinate]) * dt,
                ])
            return mink.Constraint(G=np.vstack(rows), h=np.concatenate(bounds))
    return MotionLimit()


def _foot_geoms(model, mujoco):
    result = {}
    for side in ('left', 'right'):
        result[side] = [
            i for i in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or '').startswith(side + '_foot_')
            and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CAPSULE
        ]
        if not result[side]:
            raise ValueError(f'RP1 model has no {side} foot collision capsules')
    return result


def _make_floor_limit(mink, mujoco, model, geoms, options):
    class FloorLimit(mink.Limit):
        def compute_qp_inequalities(self, configuration, dt):
            rows, bounds = [], []
            data = configuration.data
            for ids in geoms.values():
                for geom in ids:
                    axis = data.geom_xmat[geom].reshape(3, 3)[:, 2]
                    half_length = model.geom_size[geom, 1]
                    radius = model.geom_size[geom, 0]
                    # Both endpoints are constrained: a formerly horizontal
                    # capsule can tilt during the next integration step.
                    for sign in (-1., 1.):
                        endpoint = data.geom_xpos[geom] + sign * half_length * axis
                        jac = np.zeros((3, model.nv))
                        mujoco.mj_jac(model, data, jac, None, endpoint, model.geom_bodyid[geom])
                        rows.append(-jac[2])
                        distance = endpoint[2] - radius - options['floor_clearance_m']
                        available = max(distance - options['floor_braking_margin_m'], 0.)
                        acceleration = options['floor_braking_acceleration_m_s2']
                        safe_speed = math.sqrt((acceleration * dt) ** 2 + 2 * acceleration * available) - acceleration * dt
                        # A one-step nonpenetration bound alone can leave the
                        # foot too close to stop on the next acceleration-
                        # constrained step. Approach slowly and reserve braking
                        # space, including millimetres for FK linearization.
                        damped_distance = distance * options['floor_approach_gain'] if distance >= 0 else distance
                        bounds.append(min(damped_distance, safe_speed * dt))
            return mink.Constraint(G=np.asarray(rows), h=np.asarray(bounds))
    return FloorLimit()


class PNToRP1Retargeter:
    """Persistent, causal 50 Hz retargeter; one constrained QP per frame."""
    def __init__(self, config_path=DEFAULT_CONFIG, *, gmr_root=None, model_path=None):
        self.config_path = Path(config_path).expanduser().resolve()
        self.config = json.loads(self.config_path.read_text())
        for key, defaults in [('output_limits', DEFAULT_OUTPUT_LIMITS),
                              ('solver', DEFAULT_SOLVER), ('contact', DEFAULT_CONTACT)]:
            self.config[key] = {**copy.deepcopy(defaults), **self.config.get(key, {})}
        self._check_parameters()
        calibration = self.config['source_calibration']
        self.model_path = resolve_model_path(model_path,
            configured_path=calibration['robot_model_path'], config_path=self.config_path)
        if ET.parse(self.model_path).getroot().attrib.get('model') != 'rp1.3.2':
            raise ValueError('The PN calibration requires the RP1.3.2 model')
        if hashlib.sha256(self.model_path.read_bytes()).hexdigest() != calibration['robot_model_sha256']:
            raise ValueError('RP1 model changed: regenerate PN standard-T-pose calibration')
        repository = resolve_gmr_root(gmr_root)
        if repository is not None and str(repository) not in sys.path:
            sys.path.insert(0, str(repository))
        import general_motion_retargeting.motion_retarget as gm
        import mink
        import mujoco
        self.mink, self.mujoco = mink, mujoco
        registry_key = f'pn_rp1_3_2_{id(self)}'
        gm.ROBOT_XML_DICT[registry_key] = self.model_path
        gm.IK_CONFIG_DICT[registry_key] = {registry_key: self.config_path}
        # GMR loads the deployed robot and creates the calibrated task stack.
        # Its legacy retarget() loop is deliberately not used: the installed
        # mink API requires limits= as a keyword, and one output dt is used.
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                self.gmr = gm.GeneralMotionRetargeting(
                    registry_key, registry_key, verbose=False, use_velocity_limit=False,
                    solver=self.config['solver']['backend'])
        finally:
            gm.ROBOT_XML_DICT.pop(registry_key, None)
            gm.IK_CONFIG_DICT.pop(registry_key, None)
        self.model = self.gmr.model
        if (self.model.nq, self.model.nv, self.model.nu) != (35, 34, 28):
            raise ValueError('Expected floating-base RP1 with 28 hinge joints')
        self.joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                            for i in range(1, self.model.njnt)]
        if self.joint_names != list(ROBOT_JOINT_NAMES):
            raise ValueError('RP1 model joint order differs from the 35-D reference contract')
        self.joint_lower = self.model.jnt_range[1:, 0].copy()
        self.joint_upper = self.model.jnt_range[1:, 1].copy()
        self.configuration = self.gmr.configuration
        self.data = self.configuration.data
        self.adapter = PNSourceAdapter(self.config)
        self.tasks = {name: task for name, task in zip(self.config['ik_match_table1'], self.gmr.tasks1)}
        for task in self.tasks.values():
            task.gain = self.config['solver']['task_gain']
        self.posture = mink.PostureTask(self.model, cost=self.config['solver']['posture_cost'])
        self.knee_joint_indices = {side: self.joint_names.index(side + '_knee_joint')
                                   for side in ('left', 'right')}
        knee_cost = np.zeros(self.model.nv)
        for index in self.knee_joint_indices.values():
            knee_cost[6 + index] = self.config['solver']['knee_posture_cost']
        self.knee_posture = mink.PostureTask(
            self.model, cost=knee_cost, gain=self.config['solver']['task_gain'])
        self.geoms = _foot_geoms(self.model, mujoco)
        self.foot_body_ids = {side: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                                   side + '_ankle_roll_link')
                              for side in ('left', 'right')}
        # Keep capsule endpoint offsets in the foot body frame for target
        # projection; target foot rotation is not assumed to be horizontal.
        self.configuration.update(calibration['robot_tpose_qpos'])
        self.foot_endpoint_local = {}
        for side, ids in self.geoms.items():
            body = self.foot_body_ids[side]
            r = self.data.xmat[body].reshape(3, 3)
            endpoint = []
            for geom in ids:
                axis = self.data.geom_xmat[geom].reshape(3, 3)[:, 2]
                for sign in (-1., 1.):
                    p = self.data.geom_xpos[geom] + sign * self.model.geom_size[geom, 1] * axis
                    endpoint.append((r.T @ (p - self.data.xpos[body]), float(self.model.geom_size[geom, 0])))
            self.foot_endpoint_local[side] = endpoint
        self.motion_limit = _make_motion_limit(mink, self.model, self.config['output_limits'])
        self.ik_limits = [mink.ConfigurationLimit(self.model, gain=.95), self.motion_limit]
        if self.config['solver']['floor_enabled']:
            self.ik_limits.append(_make_floor_limit(mink, mujoco, self.model, self.geoms,
                                                   self.config['solver']))
        if self.config['solver']['collision_enabled']:
            torso = ['torso_lower_collision', 'torso_middle_collision', 'torso_upper_collision']
            arms = {side: [side + '_' + name + '_collision' for name in
                          ['elbow', 'wrist_roll', 'wrist_yaw', 'wrist_pitch']]
                    for side in ('left', 'right')}
            pairs = [(arms['left'], torso), (arms['right'], torso), (arms['left'], arms['right'])]
            self.ik_limits.append(mink.CollisionAvoidanceLimit(
                self.model, geom_pairs=pairs, minimum_distance_from_collisions=.005,
                collision_detection_distance=.04))
        self.reset()

    def _check_parameters(self):
        limits = self.config['output_limits']
        joint_acceleration_enabled(limits)
        for key in ('joint_velocity_rad_s', 'joint_acceleration_rad_s2'):
            value = np.asarray(limits[key], dtype=float)
            if value.shape != (28,) or not np.all(np.isfinite(value)) or np.any(value <= 0):
                raise ValueError(f'{key} must have 28 positive finite values')
            limits[key] = value.tolist()
        for key in DEFAULT_OUTPUT_LIMITS:
            if key.startswith('root_') and (not math.isfinite(float(limits[key])) or limits[key] <= 0):
                raise ValueError(f'{key} must be positive')
        if not 0 < self.config['solver']['task_gain'] <= 1:
            raise ValueError('task_gain must be in (0,1]')
        if not isinstance(self.config['solver']['reference_task_tracking'], bool):
            raise ValueError('reference_task_tracking must be boolean')
        if not isinstance(self.config['contact']['enabled'], bool):
            raise ValueError('contact.enabled must be boolean')
        for key in ('knee_posture_cost', 'stance_root_position_cost'):
            value = float(self.config['solver'][key])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{key} must be finite and nonnegative')
            self.config['solver'][key] = value

    def reset(self, qpos=None):
        q = _validated_qpos(IDLE if qpos is None else qpos, self.joint_lower, self.joint_upper)
        self.configuration.update(q)
        self.motion_limit.previous_world_velocity[:] = 0.
        self.posture.set_target(q)
        self.knee_posture.set_target(q)
        self._initialized = qpos is not None
        self._source_time = None
        self._previous_raw_feet = None
        self._contacts = {side: {'active': False, 'count': 0, 'exit_count': 0,
                                'anchor': None, 'source_anchor': None}
                          for side in ('left', 'right')}
        self.last_targets = {}
        self.last_stats = {}

    def _target_sole_height(self, side, pose):
        p, q = pose
        rotation = _rotation(q)
        return min(float((p + rotation.apply(endpoint))[2] - radius)
                   for endpoint, radius in self.foot_endpoint_local[side])

    def sole_min_height(self):
        result = {}
        for side, ids in self.geoms.items():
            result[side] = min(
                float(self.data.geom_xpos[g, 2]
                      - abs(self.data.geom_xmat[g].reshape(3, 3)[2, 2]) * self.model.geom_size[g, 1]
                      - self.model.geom_size[g, 0])
                for g in ids)
        return result

    def _contact_targets(self, targets, dt, poses):
        cfg = self.config['contact']
        clearance = self.config['solver']['floor_clearance_m']
        # Contact belongs to the observed human foot. Root and segment scales
        # differ; classifying the mapped robot target creates fictitious foot
        # motion when the human shifts their pelvis over a planted foot.
        raw = {side: [np.array(poses[side.title() + 'Foot'][0], dtype=float, copy=True),
                      _wxyz(_rotation(poses[side.title() + 'Foot'][1]))]
               for side in ('left', 'right')}
        speed = {side: 0. if self._previous_raw_feet is None else
                 float(np.linalg.norm(raw[side][0] - self._previous_raw_feet[side][0]) / dt)
                 for side in raw}
        source_offsets = self.config['source_calibration']['source_sole_offsets_local_m']
        raw_height = {side: float((raw[side][0] + _rotation(raw[side][1]).apply(
                      source_offsets[side]))[2]) - cfg['source_floor_height_m'] for side in raw}
        state = copy.deepcopy(self._contacts)
        if not cfg['enabled']:
            # Disable the entire pre-IK support stage, including target floor
            # projection. The independent IK floor/motion limits remain active.
            # Return new state so a failed QP cannot commit a contact release.
            for item in state.values():
                item.update(active=False, count=0, exit_count=0, anchor=None, source_anchor=None)
            return targets, state, raw, raw_height, speed
        for side in raw:
            name = side + '_ankle_roll_link'
            item = state[side]
            if item['active']:
                # Vertical floor projection is a calibration correction, not
                # a horizontal step. Height already has its own exit test.
                moved = np.linalg.norm(raw[side][0][:2] - item['source_anchor'][:2])
                exit_contact = (raw_height[side] > cfg['exit_height_m'] or speed[side] > cfg['exit_speed_m_s']
                                or moved > cfg['anchor_release_distance_m'])
                item['exit_count'] = item['exit_count'] + 1 if exit_contact else 0
                if item['exit_count'] >= cfg['exit_frames'] or not cfg['enabled']:
                    item.update(active=False, count=0, exit_count=0, anchor=None, source_anchor=None)
            if not item['active']:
                enter = (cfg['enabled'] and raw_height[side] < cfg['enter_height_m']
                         and speed[side] < cfg['enter_speed_m_s'])
                item['count'] = item['count'] + 1 if enter else 0
                if item['count'] >= cfg['enter_frames']:
                    mapped = targets[name]
                    rotation = _rotation(mapped[1]).as_matrix()
                    # A support sole is flat. Retain heading, suppress PN
                    # ankle roll/pitch noise while this foot remains planted.
                    heading = math.atan2(rotation[1, 0], rotation[0, 0])
                    anchor = [mapped[0].copy(), _wxyz(Rotation.from_euler('z', heading))]
                    anchor[0][2] += clearance - self._target_sole_height(side, anchor)
                    item.update(active=True, anchor=anchor, source_anchor=raw[side][0].copy(), exit_count=0)
            if item['active']:
                targets[name] = [v.copy() for v in item['anchor']]
            else:
                # Preserve lifted feet; never flatten the entire motion to
                # whichever foot happens to be lowest in the current frame.
                h = self._target_sole_height(side, targets[name])
                if h < clearance:
                    targets[name][0][2] += clearance - h
        return targets, state, raw, raw_height, speed

    def retarget(self, poses, dt=.02, source_time_s=None):
        started = time.perf_counter()
        dt = float(dt)
        if not math.isfinite(dt) or not .001 <= dt <= .1:
            raise ValueError('dt must be a finite output period between .001 and .1 s')
        if source_time_s is not None:
            source_time_s = float(source_time_s)
            if not math.isfinite(source_time_s) or (self._source_time is not None
                                                     and source_time_s < self._source_time - 1e-9):
                raise ValueError('source time must be finite and cannot go backwards')
        # No state changes before all required source poses have validated.
        targets = self.adapter.target_frames(poses)
        knee_flexion = pn_knee_flexion(poses)
        contact_dt = dt if self._source_time is None or source_time_s is None else max(
            source_time_s - self._source_time, .001)
        targets, contact_state, raw_feet, raw_height, raw_speed = self._contact_targets(targets, contact_dt, poses)
        rollback_q = self.configuration.q.copy()
        previous = rollback_q.copy()
        if not self._initialized:
            initial = previous.copy()
            initial[:3], initial[3:7] = targets['base_link']
            # Cold solver initialization is not a published jump. Counter the
            # virtual pelvis pitch at both mechanical hip axes before the
            # first FK, instead of tilting the entire idle leg chain forward.
            # Subsequent and published transitions still use bounded QPs.
            pitch = self.adapter.torso_compensation_angle(poses)
            initial[7 + self.joint_names.index('left_hip_pitch_joint')] -= pitch
            initial[7 + self.joint_names.index('right_hip_pitch_joint')] += pitch
            self.configuration.update(initial)
            minimum = min(self.sole_min_height().values())
            initial[2] += max(0., self.config['solver']['floor_clearance_m']
                             + self.config['solver']['floor_braking_margin_m'] - minimum)
            self.configuration.update(initial)
            previous = initial.copy()
        preprocess_ms = (time.perf_counter() - started) * 1000.
        knee_target = self.configuration.q.copy()
        for side, sign in (('left', -1.), ('right', 1.)):
            index = self.knee_joint_indices[side]
            knee_target[7 + index] = np.clip(
                sign * knee_flexion[side], self.joint_lower[index] + 1e-5,
                self.joint_upper[index] - 1e-5)
        self.knee_posture.set_target(knee_target)
        has_support = any(item['active'] for item in contact_state.values())
        for name, task in self.tasks.items():
            pos, quat = targets[name]
            task.set_target(self.mink.SE3.from_rotation_and_translation(self.mink.SO3(quat), pos))
            entry = self.config['ik_match_table1'][name]
            side = 'left' if name.startswith('left_') else 'right' if name.startswith('right_') else None
            planted = name.endswith('ankle_roll_link') and contact_state[side]['active']
            position_cost = self.config['contact']['position_cost'] if planted else entry[1]
            if name == 'base_link' and has_support:
                # Fixed feet take priority over the differently scaled human
                # pelvis trajectory. Let the base move to a reachable support
                # pose instead of obtaining reach through extreme roll/yaw.
                position_cost = self.config['solver']['stance_root_position_cost']
            task.set_position_cost(position_cost)
            task.set_orientation_cost(self.config['contact']['orientation_cost'] if planted else entry[2])
        solve_start = time.perf_counter()
        try:
            velocity = self.mink.solve_ik(
                self.configuration, list(self.tasks.values()) + [self.posture, self.knee_posture], dt,
                solver=self.config['solver']['backend'], damping=self.config['solver']['damping'],
                safety_break=True, limits=self.ik_limits,
                **({'primal_tol': 1e-9, 'dual_tol': 1e-9}
                   if self.config['solver']['backend'] == 'daqp' else {}))
            self.configuration.integrate_inplace(velocity, dt)
            q = self.configuration.q.copy()
            if not np.all(np.isfinite(q)):
                raise RuntimeError('mink produced a non-finite reference')
            # Verify the FINAL result, rather than merely the per-QP limit.
            actual_v = np.empty(self.model.nv)
            self.mujoco.mj_differentiatePos(self.model, actual_v, dt, previous, q)
            actual_v[3:6] = _rotation(previous[3:7]).apply(actual_v[3:6])
            self._check_motion_result(actual_v, dt, q)
        except Exception:
            self.configuration.update(rollback_q)
            raise
        solver_ms = (time.perf_counter() - solve_start) * 1000.
        self.motion_limit.previous_world_velocity = actual_v.copy()
        self._contacts, self._previous_raw_feet = contact_state, raw_feet
        self._source_time = source_time_s
        self._initialized = True
        self.last_targets = {name: [pos.copy(), quat.copy()] for name, (pos, quat) in targets.items()}
        self.last_stats = {
            'preprocess_ms': preprocess_ms, 'solver_ms': solver_ms,
            'total_ms': (time.perf_counter() - started) * 1000., 'qp_solves': 1,
            'contacts': {s: bool(contact_state[s]['active']) for s in contact_state},
            'pre_ik_support_enabled': self.config['contact']['enabled'],
            'contact_source': 'native PN world feet before robot root/segment scaling',
            'torso_compensation_pitch_deg': math.degrees(self.adapter.torso_compensation_angle(poses)),
            'source_knee_flexion_deg': {s: math.degrees(v) for s, v in knee_flexion.items()},
            'raw_target_sole_height_m': raw_height, 'raw_target_foot_speed_m_s': raw_speed,
            'robot_sole_min_height_m': self.sole_min_height(),
            'joint_speed_max_rad_s': float(np.max(np.abs(actual_v[6:]))),
            'root_linear_speed_m_s': float(np.linalg.norm(actual_v[:3])),
            'root_angular_speed_rad_s': float(np.linalg.norm(actual_v[3:6])),
        }
        return q

    def _check_motion_result(self, velocity_world, dt, qpos):
        limits = self.config['output_limits']
        acceleration = (velocity_world - self.motion_limit.previous_world_velocity) / dt
        checks = [('joint velocity', np.abs(velocity_world[6:]),
                   np.asarray(limits['joint_velocity_rad_s']))]
        if joint_acceleration_enabled(limits):
            checks.append(('joint acceleration', np.abs(acceleration[6:]),
                           np.asarray(limits['joint_acceleration_rad_s2'])))
        checks.extend([
            ('root linear velocity', np.linalg.norm(velocity_world[:3]), limits['root_linear_velocity_limit_m_s']),
            ('root angular velocity', np.linalg.norm(velocity_world[3:6]), limits['root_angular_velocity_limit_rad_s']),
            ('root linear acceleration', np.linalg.norm(acceleration[:3]), limits['root_linear_acceleration_limit_m_s2']),
            ('root angular acceleration', np.linalg.norm(acceleration[3:6]), limits['root_angular_acceleration_limit_rad_s2']),
        ])
        for label, value, maximum in checks:
            excess = np.asarray(value) - np.asarray(maximum)
            if np.any(excess > 2e-4):
                raise RuntimeError(f'final IK output violated velocity/acceleration constraints: '
                                   f'{label}, maximum excess {float(np.max(excess)):.9g}')
        if np.any(qpos[7:] < self.joint_lower - 1e-7) or np.any(qpos[7:] > self.joint_upper + 1e-7):
            raise RuntimeError('final IK output violated joint limits')

    def metadata(self):
        from pn_rp1_kinematic_model import export_kinematic_xml

        limits = self.config['output_limits']
        acceleration_enabled = joint_acceleration_enabled(limits)
        return {
            'type': 'ready', 'model': 'rp1.3.2', 'joint_names': self.joint_names,
            'joint_lower': self.joint_lower.tolist(), 'joint_upper': self.joint_upper.tolist(),
            'velocity_limits': limits['joint_velocity_rad_s'],
            'acceleration_limits': limits['joint_acceleration_rad_s2'],
            'joint_acceleration_enabled': acceleration_enabled,
            'joint_acceleration_braking_enabled': acceleration_enabled,
            'acceleration_limits_semantics': ('configured hinge acceleration limits; enforced with hinge braking'
                if acceleration_enabled else
                'configured hinge values only; hinge acceleration and its braking envelope are disabled; root limits remain enforced'),
            **{k: v for k, v in limits.items() if k.startswith('root_')},
            'default_dt_s': .02, 'standard_tpose': True,
            'kinematic_model_xml': export_kinematic_xml(self.model),
            'floor_options': self.config['solver'],
            'contact_options': self.config['contact'],
        }


def worker(config, gmr_root, dt, model_path=None):
    solver = PNToRP1Retargeter(config, gmr_root=gmr_root, model_path=model_path)
    print(json.dumps(solver.metadata()), flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        request = {}
        try:
            request = json.loads(line)
            if request.get('type') == 'reset':
                solver.reset(request.get('qpos'))
                response = {'type': 'reset', 'seq': request.get('seq')}
            else:
                qpos = solver.retarget(request['poses'], request.get('dt_s', dt), request.get('source_time_s'))
                response = {'seq': request.get('seq'), 'qpos': qpos.tolist(), 'stats': solver.last_stats,
                            'source_time_s': request.get('source_time_s'),
                            'received_monotonic_s': request.get('received_monotonic_s')}
            print(json.dumps(response, allow_nan=False), flush=True)
        except Exception as exc:
            print(json.dumps({'seq': request.get('seq'), 'error': f'{type(exc).__name__}: {exc}'}), flush=True)
            print(f'PN retarget rejected frame: {exc}', file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    command = sub.add_parser('worker', help='JSONL stdin/stdout solver process; no ROS/hardware')
    command.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    command.add_argument('--gmr-root', type=Path, default=None,
                         help='external GMR checkout; defaults to PN_RP1_GMR_ROOT or installed GMR')
    command.add_argument('--model', type=Path, default=None,
                         help='relocated RP1 MJCF; otherwise PN_RP1_MODEL or calibration path')
    command.add_argument('--dt', type=float, default=.02)
    args = parser.parse_args()
    worker(args.config, args.gmr_root, args.dt, args.model)


if __name__ == '__main__':
    main()
