"""Browser 3D playback of the actual RP1.3.2 MJCF visual model and references.

MuJoCo computes the exact FK; Viser draws the compiled model's real meshes.
This viewer never sends robot commands or opens a PN device. Default playback
uses the final published reference, including the initial follow transition.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np

from pn_rp1_calibration import PN_JOINT_NAMES, PN_PARENTS, ROBOT_JOINT_NAMES
from pn_rp1_paths import default_mesh_cache, resolve_model_path


HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent / "analysis/retarget_audit/pn_rp1_gmr"
REPLAY_AUDIT = HERE.parent / "analysis/retarget_audit/pn_rp1_gmr_v2"
DEFAULT_CONFIG = HERE / "rp1/pn_axis_to_rp1_28dof.json"
DEFAULT_LIVE_CONFIG = HERE / 'rp1/pn_axis_to_rp1_28dof_current.json'
DEFAULT_REFERENCE = REPLAY_AUDIT / "reference_replay.jsonl"
DEFAULT_CORE = REPLAY_AUDIT / "replay.npz"
DEFAULT_SCALE_COMPARISON = AUDIT.parent / "pn_rp1_scale_trials/viewer_comparison.json"
DIRECT_REFERENCE_AUDIT = AUDIT.parent / 'pn_rp1_reference_trials/alignment_transition_only'
ROS_SOURCE_LABEL = "实时 ROS reference（只读）"
STDIN_SOURCE_LABEL = '实时 PN + reference（发布端直连）'
DEFAULT_ROBOT_OPACITY = .35
RAW_PN_COLOR = (52, 143, 235)
SCALED_PN_COLOR = (164, 94, 218)
LIVE_PN_TIMEOUT_S = 1.


def validate_reference(values):
    qpos = np.asarray(values, dtype=float)
    if qpos.shape != (35,) or not np.isfinite(qpos).all():
        raise ValueError("reference needs 35 finite values: xyz + wxyz + 28 joints")
    if abs(float(np.linalg.norm(qpos[3:7]))-1.) > 1e-4:
        raise ValueError("reference root quaternion must be normalized wxyz")
    return qpos.copy()


def validate_live_pn(value):
    """Keep the original PN geometry, in canonical display joint order."""
    names = tuple(value['joint_names'])
    if len(names) != len(PN_JOINT_NAMES) or set(names) != set(PN_JOINT_NAMES):
        raise ValueError('live PN needs the complete 21-joint body')
    positions = np.asarray(value['positions_m'], dtype=float)
    if positions.shape != (len(names), 3) or not np.isfinite(positions).all():
        raise ValueError('live PN positions need 21 finite XYZ values')
    seq = value['seq']
    if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
        raise ValueError('live PN sequence must be a nonnegative integer')
    source_time = float(value['source_time_s'])
    received = float(value['received_monotonic_s'])
    if not np.isfinite([source_time, received]).all():
        raise ValueError('live PN timestamps must be finite')
    order = [names.index(name) for name in PN_JOINT_NAMES]
    return {'positions_m': positions[order].copy(), 'seq': seq,
            'source_time_s': source_time, 'received_monotonic_s': received}


class StdinReferenceSource:
    """Receive reference and optional raw PN over one pipe, without ROS/SDK."""
    def __init__(self, stream=None):
        self.stream = sys.stdin if stream is None else stream
        self.lock = threading.Lock()
        self.latest = None
        self.latest_frame = None
        self.received = self.rejected = 0
        self.pn_rejected = 0
        self.stopped = threading.Event()
        self.eof = threading.Event()
        self.thread = threading.Thread(target=self._read, name='rp1-viser-pipe', daemon=True)
        self.thread.start()

    def _read(self):
        try:
            for line in self.stream:
                if self.stopped.is_set():
                    break
                try:
                    packet = json.loads(line)
                    qpos = validate_reference(packet['qpos'])
                except (ValueError, TypeError, KeyError):
                    with self.lock:
                        self.rejected += 1
                    continue
                pn, bad_pn = None, False
                if 'pn' in packet:
                    try:
                        pn = validate_live_pn(packet['pn'])
                    except (ValueError, TypeError, KeyError, OverflowError):
                        bad_pn = True
                reference_time = packet.get('reference_source_time_s')
                try:
                    reference_time = None if reference_time is None else float(reference_time)
                    if reference_time is not None and not np.isfinite(reference_time):
                        reference_time = None
                except (ValueError, TypeError, OverflowError):
                    reference_time = None
                received = time.monotonic()
                with self.lock:
                    self.latest = (qpos, received)
                    # A missing or invalid optional skeleton clears the previous
                    # one; it must not make a valid published robot pose stale.
                    self.latest_frame = {'qpos': qpos, 'pn': pn,
                                         'reference_source_time_s': reference_time,
                                         'received_monotonic_s': received}
                    self.received += 1
                    self.pn_rejected += int(bad_pn)
        finally:
            self.eof.set()

    def sample(self):
        with self.lock:
            return self.latest, self.received, self.rejected

    def sample_frame(self):
        """Read the robot and PN snapshots from one publication atomically."""
        with self.lock:
            return self.latest_frame, self.received, self.rejected

    def close(self):
        # The bridge closes the pipe or terminates the child. A blocked stdin
        # reader is a daemon and must never delay closing the Viser server.
        self.stopped.set()
        if self.eof.is_set():
            self.thread.join(timeout=.2)


def live_replay(calibration):
    """Bootstrap a live-only scene entirely from model calibration."""
    names = list(calibration['target_frames'])
    qpos = np.asarray(calibration['robot_tpose_qpos'], dtype=float)
    positions = np.asarray([calibration['robot_tpose_world'][name][0] for name in names])
    times = np.array([0., .02])
    replay = Replay('最终 50 Hz reference', times, np.stack([qpos, qpos]), times,
                    np.zeros(2, dtype=bool), reference_pipeline='live_reference')
    return replay, names, times, np.stack([positions, positions])


class ROSReferenceSource:
    """Read-only ROS subscription, with a single replaceable latest sample."""
    def __init__(self, topic):
        try:
            import rclpy
            from rclpy.context import Context
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
            from std_msgs.msg import Float32MultiArray
        except ImportError as error:
            raise RuntimeError("ROS visualization needs source /opt/ros/humble/setup.zsh before starting") from error
        self.rclpy = rclpy
        self.context = Context()
        rclpy.init(context=self.context)
        self.node = rclpy.create_node("pn_rp1_viser_readonly", context=self.context)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.lock = threading.Lock()
        self.latest = None
        self.received = 0
        self.rejected = 0
        self.stopped = threading.Event()
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=21,
                         reliability=ReliabilityPolicy.RELIABLE)

        def callback(message):
            try:
                qpos = validate_reference(message.data)
            except ValueError:
                with self.lock:
                    self.rejected += 1
                return
            with self.lock:
                self.latest = (qpos, time.monotonic())
                self.received += 1

        self.subscription = self.node.create_subscription(Float32MultiArray, topic, callback, qos)
        self.thread = threading.Thread(target=self._spin, name="rp1-viser-ros", daemon=True)
        self.thread.start()

    def _spin(self):
        from rclpy.executors import ExternalShutdownException, ShutdownException
        try:
            while not self.stopped.is_set() and self.context.ok():
                self.executor.spin_once(timeout_sec=.02)
        except (ExternalShutdownException, ShutdownException):
            pass

    def sample(self):
        with self.lock:
            return self.latest, self.received, self.rejected

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=2.)
        if self.thread.is_alive():
            self.context.shutdown()
            self.thread.join(timeout=1.)
        self.executor.remove_node(self.node)
        self.executor.shutdown(timeout_sec=1.)
        self.node.destroy_node()
        if self.context.ok():
            self.context.shutdown()


@dataclass(frozen=True)
class Replay:
    name: str
    times: np.ndarray
    qpos: np.ndarray
    source_times: np.ndarray
    blend: np.ndarray
    alignment: np.ndarray | None = None
    root_position_scale: float | None = None
    joint_velocity_rad_s: float | None = None
    reference_task_tracking: bool = False
    mapping_profile: str = 'yaw'
    reference_pipeline: str = 'constrained_qp'
    pre_ik_support_enabled: bool | None = None
    ik_floor_enabled: bool | None = None
    torso_pitch_compensation_enabled: bool | None = None
    method: str = 'gmr'
    display_pn_scale: float | None = None
    root_z_offset_m: float = 0.
    joint_velocity_profile: tuple[float, ...] | None = None
    joint_velocity_profile_name: str | None = None
    joint_acceleration_enabled: bool | None = None

    @property
    def duration(self):
        return float(self.times[-1])

    def index(self, seconds):
        return max(0, min(len(self.times) - 1,
                          int(np.searchsorted(self.times, seconds, side="right") - 1)))


def _validate_replay(replay):
    if len(replay.times) < 2 or replay.qpos.shape != (len(replay.times), 35):
        raise ValueError(f"{replay.name}: expected at least two 35-value references")
    if not np.isfinite(replay.qpos).all() or not np.isfinite(replay.times).all():
        raise ValueError(f"{replay.name}: nonfinite references/timestamps")
    if np.any(np.diff(replay.times) <= 0):
        raise ValueError(f"{replay.name}: playback timestamps must strictly increase")
    if (replay.source_times.shape != replay.times.shape
        or not np.isfinite(replay.source_times).all()
        or np.any(np.diff(replay.source_times) < 0)):
        raise ValueError(f"{replay.name}: invalid PN source timestamps")
    if np.max(np.abs(np.linalg.norm(replay.qpos[:, 3:7], axis=1) - 1.)) > 1e-5:
        raise ValueError(f"{replay.name}: root quaternions must be normalized wxyz")
    return replay


def load_core(path):
    with np.load(path, allow_pickle=False) as archive:
        times = archive["time_s"].astype(float)
        return _validate_replay(Replay("IK 输出", times - times[0],
            archive["qpos"].astype(float), archive["source_time_s"].astype(float),
            np.zeros(len(times), dtype=bool), reference_pipeline='core_ik'))


def _root_z_offset(value):
    """Height metadata describes publication only, never an extra pose edit."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('root_z_offset_m must be a finite number')
    try:
        offset = float(value)
    except OverflowError as error:
        raise ValueError('root_z_offset_m must be a finite number') from error
    if not np.isfinite(offset):
        raise ValueError('root_z_offset_m must be a finite number')
    return offset


def load_final(path, *, allowed_pipelines=None):
    rows = []
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row.get("mode") == "live" and row.get("source_time_s") is not None:
                    rows.append(row)
    if not rows:
        raise ValueError("final reference log has no active source references")
    offsets = [_root_z_offset(row.get('root_z_offset_m', 0.)) for row in rows]
    if any(offset != offsets[0] for offset in offsets[1:]):
        raise ValueError('final reference log has mixed root height offsets')
    pipelines = {row.get('reference_pipeline', 'constrained_qp') for row in rows}
    allowed = ({'constrained_qp', 'alignment_transition_only'} if allowed_pipelines is None
               else set(allowed_pipelines))
    if len(pipelines) != 1 or not pipelines.issubset(allowed):
        raise ValueError('final reference log has mixed or unknown processing pipelines')
    pipeline = pipelines.pop()
    if pipeline == 'alignment_transition_only':
        if any(row.get('final_constraints_enabled') is not False for row in rows):
            raise ValueError('direct reference must declare final constraints disabled')
        for row in rows:
            if row.get('desired_qpos') != row['qpos']:
                raise ValueError('direct reference differs from alignment/transition proposal')
    # Keep every actual publication while the recording is advancing, including
    # repeated source samples and the initial two-second transition. Omit the
    # trailing stale-source hold/return so replay ends at the take's final frame.
    final_source_time = max(float(row["source_time_s"]) for row in rows)
    last = next(i for i, row in enumerate(rows)
                if float(row["source_time_s"]) == final_source_time)
    rows = rows[:last + 1]
    times = np.asarray([row["time_s"] for row in rows], dtype=float)
    align = np.asarray([row.get("alignment_se2") for row in rows], dtype=float)
    if align.shape != (len(rows), 6) or not np.isfinite(align).all():
        raise ValueError("final references need recorded SE(2) alignment for target overlays")
    return _validate_replay(Replay("最终 50 Hz reference", times - times[0],
        np.asarray([row["qpos"] for row in rows], dtype=float),
        np.asarray([row["source_time_s"] for row in rows], dtype=float),
        np.asarray([row.get("blend_active", False) for row in rows], dtype=bool), align,
        reference_pipeline=pipeline, root_z_offset_m=offsets[0]))


@dataclass(frozen=True)
class PNSkeleton:
    names: tuple
    parents: np.ndarray
    times: np.ndarray
    positions: np.ndarray
    source_indices: np.ndarray | None = None


def pn_edges(names, parents=None):
    names = tuple(str(name) for name in names)
    if len(names) != 21 or len(set(names)) != 21 or set(names) != set(PN_JOINT_NAMES):
        raise ValueError('expected the 21 PN body joint names')
    indices = {name: i for i, name in enumerate(names)}
    expected = np.asarray([-1 if PN_PARENTS[name] is None else indices[PN_PARENTS[name]]
                           for name in names], dtype=int)
    if parents is not None and not np.array_equal(np.asarray(parents), expected):
        raise ValueError('PN parent graph differs from the body skeleton')
    return [(int(parent), i) for i, parent in enumerate(expected) if parent >= 0]


def load_pn_layers(core_path):
    """Read original PN samples paired with the core's processed target rows."""
    with np.load(core_path, allow_pickle=False) as archive:
        fields = {'source_position_m', 'source_joint_names', 'source_parent_index'}
        present = fields.intersection(archive.files)
        if not present:
            return None
        if present != fields:
            raise ValueError('incomplete original PN geometry in core replay')
        names = tuple(str(name) for name in archive['source_joint_names'])
        parents = archive['source_parent_index'].copy()
        pn_edges(names, parents)
        positions = archive['source_position_m'].astype(float)
        times = archive['source_time_s'].astype(float)
        source_indices = archive['source_index'].copy() if 'source_index' in archive else None
        if 'target_position_m' in archive and archive['target_position_m'].shape[0] != len(times):
            raise ValueError('PN and processed targets must have paired rows')
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0.) or positions.shape != (len(times), 21, 3)
        or not np.isfinite(positions).all()):
        raise ValueError('invalid original PN samples or source timestamps')
    if source_indices is not None and (source_indices.shape != times.shape
        or not np.issubdtype(source_indices.dtype, np.integer) or np.any(source_indices < 0)):
        raise ValueError('invalid original PN frame indices')
    return PNSkeleton(names, parents, times, positions, source_indices)


def causal_index(times, seconds):
    return max(0, min(len(times)-1, int(np.searchsorted(times, seconds, side='right')-1)))


def scale_only_pn(positions, scale, source_root_ref, robot_root_ref):
    """Uniform scale of every human joint about the fixed standard-T root."""
    positions = np.asarray(positions, dtype=float)
    source = np.asarray(source_root_ref, dtype=float)
    robot = np.asarray(robot_root_ref, dtype=float)
    scale = float(scale)
    if (positions.ndim < 2 or positions.shape[-1] != 3 or not np.isfinite(positions).all()
        or source.shape != (3,) or robot.shape != (3,) or not np.isfinite(source).all()
        or not np.isfinite(robot).all() or not np.isfinite(scale) or scale <= 0.):
        raise ValueError('uniform PN scaling needs finite XYZ and a positive scale')
    return robot + scale*(positions-source)


def target_edges(names):
    """Connect the actual task frames, including either proximal mapping."""
    indices = {name: i for i, name in enumerate(names)}
    chains = [['base_link', 'waist_yaw_link']]
    for side in ('left', 'right'):
        hip = next((f'{side}_hip_{axis}_link' for axis in ('pitch', 'yaw')
                    if f'{side}_hip_{axis}_link' in indices), None)
        shoulder = next((f'{side}_shoulder_{axis}_link' for axis in ('pitch', 'yaw')
                         if f'{side}_shoulder_{axis}_link' in indices), None)
        chains.extend([
            ['base_link', hip, f'{side}_knee_link', f'{side}_ankle_roll_link'],
            ['waist_yaw_link', shoulder, f'{side}_elbow_link', f'{side}_wrist_pitch_link']])
    return [(indices[a], indices[b]) for chain in chains for a, b in zip(chain, chain[1:])
            if a in indices and b in indices]


def support_metadata(config):
    """Keep contact-target preparation distinct from the IK floor constraint."""
    support = config.get('contact', {}).get('enabled')
    floor = config.get('solver', {}).get('floor_enabled')
    if any(value is not None and not isinstance(value, bool) for value in (support, floor)):
        raise ValueError('comparison contact and floor settings must be boolean')
    return {'pre_ik_support_enabled': support, 'ik_floor_enabled': floor}


def joint_acceleration_metadata(config):
    enabled = config.get('output_limits', {}).get('joint_acceleration_enabled', True)
    if not isinstance(enabled, bool):
        raise ValueError('joint_acceleration_enabled must be boolean')
    return {'joint_acceleration_enabled': enabled}


def support_description(replay):
    support = getattr(replay, 'pre_ik_support_enabled', None)
    if support is None:
        return ''
    description = f"IK 前脚部支撑处理：{'开' if support else '关'}"
    floor = getattr(replay, 'ik_floor_enabled', None)
    if floor is not None:
        description += f"；IK 内部地板约束{'保留' if floor else '关闭'}"
        if floor and getattr(replay, 'root_z_offset_m', 0.) != 0.:
            description += '（仅约束高度偏移前）'
    return description


def height_offset_description(replay):
    if getattr(replay, 'method', 'gmr') != 'gmr':
        return ''
    if replay.reference_pipeline in ('core_ik', 'live_reference'):
        return ('发布高度偏移：0.000 m（IK 输出，未进入发布处理）'
                if replay.reference_pipeline == 'core_ik' else '')
    offset = replay.root_z_offset_m
    description = f'发布高度偏移：{offset:+.3f} m（IK 后，过渡前）'
    if offset < 0.:
        description += '；整体降低后可能穿地，不做二次地板修正'
    return description


def torso_compensation_metadata(config):
    """Describe adapter pitch compensation, separately from pitch-link mapping."""
    enabled = config.get('torso_compensation', {}).get('enabled')
    if enabled is not None and not isinstance(enabled, bool):
        raise ValueError('comparison torso compensation setting must be boolean')
    return {'torso_pitch_compensation_enabled': enabled}


def torso_compensation_description(replay):
    enabled = getattr(replay, 'torso_pitch_compensation_enabled', None)
    if enabled is None:
        return ''
    return f"躯干→骨盆 pitch 补偿：{'开' if enabled else '关'}"


def processed_target_hint(replay):
    if getattr(replay, 'method', 'gmr') == 'legacy_euler':
        return '原版欧拉角方法没有 GMR 任务目标；橙色层在此版本禁用。'
    pitch = getattr(replay, 'torso_pitch_compensation_enabled', None)
    hint = ('14 个机器人任务点，包含 CAD 骨长；腰 pitch 补偿关闭'
            if pitch is False else '14 个机器人任务点，包含 CAD 骨长、腰 pitch 补偿')
    support = getattr(replay, 'pre_ik_support_enabled', None)
    if support is False:
        return hint + '；已取消 IK 前的支撑检测、锁脚和支撑姿态修正。'
    if support is True:
        return hint + '与 IK 前脚部支撑处理（检测、锁脚、支撑姿态修正）。'
    return hint + '与脚部接触修正。'


def _positive_joint_velocity(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('joint velocity must be finite positive numbers')
    try:
        speed = float(value)
    except OverflowError as error:
        raise ValueError('joint velocity must be finite positive numbers') from error
    if not np.isfinite(speed) or speed <= 0.:
        raise ValueError('joint velocity must be finite positive numbers')
    return speed


def joint_velocity_metadata(value, configured, *, profile_name=None, joint_names=None):
    """Validate canonical per-joint limits without replacing them by a scalar."""
    empty = {'joint_velocity_rad_s': None, 'joint_velocity_profile': None,
             'joint_velocity_profile_name': None}
    if value is None:
        if profile_name is not None:
            raise ValueError('joint velocity profile needs 28 declared limits')
        return empty
    if not isinstance(configured, list) or len(configured) != len(ROBOT_JOINT_NAMES):
        raise ValueError('comparison joint velocity differs from calibration')
    actual = tuple(_positive_joint_velocity(item) for item in configured)
    if isinstance(value, list):
        if len(value) != len(ROBOT_JOINT_NAMES):
            raise ValueError('joint velocity profile needs 28 declared limits')
        speeds = tuple(_positive_joint_velocity(item) for item in value)
        if joint_names is not None and tuple(joint_names) != tuple(ROBOT_JOINT_NAMES):
            raise ValueError('joint velocity profile needs canonical RP1 joint order')
        if speeds != actual:
            raise ValueError('comparison joint velocity differs from calibration')
        if profile_name not in (None, 'legacy_sdk'):
            raise ValueError('unknown joint velocity profile')
        if profile_name == 'legacy_sdk':
            legacy = json.loads((HERE / 'rp1/retarget.json').read_text())['retargetJoints']
            limits = {joint['retargetJoint']: joint['maxSpeed'] for joint in legacy}
            expected = tuple(_positive_joint_velocity(limits[name]) for name in ROBOT_JOINT_NAMES)
            if speeds != expected:
                raise ValueError('joint velocity profile differs from legacy SDK joint limits')
        return dict(empty, joint_velocity_profile=speeds,
                    joint_velocity_profile_name=profile_name)
    speed = _positive_joint_velocity(value)
    if profile_name is not None:
        raise ValueError('joint velocity profile needs 28 declared limits')
    if not np.allclose(actual, speed, rtol=0., atol=1e-10):
        raise ValueError('comparison joint velocity differs from calibration')
    return dict(empty, joint_velocity_rad_s=speed)


def joint_velocity_description(replay):
    if replay.joint_velocity_profile is not None:
        values = '/'.join(f'{value:.2f}' for value in sorted(set(replay.joint_velocity_profile)))
        source = ('按旧SDK逐关节' if replay.joint_velocity_profile_name == 'legacy_sdk'
                  else '逐关节配置')
        return f'关节速度：{source} {values} rad/s'
    if replay.joint_velocity_rad_s is not None:
        return f'关节速度配置 {replay.joint_velocity_rad_s:g} rad/s'
    return ''


def load_scale_comparisons(manifest_path, calibration, target_names):
    """Load each trial with its own reference, source targets and scale metadata."""
    path = Path(manifest_path)
    result = {}
    for trial in json.loads(path.read_text())["variants"]:
        scale = float(trial["scale"])
        if not np.isfinite(scale) or not .1 <= scale <= 2.:
            raise ValueError("invalid comparison root scale")
        config = json.loads((path.parent / trial["config"]).read_text())
        candidate = config["source_calibration"]
        support = support_metadata(config)
        compensation = torso_compensation_metadata(config)
        acceleration = joint_acceleration_metadata(config)
        if ('joint_acceleration_enabled' in trial
            and (not isinstance(trial['joint_acceleration_enabled'], bool)
                 or trial['joint_acceleration_enabled'] is not acceleration['joint_acceleration_enabled'])):
            raise ValueError('comparison joint acceleration setting differs from configuration')
        if 'contact_enabled' in trial:
            enabled = trial['contact_enabled']
            if (not isinstance(enabled, bool)
                or enabled is not support['pre_ik_support_enabled']):
                raise ValueError('comparison contact setting differs from configuration')
        if 'torso_compensation_enabled' in trial:
            enabled = trial['torso_compensation_enabled']
            if (not isinstance(enabled, bool)
                or enabled is not compensation['torso_pitch_compensation_enabled']):
                raise ValueError('comparison torso compensation setting differs from configuration')
        variant_label = trial.get('label')
        if 'label' in trial:
            if not isinstance(variant_label, str) or not variant_label.strip():
                raise ValueError('comparison label must be a nonempty string')
            variant_label = variant_label.strip()
        if not np.isclose(float(candidate["root_position_scale"]), scale, rtol=0., atol=1e-10):
            raise ValueError("comparison scale differs from calibration")
        if (candidate["robot_model_sha256"] != calibration["robot_model_sha256"]
            or not np.array_equal(candidate["robot_tpose_qpos"], calibration["robot_tpose_qpos"])):
            raise ValueError("comparison model or standard T-pose differs")
        profile = trial.get('mapping_profile')
        if profile is not None and (profile not in ('yaw', 'pitch')
            or candidate.get('mapping_profile', 'yaw') != profile):
            raise ValueError('comparison mapping profile differs from calibration')
        if 'joint_velocity_rad_s' in trial and trial['joint_velocity_rad_s'] is None:
            raise ValueError('joint velocity must be finite positive numbers')
        velocity = joint_velocity_metadata(
            trial.get('joint_velocity_rad_s'),
            config.get('output_limits', {}).get('joint_velocity_rad_s'),
            profile_name=trial.get('joint_velocity_profile'),
            joint_names=candidate.get('robot_joint_names'))
        speed = velocity['joint_velocity_rad_s']
        tracking = trial.get('reference_task_tracking', False)
        configured_tracking = config.get('solver', {}).get('reference_task_tracking', False)
        if (not isinstance(tracking, bool) or not isinstance(configured_tracking, bool)
            or tracking != configured_tracking):
            raise ValueError('comparison task tracking differs from calibration')
        include_final = trial.get('include_final', True)
        include_core = trial.get('include_core', True)
        if not isinstance(include_final, bool) or not isinstance(include_core, bool):
            raise ValueError('comparison include_final and include_core must be boolean')
        if not include_final and not include_core:
            raise ValueError('comparison must include a core or final replay')
        core_path = path.parent / trial["core"]
        core = load_core(core_path)
        final = load_final(path.parent / trial["reference"]) if include_final else None
        if 'root_z_offset_m' in trial:
            expected_offset = _root_z_offset(trial['root_z_offset_m'])
            if final is None or expected_offset != final.root_z_offset_m:
                raise ValueError('comparison root height offset differs from final reference')
        with np.load(core_path, allow_pickle=False) as archive:
            names = [str(name) for name in archive["target_names"]]
            times = archive["source_time_s"].astype(float)
            positions = archive["target_position_m"].astype(float)
        expected_names = list(target_names) if profile is None else list(candidate['target_frames'])
        if names != expected_names or len(names) != len(set(names)):
            raise ValueError("comparison target names differ from their calibration")
        if (times.shape != (len(core.qpos),) or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)
            or positions.shape != (len(times), len(names), 3) or not np.isfinite(positions).all()):
            raise ValueError("comparison target times or positions are invalid")
        if final is not None and (not np.isfinite(final.source_times).all()
            or np.min(final.source_times) < times[0]-.05
            or np.max(final.source_times) > times[-1]+.05):
            raise ValueError("comparison reference source times do not match its core")
        for replay, prefix in ((final, "最终 50 Hz"),
                               (core if include_core else None, "IK 输出")):
            if replay is None:
                continue
            label = (f'{prefix} · 限速 {speed:g} rad/s' if speed is not None else
                     f"{prefix} · 根位移 {scale:.2f}")
            if velocity['joint_velocity_profile'] is not None:
                label = prefix + (' · 旧SDK逐关节限速' if
                    velocity['joint_velocity_profile_name'] == 'legacy_sdk' else ' · 逐关节限速')
            if variant_label is not None:
                # Put the experimental difference before the shared settings so
                # it stays visible in the narrow browser dropdown.
                label = label.replace(prefix, f'{prefix} · {variant_label}', 1)
            if tracking:
                label += ' · 姿态修正'
            if profile == 'pitch':
                label += f' · {profile}映射 · 根位移 {scale:.2f}'
            if label in result:
                raise ValueError("duplicate comparison label")
            result[label] = {"replay": replace(replay, name=label, root_position_scale=scale,
                                              **velocity,
                                              reference_task_tracking=tracking,
                                              mapping_profile=candidate.get('mapping_profile', 'yaw'),
                                              **support, **compensation, **acceleration),
                             "target_times": times, "target_positions": positions,
                             'target_names': names,
                             'pn': load_pn_layers(core_path)}
    return result


def pn_display_scale(replay, calibration):
    """Keep a display-only PN comparison scale separate from algorithm scaling."""
    if getattr(replay, 'display_pn_scale', None) is not None:
        return replay.display_pn_scale
    return (replay.root_position_scale if replay.root_position_scale is not None
            else float(calibration['root_position_scale']))


def load_method_comparisons(manifest_path):
    """Load native legacy output without pretending it has GMR target metadata.

    The original SDK retarget qpos and the original published reference are
    intentionally separate: the latter includes the legacy -0.146 m z bias.
    Original PN samples are mandatory, and the violet layer's scale is solely
    a display comparison.  There is no fabricated orange IK target geometry.
    """
    path = Path(manifest_path)
    result = {}
    for trial in json.loads(path.read_text())['variants']:
        if trial.get('method') != 'legacy_euler':
            raise ValueError('unsupported comparison method')
        label = trial.get('label')
        if not isinstance(label, str) or not label.strip():
            raise ValueError('method comparison label must be a nonempty string')
        label = label.strip()
        scale = trial.get('display_scale', .78)
        if isinstance(scale, bool) or not isinstance(scale, (float, int)):
            raise ValueError('invalid display-only PN scale')
        scale = float(scale)
        if not np.isfinite(scale) or not .1 <= scale <= 2.:
            raise ValueError('invalid display-only PN scale')
        root_scale = trial.get('root_position_scale')
        if root_scale is not None:
            if isinstance(root_scale, bool) or not isinstance(root_scale, (float, int)):
                raise ValueError('invalid legacy SDK root scale metadata')
            root_scale = float(root_scale)
            if not np.isfinite(root_scale) or not .1 <= root_scale <= 2.:
                raise ValueError('invalid legacy SDK root scale metadata')
        core_path = path.parent / trial['core']
        core = load_core(core_path)
        with np.load(core_path, allow_pickle=False) as archive:
            if tuple(str(name) for name in archive['joint_names']) != tuple(ROBOT_JOINT_NAMES):
                raise ValueError('legacy joint order differs from the canonical RP1 order')
            # Old exporters may retain FK task geometry for diagnostics. If it
            # is present, validate it even though it is not an IK target layer.
            fields = {'target_names', 'target_position_m'}
            if fields.intersection(archive.files):
                if not fields.issubset(archive.files):
                    raise ValueError('incomplete legacy diagnostic target geometry')
                names = [str(name) for name in archive['target_names']]
                positions = np.asarray(archive['target_position_m'], dtype=float)
                if (len(names) != len(set(names)) or not names
                    or positions.shape != (len(core.qpos), len(names), 3)
                    or not np.isfinite(positions).all()):
                    raise ValueError('invalid legacy diagnostic target geometry')
        pn = load_pn_layers(core_path)
        if pn is None or len(pn.times) != len(core.qpos):
            raise ValueError('legacy comparison needs original PN geometry for every frame')
        if not np.array_equal(pn.times, core.source_times):
            raise ValueError('legacy PN source times differ from its retarget frames')
        replays = [(core, '原生retarget（未扣0.146m高度）')]
        if trial.get('reference') is not None:
            final = load_final(path.parent / trial['reference'],
                               allowed_pipelines={'legacy_euler_reference'})
            if (np.min(final.source_times) < pn.times[0]-.05
                or np.max(final.source_times) > pn.times[-1]+.05):
                raise ValueError('legacy reference source times do not match its core')
            replays.append((final, '原版发布reference（已扣0.146m）'))
        for replay, suffix in replays:
            name = f'{label} · {suffix}'
            if name in result:
                raise ValueError('duplicate method comparison label')
            replay = replace(replay, name=name, method='legacy_euler',
                             mapping_profile='sdk_euler', display_pn_scale=scale,
                             root_position_scale=root_scale,
                             reference_pipeline=('legacy_euler_core' if replay is core
                                                 else 'legacy_euler_reference'))
            result[name] = {'replay': replay, 'target_times': core.source_times,
                            'target_positions': None, 'target_names': [], 'pn': pn}
    return result


def pipeline_description(replay):
    return {'alignment_transition_only': 'reference 高度偏移 → 对齐 → 过渡 · 最终约束求解已移除',
            'constrained_qp': '历史 reference · 含最终约束求解',
            'core_ik': '核心 IK · 未进入发布处理',
            'legacy_euler_core': '原版 SDK 欧拉角 retarget · 原生输出（未扣 0.146 m 高度）',
            'legacy_euler_reference': '原版 SDK 发布流程 · 高度 −0.146 m → 初始对齐 → 2 秒过渡'}[replay.reference_pipeline]


def processing_note(replay):
    if getattr(replay, 'method', 'gmr') == 'legacy_euler':
        return ('**原版 SDK 欧拉角方法**\n\n原生 retarget 与旧发布 reference 分别显示；'
                '旧发布层扣除 0.146 m 高度，再执行原版初始对齐与过渡。'
                '不使用 GMR 的任务点、地板约束、6 rad/s 限速或腰 pitch 补偿配置。'
                f'紫色 {replay.display_pn_scale:g} 缩放只作人体显示对照，不代表 SDK 内部 FSC 缩放。')
    acceleration_note = ('核心 IK 的关节加速度限制及关联关节刹车包络已关闭；'
                         '关节速度、关节角度、根部限制和地板配置保持不变。\n\n'
                         if getattr(replay, 'joint_acceleration_enabled', None) is False else '')
    if replay.reference_pipeline == 'alignment_transition_only':
        return ('**当前 GMR reference：固定高度偏移 + 初始位置/朝向对齐 + 启动过渡**\n\n'
                + acceleration_note
                + height_offset_description(replay) + '\n\n'
                'retarget → 高度偏移 → 对齐 → 过渡 → 50 Hz 发布。最终层不再求解、限速或锁脚；'
                'IK 地板约束只作用于偏移前的 IK 输出。蓝/紫 PN 骨骼与橙色 IK 任务点不加发布高度偏移；'
                '启动过渡期间偏移也随过渡混合，完全跟随后才是固定偏移。')
    return '**GMR / mink 方法**\n\n' + acceleration_note + pipeline_description(replay)


def aligned_targets(positions, alignment):
    positions = np.array(positions, dtype=float, copy=True)
    if alignment is not None:
        sx, sy, syaw, rx, ry, ryaw = alignment
        c, s = np.cos(ryaw-syaw), np.sin(ryaw-syaw)
        delta = positions[:, :2] - [sx, sy]
        positions[:, 0] = rx + c*delta[:, 0] - s*delta[:, 1]
        positions[:, 1] = ry + s*delta[:, 0] + c*delta[:, 1]
    return positions


def display_translation(root_position, centered):
    """Translate only the displayed stage XY; never change reference/FK data."""
    root = np.asarray(root_position, dtype=float)
    return np.array([-root[0], -root[1], 0.]) if centered else np.zeros(3)


def initial_playback_time(seconds, duration):
    """Reject invalid inspection starts instead of silently clamping them."""
    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
        or not np.isfinite(seconds) or not 0. <= seconds <= duration):
        raise ValueError(f'--start-time-s must be finite and within 0..{duration:g} s')
    return float(seconds)


def initial_named_replay(replays, current, name):
    """An exact name can select an explicitly labelled experimental profile."""
    if name is None:
        return current
    if not isinstance(name, str) or name not in replays:
        raise ValueError(f'initial replay is not available: {name!r}')
    return replays[name]


def ground_trajectory(qpos):
    """Project actual reference root positions onto the fixed world ground."""
    points = np.zeros((len(qpos), 3), dtype=np.float32)
    points[:, :2] = np.asarray(qpos)[:, :2]
    points[:, 2] = .006
    return points


def trajectory_segments(points, end_index=None):
    points = np.asarray(points, dtype=np.float32)
    count = len(points) if end_index is None else max(0, min(int(end_index)+1, len(points)))
    if count < 2:
        return np.empty((0, 2, 3), dtype=np.float32)
    return np.stack((points[:count-1], points[1:count]), axis=1)


def world_camera(qpos, aspect=1.):
    """Fit the complete world path and robot envelope in a fixed perspective."""
    roots = np.asarray(qpos, dtype=float)[:, :3]
    low, high = roots.min(axis=0), roots.max(axis=0)
    low = low - [.65, .65, 0.]
    high = high + [.65, .65, 1.3]
    low[2] = min(0., low[2])
    return _camera_bounds(low, high, aspect)


def _camera_bounds(low, high, aspect):
    center = (low+high)/2.
    direction = np.array([1.6, -1.9, 1.05])
    direction /= np.linalg.norm(direction)
    right = np.cross(-direction, [0., 0., 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, -direction)
    corners = np.asarray([[x, y, z] for x in (low[0], high[0])
                          for y in (low[1], high[1]) for z in (low[2], high[2])])-center
    tan_vertical = np.tan(np.deg2rad(55.)/2.)
    tan_horizontal = tan_vertical * max(.1, float(aspect))
    distance = np.max(corners @ direction + np.maximum(
        np.abs(corners @ up)/tan_vertical, np.abs(corners @ right)/tan_horizontal))
    return center + direction*max(2., distance*1.08), center


def load_model(config_path, model_path=None):
    import mujoco

    config_path = Path(config_path).expanduser().resolve()
    config = json.loads(config_path.read_text())
    calibration = config["source_calibration"]
    model_path = resolve_model_path(model_path,
        configured_path=calibration["robot_model_path"], config_path=config_path)
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != calibration["robot_model_sha256"]:
        raise ValueError("RP1 model hash differs from the PN calibration")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if (model.nq, model.nv, model.nu) != (35, 34, 28):
        raise ValueError("expected the real RP1.3.2 floating-base 28-joint model")
    tpose = np.asarray(calibration["robot_tpose_qpos"], dtype=float)
    if tpose.shape != (35,) or calibration.get("robot_name") != "rp1.3.2":
        raise ValueError("expected the explicit standard RP1.3.2 T-pose calibration")
    return model, calibration


def load_scene_meshes(model, model_hash, cache_path, triangle_budget):
    """Reuse the validated CAD cache; still take every body transform from FK."""
    cache_path = Path(cache_path).expanduser()
    if triangle_budget == 120000 and cache_path.is_file():
        cache = json.loads(cache_path.read_text())
        if cache.get("model_sha256") == model_hash:
            mesh_by_id = {mesh["id"]: mesh for mesh in cache["meshes"]}
            records = []
            for geom in cache["geoms"]:
                gid = int(geom["id"])
                if (gid >= model.ngeom or int(model.geom_bodyid[gid]) != geom["body"]
                    or int(model.geom_dataid[gid]) != geom["mesh"]
                    or not np.allclose(model.geom_pos[gid], geom["pos"], atol=1e-10)
                    or not np.allclose(model.geom_quat[gid], geom["quat"], atol=1e-10)):
                    raise ValueError("CAD cache compiled geometry transforms differ from the RP1 model")
                mesh = mesh_by_id[geom["mesh"]]
                vertices = np.asarray(mesh["vertices"], dtype=np.float32)
                faces = np.asarray(mesh["faces"], dtype=np.uint32)
                if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
                    or faces.ndim != 2 or faces.shape[1] != 3 or faces.max(initial=0) >= len(vertices)):
                    raise ValueError("CAD cache has invalid mesh buffers")
                records.append({"geom_id":gid, "body_id":geom["body"],
                    "vertices":vertices, "faces":faces, "rgba":np.asarray(geom["rgba"])})
            return records
    return _build_visual_meshes(model, triangle_budget=triangle_budget)


def _build_visual_meshes(model, triangle_budget=120000):
    """Extract compiled CAD directly; no analysis scripts or bundled CAD cache.

    MuJoCo's compiled vertices already include the mesh asset transform. Only
    compiled geom_pos/geom_quat are applied by the viewer's body scene nodes.
    Decimation preserves the existing bounds check and full-detail fallback.
    """
    import mujoco

    meshes = []
    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh_id = int(model.geom_dataid[geom_id])
        va, vn = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
        fa, fn = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
        material = int(model.geom_matid[geom_id])
        rgba = model.mat_rgba[material] if material >= 0 else model.geom_rgba[geom_id]
        meshes.append({'geom_id': geom_id, 'body_id': int(model.geom_bodyid[geom_id]),
            'mesh_id': mesh_id,
            'name': mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id),
            'vertices': np.array(model.mesh_vert[va:va+vn], dtype=np.float32),
            'faces': np.array(model.mesh_face[fa:fa+fn], dtype=np.uint32),
            'original_vertex_count': vn, 'original_triangle_count': fn,
            'position': model.geom_pos[geom_id].copy(),
            'wxyz': model.geom_quat[geom_id].copy(),
            'rgba': np.array(rgba, dtype=np.float32)})
    if triangle_budget is None or not meshes:
        return meshes
    if triangle_budget < len(meshes)*32:
        raise ValueError('A triangle budget must allow at least 32 faces per mesh')
    import open3d as o3d

    weights = np.sqrt([mesh['original_triangle_count'] for mesh in meshes])
    targets = 32 + np.floor(weights/weights.sum()*(triangle_budget-32*len(meshes))).astype(int)
    for mesh, target in zip(meshes, targets):
        original = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh['vertices']),
                                            o3d.utility.Vector3iVector(mesh['faces']))
        original.remove_degenerate_triangles()
        original.remove_duplicated_triangles()
        original.remove_unreferenced_vertices()
        reduced = original.simplify_quadric_decimation(int(target))
        reduced.remove_degenerate_triangles()
        reduced.remove_duplicated_triangles()
        reduced.remove_unreferenced_vertices()
        vertices = np.asarray(reduced.vertices, dtype=np.float32).copy()
        source = np.asarray(original.vertices)
        error = float(np.max(np.abs(np.r_[source.min(0), source.max(0)]
                                   - np.r_[vertices.min(0), vertices.max(0)])))
        mesh['simplified_bounds_max_error_m'] = error
        if error > .002:
            mesh['simplification_fallback_full_detail'] = True
        else:
            mesh['vertices'] = vertices
            mesh['faces'] = np.asarray(reduced.triangles, dtype=np.uint32).copy()
    return meshes


class RP1Viewer:
    def __init__(self, args):
        # Drain the private publication pipe before importing the rendering
        # stack or loading CAD, so cold startup cannot accumulate old poses.
        self.ros_source = StdinReferenceSource() if args.source == 'stdin' else None
        import mujoco
        import viser

        self.mujoco = mujoco
        self.model, self.calibration = load_model(args.config, model_path=getattr(args, 'model', None))
        self.data = mujoco.MjData(self.model)
        self.live_view_only = args.source in ('ros', 'stdin')
        if self.live_view_only:
            replay, self.target_names, self.target_times, self.target_positions = live_replay(self.calibration)
            self.replays = {replay.name: replay}
        else:
            self.replays = {"最终 50 Hz reference": load_final(args.reference),
                            "IK 输出": load_core(args.core)}
            baseline_config = json.loads(Path(args.config).read_text())
            support = support_metadata(baseline_config)
            compensation = torso_compensation_metadata(baseline_config)
            acceleration = joint_acceleration_metadata(baseline_config)
            self.replays = {name: replace(replay, **support, **compensation, **acceleration)
                            for name, replay in self.replays.items()}
            with np.load(args.core, allow_pickle=False) as archive:
                self.target_times = archive["source_time_s"].astype(float)
                self.target_positions = archive["target_position_m"].astype(float)
                self.target_names = [str(name) for name in archive["target_names"]]
        baseline_scale = float(self.calibration["root_position_scale"])
        self.replays = {name: replace(replay, root_position_scale=baseline_scale,
                                     mapping_profile=self.calibration.get('mapping_profile', 'yaw'))
                        for name, replay in self.replays.items()}
        self.active = self.replays[args.initial_source]
        self.tpose = np.asarray(self.calibration["robot_tpose_qpos"], dtype=float)
        self.replay_targets = {name: (self.target_times, self.target_positions)
                               for name in self.replays}
        self.replay_target_names = {name: self.target_names for name in self.replays}
        baseline_pn = None if self.live_view_only else load_pn_layers(args.core)
        self.replay_pn = {name: baseline_pn for name in self.replays}
        self.pn_names = tuple(PN_JOINT_NAMES)
        self.pn_edges = pn_edges(self.pn_names)
        self.pn_tpose = np.asarray([self.calibration['source_tpose_world'][name][0]
                                   for name in self.pn_names], dtype=float)
        self.source_root_ref = np.asarray(self.calibration['source_tpose_world']['Hips'][0])
        self.robot_root_ref = np.asarray(self.calibration['robot_tpose_world']['base_link'][0])
        comparison_path = getattr(args, 'scale_comparison', None)
        if not self.live_view_only and comparison_path is not None and Path(comparison_path).is_file():
            for name, bundle in load_scale_comparisons(
                    comparison_path, self.calibration, self.target_names).items():
                self.replays[name] = bundle['replay']
                self.replay_targets[name] = (bundle['target_times'], bundle['target_positions'])
                self.replay_target_names[name] = bundle['target_names']
                self.replay_pn[name] = bundle['pn']
        method_path = getattr(args, 'method_comparison', None)
        if not self.live_view_only and method_path is not None:
            for name, bundle in load_method_comparisons(method_path).items():
                if name in self.replays:
                    raise ValueError('duplicate method comparison source label')
                self.replays[name] = bundle['replay']
                self.replay_targets[name] = (bundle['target_times'], bundle['target_positions'])
                self.replay_target_names[name] = bundle['target_names']
                self.replay_pn[name] = bundle['pn']
        initial_scale = getattr(args, 'initial_scale', None)
        if initial_scale is not None and getattr(args, 'initial_mapping_profile', None) is None:
            prefix = "IK 输出" if args.source == "core" else "最终 50 Hz"
            name = f"{prefix} · 根位移 {initial_scale:.2f}"
            if name not in self.replays:
                raise ValueError(f"scale comparison is not available: {initial_scale}")
            self.active = self.replays[name]
        initial_joint_speed = getattr(args, 'initial_joint_speed', None)
        if initial_joint_speed is not None:
            prefix = "IK 输出" if args.source == "core" else "最终 50 Hz"
            name = f'{prefix} · 限速 {initial_joint_speed:g} rad/s'
            if getattr(args, 'initial_task_tracking', False):
                name += ' · 姿态修正'
            profile = getattr(args, 'initial_mapping_profile', None)
            if profile is not None:
                if initial_scale is None:
                    raise ValueError('initial mapping profile needs --initial-scale')
                if profile == 'pitch':
                    name += f' · {profile}映射 · 根位移 {initial_scale:.2f}'
            if name not in self.replays:
                raise ValueError(f'joint velocity comparison is not available: {name}')
            self.active = self.replays[name]
            if profile is not None and not np.isclose(self.active.root_position_scale, initial_scale):
                raise ValueError('initial mapping trial scale differs from requested scale')
        self.active = initial_named_replay(self.replays, self.active,
                                          getattr(args, 'initial_replay_name', None))
        self.target_names = self._active_target_names()
        self.tpose_targets = self._target_tpose()
        self.lock = threading.RLock()
        self.seconds = initial_playback_time(getattr(args, 'start_time_s', 0.),
                                             self.active.duration)
        self.initial_centered = self.live_view_only or getattr(args, 'centered', False)
        self.playing = False
        if args.source == 'ros':
            self.ros_source = ROSReferenceSource(args.reference_topic)
        self.live_source_label = STDIN_SOURCE_LABEL if args.source == 'stdin' else ROS_SOURCE_LABEL
        self.ros_mode = self.ros_source is not None
        self.ros_playing = True
        self.ros_pose = self.active.qpos[0].copy()
        self.live_frame = None
        self.show_tpose = False
        self.seek_requested = None
        self.server = viser.ViserServer(host=args.host, port=args.port, label="RP1.3.2 · PN retarget 对比")
        if args.port and self.server.get_port() != args.port:
            self.server.stop()
            if self.ros_source is not None:
                self.ros_source.close()
            raise RuntimeError(f'Viser port {args.port} is busy; choose another --viser-port / --port')
        self.server.scene.set_up_direction("+z")
        self.server.scene.world_axes.visible = True
        self.server.scene.add_grid("/ground", width=12., height=12., plane="xy",
                                   cell_size=.2, section_size=1.)
        self.server.scene.configure_default_lights(cast_shadow=False)
        self.robot_frame = self.server.scene.add_frame("/robot", show_axes=False)
        self.body_nodes = {}
        for body_id in range(1, self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            self.body_nodes[body_id] = self.server.scene.add_frame(
                f"/robot/{name}", show_axes=False)
        meshes = load_scene_meshes(self.model, self.calibration["robot_model_sha256"],
                                  args.mesh_cache, args.triangle_budget)
        self.mesh_count = len(meshes)
        self.mesh_triangles = sum(len(mesh["faces"]) for mesh in meshes)
        self.mesh_nodes = []
        self.mesh_opacities = []
        for mesh in meshes:
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, mesh["body_id"])
            geom_id = mesh["geom_id"]
            rgba = np.clip(mesh["rgba"], 0., 1.)
            self.mesh_nodes.append(self.server.scene.add_mesh_simple(
                f"/robot/{name}/geom_{geom_id}", vertices=mesh["vertices"], faces=mesh["faces"],
                color=tuple(int(v * 255) for v in rgba[:3]),
                opacity=float(rgba[3]) * DEFAULT_ROBOT_OPACITY,
                wxyz=self.model.geom_quat[geom_id], position=self.model.geom_pos[geom_id],
                side="double", flat_shading=False, material="standard"))
            self.mesh_opacities.append(float(rgba[3]))
        self.overlay_frame = self.server.scene.add_frame("/targets", show_axes=False, visible=False)
        self.target_nodes = [self.server.scene.add_icosphere(
            f"/targets/{name}", radius=.014, color=(238, 151, 54), subdivisions=1)
            for name in self.target_names]
        self.target_lines = None
        self._drawn_target_names = tuple(self.target_names)
        self.overlay_edges = self._target_edges()
        self.pn_layer_nodes = {}
        for key, color in (('raw', RAW_PN_COLOR), ('scaled', SCALED_PN_COLOR)):
            frame = self.server.scene.add_frame(f'/pn_{key}', show_axes=False, visible=False)
            nodes = [self.server.scene.add_icosphere(f'/pn_{key}/{name}', radius=.012,
                     color=color, subdivisions=1) for name in self.pn_names]
            lines = self.server.scene.add_line_segments(f'/pn_{key}/bones',
                points=np.zeros((20, 2, 3), dtype=np.float32), colors=color,
                thickness=2., thickness_units='screen')
            self.pn_layer_nodes[key] = (frame, nodes, lines)
        self.pn_sample_info = None
        self._pn_camera_framed = False
        self.world_paths = {name: ground_trajectory(replay.qpos)
                            for name, replay in self.replays.items()}
        self.world_distances = {name: np.r_[0., np.cumsum(np.linalg.norm(
            np.diff(replay.qpos[:, :2], axis=0), axis=1))]
            for name, replay in self.replays.items()}
        self.path_frame = self.server.scene.add_frame('/world_path', show_axes=False)
        self.path_line = self.server.scene.add_line_segments('/world_path/full',
            points=trajectory_segments(self.world_paths[self.active.name]),
            colors=(133, 163, 192), thickness=2., thickness_units='screen')
        self.path_progress = self.server.scene.add_line_segments('/world_path/travelled',
            points=np.empty((0, 2, 3), dtype=np.float32),
            colors=(0, 164, 142), thickness=4., thickness_units='screen')
        self.path_marker = self.server.scene.add_icosphere('/world_path/current',
            radius=.045, color=(0, 164, 142), subdivisions=2)
        self.path_start = self.server.scene.add_label('/world_path/start', '起点',
            position=self.world_paths[self.active.name][0]+[0., 0., .035])
        self.path_state = None
        self._build_gui()

        @self.server.on_client_connect
        def _(client):
            self._camera(client)

        self._render()

    def _target_edges(self):
        return target_edges(self._active_target_names())

    def _active_target_names(self):
        return getattr(self, 'replay_target_names', {}).get(self.active.name, self.target_names)

    def _target_tpose(self):
        return np.asarray([self.calibration['robot_tpose_world'][name][0]
                           for name in self._active_target_names()], dtype=float)

    def _sync_target_nodes(self):
        names = tuple(self._active_target_names())
        if names == getattr(self, '_drawn_target_names', names):
            return
        for node in self.target_nodes:
            node.remove()
        self.target_nodes = [self.server.scene.add_icosphere(
            f'/targets/{name}', radius=.014, color=(238, 151, 54), subdivisions=1)
            for name in names]
        self.target_names = list(names)
        self.tpose_targets = self._target_tpose()
        self.overlay_edges = target_edges(names)
        self._drawn_target_names = names

    def _build_gui(self):
        gui = self.server.gui
        gui.configure_theme(control_width="medium", show_logo=False, show_share_button=False)
        gui.main_panel.dock_right()
        gui.add_markdown("**实际 RP1.3.2 模型 · 28 自由度**\n\nMuJoCo 正运动学 + CAD 网格；鼠标拖动旋转，滚轮缩放。")
        if self.ros_mode:
            gui.add_markdown('**实时发布 reference 可视化**\n\n'
                f"标定：{self.calibration.get('mapping_profile', 'yaw')} 映射 · 根位移比例 {self.calibration['root_position_scale']:.2f}。\n\n"
                '终端按 N 采集、A 跟随、X 回站、Esc 退出。可独立隐藏机器人或调节透明度。')
        self.processing_note = gui.add_markdown(processing_note(self.active)) if not self.ros_mode else None
        sources = (self.live_source_label,) if self.live_view_only else tuple(self.replays)
        self.source_control = gui.add_dropdown("数据源", options=sources,
            initial_value=self.live_source_label if self.ros_mode else self.active.name)
        self.play_control = gui.add_button("播放 / 暂停")
        self.time_control = gui.add_slider("回放时间 (s)", min=0., max=self.active.duration,
                                          step=.02, initial_value=self.seconds)
        self.speed_control = gui.add_dropdown("速度", options=("0.25×", "0.5×", "1×", "2×"),
                                             initial_value="1×")
        self.loop_control = gui.add_checkbox("循环", initial_value=True)
        self.center_control = gui.add_checkbox("机器人居中（XY）", initial_value=self.initial_centered)
        self.path_control = gui.add_checkbox("显示世界移动轨迹", initial_value=True)
        self.world_camera_control = gui.add_button("世界全景")
        self.robot_control = gui.add_checkbox("显示机器人", initial_value=True)
        self.opacity_control = gui.add_slider("机器人不透明度", min=0., max=1., step=.05,
                                             initial_value=DEFAULT_ROBOT_OPACITY)
        bone_heading = gui.add_markdown('**骨骼对照 · 可独立显示或叠加**')
        self.raw_pn_control = gui.add_checkbox('原始 PN 骨骼（蓝）',
            initial_value=isinstance(self.ros_source, StdinReferenceSource),
            hint='21 个原始 PN 关节点；仅单位、坐标系和共同显示对齐。')
        self.scaled_pn_control = gui.add_checkbox('仅统一缩放骨骼（紫）', initial_value=False,
            hint='21 点全部统一缩放；不替换 CAD 骨长，不做弯腰或脚部修正。')
        self.target_control = gui.add_checkbox("处理后目标骨骼（橙）", initial_value=True,
            hint=processed_target_hint(self.active))
        self.target_control.disabled = self.active.method == 'legacy_euler'
        self.pelvis_compare_control = gui.add_checkbox('骨盆对齐比较（仅显示）', initial_value=False,
            hint='将蓝/紫骨盆平移到橙色根部，隐藏根位移差异以比较姿态。可能使长腿穿过显示地面，不做地板修正。')
        self.pn_compare_control = gui.add_button('对照 · PN源15.02 s')
        self.layer_status = gui.add_markdown('')
        self.reset_control = gui.add_button("回到开头")
        self.inspection_controls = [
            (gui.add_button("弯腰 · 1.0 s"), 1.),
            (gui.add_button("深蹲 · 8.5 s"), 8.5),
            (gui.add_button("落脚 · 15.0 s"), 15.),
        ]
        self.fast_control = gui.add_button("快速动作 · 13.8 s")
        self.tpose_control = gui.add_button("标准 T-pose")
        self.camera_control = gui.add_button("重置视角")
        self.status = gui.add_markdown("")
        if self.live_view_only:
            for control in (self.time_control, self.speed_control, self.loop_control,
                            self.path_control, self.world_camera_control,
                            self.scaled_pn_control, self.target_control, self.pelvis_compare_control,
                            self.pn_compare_control, self.reset_control, self.fast_control,
                            *(control for control, _ in self.inspection_controls)):
                control.visible = False
            if isinstance(self.ros_source, StdinReferenceSource):
                self.pelvis_compare_control.visible = True
                gui.add_markdown('蓝色为最新原始 PN 骨骼，机器人为已发布的 retarget reference；二者源时间分别显示。PN 仅转换单位和坐标系，不缩放、不做骨长或足部修正。默认保留各自世界位置；骨盆对齐仅平移蓝色显示。XY 居中对两者施加相同平移。网格小格 0.2 m、大格 1 m。')
            else:
                bone_heading.visible = False
                self.raw_pn_control.visible = False
                gui.add_markdown('显示当前已发布的机器人 reference。网格小格 0.2 m、大格 1 m；XY 居中只改变显示，关闭后可观察世界位移。此视图不执行动力学仿真。')
        else:
            gui.add_markdown("世界模式中地面与坐标轴固定：灰蓝线为根部 XY 轨迹，绿色为已经过的路径。网格小格 0.2 m、大格 1 m。蓝色是原始 PN 21 点，紫色是统一缩放 21 点；橙色是 GMR 处理后的 14 个任务点，原版欧拉角方法禁用橙色层。各层共享源帧和世界显示变换；骨盆对齐选项只改变显示。此视图展示目标姿态，不执行动力学仿真。")
            if any(replay.method == 'gmr' and replay.root_z_offset_m != 0.
                   for replay in self.replays.values()):
                gui.add_markdown('高度偏移对比：只移动最终机器人 reference，IK 输出、原始/缩放 PN 与橙色 IK 任务点保持偏移前数据。IK 地板约束不保证偏移后的脚底高度；负偏移可能使脚底穿过地面。')
        if len(self.replays) > 2:
            gui.add_markdown(f"未带版本标签的两项使用本次启动所选配置，根位移比例 {self.calibration['root_position_scale']:.4f}；其他条目使用独立试验数据。GMR 系数仅调整根部 XYZ 位移，四肢骨长与机器人模型尺寸保持 RP1 CAD 定义；原版欧拉角条目保留 SDK 的原生结果与原版发布处理。")
        if any(replay.joint_velocity_rad_s is not None and replay.reference_pipeline == 'constrained_qp'
               for replay in self.replays.values()):
            gui.add_markdown("历史约束求解对照：带‘限速/姿态修正’的旧 reference 曾使用最终 QP 和 0.75 余量，仅留作比较。当前‘仅对齐与过渡’版本已移除该层。8 rad/s 历史试验未完成实际时钟检查，仅提供 IK 对照。")
        if not self.live_view_only and any(replay.mapping_profile == 'pitch' for replay in self.replays.values()):
            gui.add_markdown('pitch 映射试验：UpLeg → hip_pitch_link；Arm → shoulder_pitch_link；比例 0.78。每个版本使用自己的目标点和标准 T-pose 偏置，切换时保留相近 PN 源时间。pitch 刚体位于 roll/yaw 上游，方向误差会与其他任务折中。')

        @self.play_control.on_click
        def _(_):
            with self.lock:
                self.show_tpose = False
                if self.ros_mode:
                    self.ros_playing = not self.ros_playing
                else:
                    self.playing = not self.playing

        @self.time_control.on_update
        def _(event):
            if event.client is not None:
                with self.lock:
                    self.seek_requested = float(self.time_control.value)
                    self.show_tpose = False
                    self.playing = False

        @self.source_control.on_update
        def _(_):
            with self.lock:
                previous_seconds = self.seconds if self.seek_requested is None else self.seek_requested
                source_seconds = self.active.source_times[self.active.index(previous_seconds)]
                self.ros_mode = self.source_control.value == self.live_source_label
                if not self.ros_mode:
                    self.active = self.replays[self.source_control.value]
                    self.target_control.hint = processed_target_hint(self.active)
                    self.target_control.disabled = self.active.method == 'legacy_euler'
                    if self.processing_note is not None:
                        self.processing_note.content = processing_note(self.active)
                    index = int(np.argmin(np.abs(self.active.source_times-source_seconds)))
                    self.seconds = float(self.active.times[index])
                    self.seek_requested = None
                self.time_control.max = self.active.duration
                self.seconds = min(self.seconds, self.active.duration)
                self.show_tpose = False
            self._reset_cameras()

        @self.center_control.on_update
        def _(_):
            self._reset_cameras()

        @self.world_camera_control.on_click
        def _(_):
            self.center_control.value = False
            self.path_control.value = True
            self._reset_cameras()

        @self.robot_control.on_update
        def _(_):
            self._update_robot_appearance()

        @self.opacity_control.on_update
        def _(_):
            self._update_robot_appearance()

        @self.pelvis_compare_control.on_update
        def _(_):
            self._reset_cameras()

        def frame_pn_once(_):
            with self.lock:
                if (self.raw_pn_control.value or self.scaled_pn_control.value) and not self._pn_camera_framed:
                    if self.ros_mode and not self.show_tpose and self._live_pn() is None:
                        return
                    self._pn_camera_framed = True
                    self._reset_cameras()
        self.raw_pn_control.on_update(frame_pn_once)
        self.scaled_pn_control.on_update(frame_pn_once)

        @self.pn_compare_control.on_click
        def _(_):
            self._seek_source(15.02, nearest=True)

        @self.reset_control.on_click
        def _(_):
            with self.lock:
                self.seek_requested = 0.
                self.show_tpose = False
                self.playing = False
                self.ros_playing = False
            self._reset_cameras()

        @self.fast_control.on_click
        def _(_):
            self._seek_source(13.8)

        for control, source_seconds in self.inspection_controls:
            control.on_click(lambda _, seconds=source_seconds: self._seek_source(seconds))

        @self.tpose_control.on_click
        def _(_):
            with self.lock:
                self.show_tpose = True
                self.playing = False
                self.ros_playing = False
            self._reset_cameras()

        @self.camera_control.on_click
        def _(_):
            self._reset_cameras()

    def _update_robot_appearance(self):
        """Update CAD display only; PN targets and reference poses stay independent."""
        with self.lock, self.server.atomic():
            self.robot_frame.visible = bool(self.robot_control.value)
            opacity = float(self.opacity_control.value)
            for node, material_opacity in zip(self.mesh_nodes, self.mesh_opacities):
                node.opacity = material_opacity * opacity

    def _seek_source(self, seconds, nearest=False):
        """Pause at a PN source timestamp, preserving the recorded publication timeline."""
        with self.lock:
            if self.ros_mode:
                return
            index = (int(np.argmin(np.abs(self.active.source_times-seconds))) if nearest else
                     int(np.searchsorted(self.active.source_times, seconds, side="left")))
            self.seek_requested = float(self.active.times[min(index, len(self.active.times)-1)])
            self.show_tpose = False
            self.playing = False
        self._reset_cameras()

    def _reset_cameras(self):
        if not (getattr(getattr(self, 'raw_pn_control', None), 'value', False)
                or getattr(getattr(self, 'scaled_pn_control', None), 'value', False)):
            self._pn_camera_framed = False
        for client in self.server.get_clients().values():
            self._camera(client)

    def _update_live_sample(self):
        """Freeze or replace both layers together, without another subscription."""
        if not self.ros_mode or not self.ros_playing or self.show_tpose:
            return
        if isinstance(self.ros_source, StdinReferenceSource):
            frame, _, _ = self.ros_source.sample_frame()
            if frame is not None:
                self.live_frame = frame
                self.ros_pose = frame['qpos']
                if (self._live_pn() is not None and self.raw_pn_control.value
                        and not self._pn_camera_framed):
                    self._pn_camera_framed = True
                    self._reset_cameras()
        else:
            latest, _, _ = self.ros_source.sample()
            if latest is not None:
                self.ros_pose = latest[0]

    def _live_pn(self):
        frame = getattr(self, 'live_frame', None)
        pn = None if frame is None else frame['pn']
        if pn is not None and (not self.ros_playing or
                time.monotonic()-pn['received_monotonic_s'] <= LIVE_PN_TIMEOUT_S):
            return pn
        return None

    def _skeleton_positions(self, index):
        """Recorded layers share a sample; live PN retains its source world pose."""
        alignment, sample_time, source_index = None, None, None
        if self.show_tpose:
            processed = (None if getattr(self.active, 'method', 'gmr') == 'legacy_euler' else
                         self._target_tpose() if hasattr(self, 'replay_target_names')
                         else self.tpose_targets.copy())
            raw = getattr(self, 'pn_tpose', None)
        elif getattr(self, 'ros_mode', False):
            pn = self._live_pn()
            if pn is None:
                return None, None, None, None
            raw = pn['positions_m'].copy()
            if self.pelvis_compare_control.value:
                raw += self.ros_pose[:3]-raw[self.pn_names.index('Hips')]
            return None, raw, None, {'source_time_s': pn['source_time_s'],
                                    'source_index': pn['seq'],
                                    'reference_source_time_s': self.live_frame['reference_source_time_s']}
        else:
            times, positions = getattr(self, 'replay_targets', {}).get(
                self.active.name, (self.target_times, self.target_positions))
            target_index = causal_index(times, self.active.source_times[index])
            processed = None if positions is None else positions[target_index].copy()
            alignment = None if self.active.alignment is None else self.active.alignment[index]
            pn = getattr(self, 'replay_pn', {}).get(self.active.name)
            raw = None
            if pn is not None:
                order = [pn.names.index(name) for name in self.pn_names]
                raw = pn.positions[target_index, order].copy()
                sample_time = float(pn.times[target_index])
                source_index = None if pn.source_indices is None else int(pn.source_indices[target_index])
        processed = None if processed is None else aligned_targets(processed, alignment)
        if raw is None:
            return processed, None, None, None
        scale = pn_display_scale(self.active, getattr(self, 'calibration', {}))
        scaled = scale_only_pn(raw, scale, self.source_root_ref, self.robot_root_ref)
        raw = aligned_targets(raw, alignment)
        scaled = aligned_targets(scaled, alignment)
        if getattr(getattr(self, 'pelvis_compare_control', None), 'value', False):
            pelvis = (processed[self._active_target_names().index('base_link')]
                      if processed is not None else
                      self.tpose[:3] if self.show_tpose else self.active.qpos[index, :3])
            hip = self.pn_names.index('Hips')
            raw += pelvis-raw[hip]
            scaled += pelvis-scaled[hip]
        info = {'source_time_s': sample_time, 'source_index': source_index}
        return processed, raw, scaled, info

    def _camera(self, client, root=None):
        seconds = self.seconds if self.seek_requested is None else self.seek_requested
        index = self.active.index(seconds)
        if root is None:
            root = self.tpose[:3] if self.show_tpose else (
                self.ros_pose[:3] if self.ros_mode else self.active.qpos[index, :3])
        try:
            aspect = client.camera.aspect
        except (AssertionError, AttributeError):
            aspect = 1.
        inspect_pn = (getattr(getattr(self, 'raw_pn_control', None), 'value', False)
                      or getattr(getattr(self, 'scaled_pn_control', None), 'value', False))
        if not self.center_control.value and not self.ros_mode and not self.show_tpose:
            # Keep an identical metric camera across scale variants of this
            # source family, so auto-framing cannot conceal the scale change.
            def family(replay):
                return ('core' if replay.reference_pipeline in ('core_ik', 'legacy_euler_core')
                        else 'final', replay.method)
            replays = getattr(self, 'replays', {self.active.name: self.active})
            roots = [replay.qpos[:, :3] for replay in replays.values()
                     if family(replay) == family(self.active)]
            pn = getattr(self, 'replay_pn', {}).get(self.active.name)
            if inspect_pn and pn is not None and not self.pelvis_compare_control.value:
                hips = pn.positions[:, pn.names.index('Hips')]
                scaled = scale_only_pn(hips, pn_display_scale(self.active, self.calibration),
                                       self.source_root_ref, self.robot_root_ref)
                alignment = None if self.active.alignment is None else self.active.alignment[0]
                roots.extend([aligned_targets(hips, alignment), aligned_targets(scaled, alignment)])
            positions = np.concatenate(roots)
            position, center = world_camera(positions, aspect)
        elif inspect_pn:
            processed, raw, scaled, _ = self._skeleton_positions(index)
            shift = display_translation(root, self.center_control.value)
            clouds = [points+shift for points in (processed, raw, scaled) if points is not None]
            # Include an envelope for the CAD robot as well as all three layers,
            # retaining a fixed camera while individual checkboxes are switched.
            origin = np.asarray(root)+shift
            clouds.append(origin + np.asarray([[x, y, z] for x in (-.65, .65)
                          for y in (-.65, .65) for z in (-.8, 1.)]))
            points = np.concatenate(clouds)
            position, center = _camera_bounds(points.min(0)-.15, points.max(0)+.15, aspect)
        else:
            root = np.asarray(root) + display_translation(root, self.center_control.value)
            center = root + [0., 0., -.1]
            position = center + [1.6, -1.9, 1.05]
        with client.atomic():
            client.camera.position = position
            client.camera.look_at = center
            client.camera.up_direction = (0., 0., 1.)
            client.camera.fov = np.deg2rad(55.)

    def _render(self):
        index = self.active.index(self.seconds)
        qpos = self.tpose if self.show_tpose else (self.ros_pose if self.ros_mode else self.active.qpos[index])
        self.data.qpos[:] = qpos
        self.mujoco.mj_forward(self.model, self.data)
        with self.server.atomic():
            self._sync_target_nodes()
            shift = display_translation(qpos[:3], self.center_control.value)
            self.robot_frame.position = shift
            self.overlay_frame.position = shift
            for body_id, node in self.body_nodes.items():
                node.position = self.data.xpos[body_id]
                node.wxyz = self.data.xquat[body_id]
            self.overlay_frame.visible = (self.target_control.value
                and getattr(self.active, 'method', 'gmr') != 'legacy_euler'
                and (not self.ros_mode or self.show_tpose))
            positions, raw, scaled, info = self._skeleton_positions(index)
            self.pn_sample_info = info
            if self.overlay_frame.visible:
                for node, position in zip(self.target_nodes, positions):
                    node.position = position
                segments = np.asarray([[positions[a], positions[b]] for a,b in self.overlay_edges], dtype=np.float32)
                if len(segments):
                    if self.target_lines is None:
                        self.target_lines = self.server.scene.add_line_segments(
                            "/targets/bones", points=segments, colors=(238,151,54), line_width=2.)
                    else:
                        self.target_lines.points = segments
            if hasattr(self, 'pn_layer_nodes'):
                for key, points, control in (('raw', raw, self.raw_pn_control),
                                            ('scaled', scaled, self.scaled_pn_control)):
                    frame, nodes, lines = self.pn_layer_nodes[key]
                    frame.position = shift
                    frame.visible = bool(control.value and points is not None)
                    if frame.visible:
                        for node, position in zip(nodes, points):
                            node.position = position
                        lines.points = np.asarray([[points[a], points[b]] for a, b in self.pn_edges],
                                                  dtype=np.float32)
            self.time_control.value = min(self.seconds, self.active.duration)
            if hasattr(self, 'path_frame'):
                self._render_world_path(qpos, index)
        return index

    def _render_world_path(self, qpos, index):
        visible = (self.path_control.value and not self.center_control.value
                   and not self.ros_mode and not self.show_tpose)
        self.path_frame.visible = visible
        if not visible:
            return
        points = self.world_paths[self.active.name]
        # Coalesce five recording samples while playing; paused seeks stay exact.
        state = (self.active.name, index//5, index if not self.playing else None,
                 index == len(points)-1)
        if self.path_state is None or self.path_state[0] != self.active.name:
            self.path_line.points = trajectory_segments(points)
            self.path_start.position = points[0]+[0., 0., .035]
        if state != self.path_state:
            self.path_progress.points = trajectory_segments(points, index)
            self.path_state = state
        self.path_marker.position = (float(qpos[0]), float(qpos[1]), .012)

    def run(self, duration=None):
        started = previous = time.monotonic()
        last_status = 0.
        try:
            while duration is None or time.monotonic()-started < duration:
                now = time.monotonic()
                if isinstance(self.ros_source, StdinReferenceSource) and self.ros_source.eof.is_set():
                    break
                with self.lock:
                    self._update_live_sample()
                    self.time_control.disabled = self.ros_mode
                    self.speed_control.disabled = self.ros_mode
                    self.loop_control.disabled = self.ros_mode
                    self.reset_control.disabled = self.ros_mode
                    self.fast_control.disabled = self.ros_mode
                    for control, _ in self.inspection_controls:
                        control.disabled = self.ros_mode
                    self.target_control.disabled = (self.active.method == 'legacy_euler'
                                                    or self.ros_mode and not self.show_tpose)
                    pn_available = (self.show_tpose or
                        (self.ros_mode and self._live_pn() is not None) or
                        (not self.ros_mode and self.replay_pn.get(self.active.name) is not None))
                    self.raw_pn_control.disabled = not pn_available
                    self.scaled_pn_control.disabled = not pn_available
                    self.pelvis_compare_control.disabled = not pn_available
                    self.pn_compare_control.disabled = self.ros_mode
                    self.path_control.disabled = self.ros_mode or self.show_tpose
                    self.world_camera_control.disabled = self.ros_mode or self.show_tpose
                    if self.seek_requested is not None:
                        self.seconds, self.seek_requested = self.seek_requested, None
                    elif self.playing and not self.ros_mode:
                        self.seconds += (now-previous)*float(self.speed_control.value.rstrip("×"))
                        if self.seconds > self.active.duration:
                            if self.loop_control.value:
                                self.seconds %= self.active.duration
                            else:
                                self.seconds = self.active.duration
                                self.playing = False
                    index = self._render()
                    if now-last_status > .15:
                        if self.show_tpose:
                            label = "标准 T-pose · 固定标定值"
                        elif self.ros_mode:
                            latest, received, rejected = self.ros_source.sample()
                            age = "等待数据" if latest is None else f"接收距今 {max(0., now-latest[1])*1000:.0f} ms"
                            label = f"{'实时显示' if self.ros_playing else '画面已暂停'} · {age}\n\n已接收 {received} · 拒绝 {rejected}"
                        else:
                            label = (f"{'播放中' if self.playing else '已暂停'} · 源时间 {self.active.source_times[index]:.2f} s"
                                + (" · 跟随过渡" if self.active.blend[index] else ""))
                        coordinates = "XY 居中显示 · 实际高度" if self.center_control.value else "固定世界坐标 · 实际 reference 位移"
                        root = self.data.qpos[:3]
                        movement = f"根部世界 XYZ：{root[0]:+.2f}, {root[1]:+.2f}, {root[2]:.2f} m"
                        if not self.ros_mode and self.active.method == 'legacy_euler':
                            scale_label = ('原版 SDK / FSC 结果' if self.active.root_position_scale is None else
                                           f'旧SDK根位移比例（实测）{self.active.root_position_scale:.7f}')
                            movement = f'根位移：{scale_label}；未套用 GMR 的 0.78 系数\n\n'+movement
                        elif not self.ros_mode and self.active.root_position_scale is not None:
                            movement = f"根位移比例 {self.active.root_position_scale:.4f} · RP1 固定骨长\n\n"+movement
                            movement = (f'大腿 → hip_{self.active.mapping_profile}_link；'
                                        f'上臂 → shoulder_{self.active.mapping_profile}_link\n\n')+movement
                        if not self.ros_mode:
                            velocity = joint_velocity_description(self.active)
                            if velocity:
                                movement = velocity+(' · 保留任务目标' if self.active.reference_task_tracking else '')+'\n\n'+movement
                        if not self.ros_mode:
                            processing = pipeline_description(self.active)
                            movement = processing+'\n\n'+movement
                            height = height_offset_description(self.active)
                            if height:
                                movement = height+'\n\n'+movement
                            support = support_description(self.active)
                            if support:
                                movement = support+'\n\n'+movement
                            compensation = torso_compensation_description(self.active)
                            if compensation:
                                movement = compensation+'\n\n'+movement
                        if not self.ros_mode and not self.show_tpose:
                            delta = root-self.active.qpos[0, :3]
                            movement += (f"\n\n相对起点 ΔXYZ：{delta[0]:+.2f}, {delta[1]:+.2f}, {delta[2]:+.2f} m"
                                f" · 已走 XY 路径 {self.world_distances[self.active.name][index]:.2f} m")
                        self.status.content = f"{label}\n\n{coordinates}\n\n{movement}\n\n{self.mesh_count} 个真实网格 · {self.mesh_triangles:,} 三角面"
                        if self.show_tpose:
                            self.layer_status.content = ('标准 T-pose：PN 使用解析标准姿态；原版欧拉角条目禁用橙色层。'
                                if self.active.method == 'legacy_euler' else
                                '标准 T-pose：PN 使用解析标准姿态，橙色使用固定 RP1 标定值。')
                        elif self.ros_mode:
                            sample = self.pn_sample_info
                            if sample is None:
                                self.layer_status.content = ('原始 PN 骨骼：等待采集或数据已过期。机器人继续显示最新已发布的 reference。'
                                    if isinstance(self.ros_source, StdinReferenceSource) else
                                    '当前 ROS reference 模式只接收机器人数据。')
                            else:
                                reference_time = sample['reference_source_time_s']
                                reference_label = '等待首个求解结果' if reference_time is None else f'{reference_time:.6f} s'
                                self.layer_status.content = (
                                    f"最新 PN：{sample['source_time_s']:.6f} s · 帧 {sample['source_index']}"
                                    f'\n\nreference 对应 PN 源时间：{reference_label}'
                                    '\n\n两层使用各自最新数据，可能来自不同采样；原始 PN 未缩放。'
                                    + ('\n\n蓝色骨盆已平移到机器人根部，仅影响显示。'
                                       if self.pelvis_compare_control.value else '\n\n保留各自原世界位置。'))
                        elif self.pn_sample_info is not None:
                            sample = self.pn_sample_info
                            stamp = sample['source_time_s']
                            frame = '' if sample['source_index'] is None else f" · 原始帧 {sample['source_index']}"
                            alignment_label = ('骨盆已对齐：蓝/紫层的根位移差异仅在显示中隐藏。'
                                               if self.pelvis_compare_control.value else
                                               '共同世界坐标：保留各层根部位置与位移差异。')
                            display_scale = pn_display_scale(self.active, self.calibration)
                            scale_label = (f'紫色统一缩放比例 {display_scale:.4f}，仅用于显示比较，不是 SDK retarget 系数；'
                                           '蓝/紫均为 21 点，橙色层禁用。'
                                           if self.active.method == 'legacy_euler' else
                                           f'紫色统一缩放比例 {display_scale:.4f}；蓝/紫 21 点，橙 14 点。')
                            self.layer_status.content = (
                                f"同一 PN 采样：{stamp:.6f} s{frame}"
                                f"\n\nreference 源时间 {self.active.source_times[index]:.6f} s"
                                f"\n\n{scale_label}"
                                f"\n\n{alignment_label}")
                        else:
                            self.layer_status.content = '这份旧回放未保存原始 PN 点，只显示处理后目标。'
                        last_status = now
                previous = now
                time.sleep(max(0., .02-(time.monotonic()-now)))
        finally:
            if self.ros_source is not None:
                self.ros_source.close()
            self.server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8890)
    direct_available = (DIRECT_REFERENCE_AUDIT / 'reference_replay.jsonl').is_file()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument('--model', type=Path, default=None,
                        help='relocated RP1 MJCF; otherwise PN_RP1_MODEL or calibration path')
    parser.add_argument("--reference", type=Path, default=(DIRECT_REFERENCE_AUDIT / 'reference_replay.jsonl'
                                                         if direct_available else DEFAULT_REFERENCE))
    parser.add_argument("--core", type=Path, default=(DIRECT_REFERENCE_AUDIT / 'replay.npz'
                                                    if direct_available else DEFAULT_CORE))
    parser.add_argument("--scale-comparison", type=Path, default=DEFAULT_SCALE_COMPARISON,
                        help="optional manifest of separately evaluated root-scale trials")
    parser.add_argument('--method-comparison', type=Path, default=None,
                        help='optional native legacy-Euler core/reference comparison manifest')
    parser.add_argument("--initial-scale", type=float, default=None,
                        help="start on a trial, e.g. 0.78 or 0.8; leave unset for the baseline")
    parser.add_argument('--initial-joint-speed', type=float, default=None,
                        help='start on an evaluated joint-velocity trial')
    parser.add_argument('--initial-replay-name', default=None,
                        help='exact replay source label; overrides initial scale/speed selection')
    parser.add_argument('--initial-task-tracking', action='store_true',
                        help='select the trial with final Cartesian task tracking')
    parser.add_argument('--initial-mapping-profile', choices=('yaw', 'pitch'), default=None,
                        help='select a mapping trial with --initial-scale and --initial-joint-speed')
    parser.add_argument("--source", choices=("final", "core", "ros", "stdin"), default="final")
    parser.add_argument('--start-time-s', type=float, default=0.,
                        help='paused initial playback time within the selected replay')
    parser.add_argument('--centered', action='store_true',
                        help='initially center display XY while retaining actual world height')
    parser.add_argument("--reference-topic", default="/tracking/current_reference")
    parser.add_argument("--duration-s", type=float, default=None,
                        help="bounded server lifetime for smoke tests")
    parser.add_argument("--triangle-budget", type=int, default=120000,
                        help="CAD mesh triangle budget; 0 keeps all original CAD detail")
    parser.add_argument("--mesh-cache", type=Path,
                        default=default_mesh_cache(AUDIT / 'rp1_web_model.json'),
                        help='optional CAD cache; defaults to PN_RP1_MESH_CACHE or the user cache')
    args = parser.parse_args()
    if args.config is None:
        args.config = (DEFAULT_LIVE_CONFIG if args.source in ('ros', 'stdin') else
                       DIRECT_REFERENCE_AUDIT / 'config.json' if direct_available else DEFAULT_CONFIG)
    if args.initial_mapping_profile is not None and (args.initial_scale is None
                                                    or args.initial_joint_speed is None):
        parser.error('--initial-mapping-profile needs --initial-scale and --initial-joint-speed')
    if args.triangle_budget == 0:
        args.triangle_budget = None
    args.initial_source = "IK 输出" if args.source == "core" else "最终 50 Hz reference"
    viewer = RP1Viewer(args)
    print(json.dumps({"url":f"http://{args.host}:{viewer.server.get_port()}/",
        "model":"rp1.3.2", "meshes":viewer.mesh_count, "triangles":viewer.mesh_triangles,
        "source":viewer.live_source_label if viewer.ros_mode else viewer.active.name, "frames":len(viewer.active.times),
        "duration_s":viewer.active.duration, "standard_tpose":True,
        "start_time_s":viewer.seconds, "root_z_offset_m":viewer.active.root_z_offset_m,
        "centered":bool(viewer.center_control.value), "playing":viewer.playing},
        ensure_ascii=False), flush=True)
    try:
        viewer.run(args.duration_s)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
