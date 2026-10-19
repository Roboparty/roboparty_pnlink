#!/usr/bin/env python3
"""Replay a saved PN take through the bounded RP1 retargeter without ROS.

Uses reception timestamps and the most recent available sample, never a future
sample or interpolation. The resulting measurements are offline CPU/FK
measurements, not policy tracking accuracy or device-to-actuator latency.

Run with the Python runtime containing the IK dependencies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "outputs/demo_pn.npz"
DEFAULT_CONFIG = ROOT / "mocapapi_python/rp1/pn_axis_to_rp1_28dof_current.json"
DEFAULT_OUTPUT = ROOT / "outputs/offline"


def percentiles(values):
    a = np.asarray(values, dtype=float).reshape(-1)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"count": 0}
    return {"count": int(len(a)), "mean": float(a.mean()),
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
            "min": float(a.min()), "max": float(a.max())}


def causal_samples(source_time, rate_hz=50., start_s=None, duration_s=None):
    """Most recent received sample at every output tick, including exact ticks."""
    source_time = np.asarray(source_time, dtype=float)
    if source_time.ndim != 1 or len(source_time) < 2 or not np.isfinite(source_time).all() or np.any(np.diff(source_time) <= 0):
        raise ValueError("source timestamps must be strictly increasing")
    if not np.isfinite(rate_hz) or rate_hz <= 0:
        raise ValueError("rate_hz must be positive and finite")
    start = float(source_time[0] if start_s is None else start_s)
    end = float(source_time[-1])
    if duration_s is not None:
        end = min(end, start + duration_s)
    if start < source_time[0] or start > end:
        raise ValueError("requested interval is outside the recording")
    ticks = start + np.arange(int(np.floor((end - start) * rate_hz + 1e-8)) + 1) / rate_hz
    indices = np.searchsorted(source_time, ticks, side="right") - 1
    if np.any(indices < 0) or np.any(source_time[indices] > ticks):
        raise AssertionError("non-causal sample selection")
    return ticks, indices


def load_recording(path):
    with np.load(path, allow_pickle=False) as z:
        return {name: z[name].copy() for name in z.files}


def world_pose(recording, index):
    names = recording["joint_names"].tolist()
    pos = recording["global_position_ros_m"][index]
    quat = recording["global_rotation_ros_xyzw"][index][..., [3, 0, 1, 2]]
    return {n: (pos[j].copy(), quat[j].copy()) for j, n in enumerate(names)}


def foot_geometry(model, data):
    """Exact lowest point of each of the RP1 foot collision capsules.

    Capsule endpoint/sphere geometry is used at the actual output orientation.
    Sole centers are ankle-body local (0.0225, 0, -0.045), useful for sliding
    comparisons; minimum height uses all seven capsules, not this center.
    """
    import mujoco
    heights, centers = [], []
    for side in ("left", "right"):
        lower = []
        for gid in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if name.startswith(f"{side}_foot_") and name.endswith("_collision"):
                if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_CAPSULE:
                    raise ValueError(f"expected capsule geometry: {name}")
                radius, half_length = model.geom_size[gid, :2]
                axis_z = data.geom_xmat[gid].reshape(3, 3)[2, 2]
                lower.append(data.geom_xpos[gid, 2] - abs(axis_z) * half_length - radius)
        if not lower:
            raise ValueError(f"RP1 {side} foot collision geometry not found")
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link")
        center = data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ np.array([.0225, 0., -.045])
        heights.append(min(lower))
        centers.append(center)
    return np.asarray(heights), np.asarray(centers)


def frame_fk(model, data, qpos, targets):
    import mujoco
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    names = list(targets)
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in names]
    if any(i < 0 for i in ids):
        raise ValueError("last_targets must use actual robot body names")
    pos = data.xpos[ids].copy()
    quat = data.xquat[ids].copy()
    target_pos = np.asarray([targets[n][0] for n in names])
    target_quat = np.asarray([targets[n][1] for n in names])
    dots = np.abs(np.sum(quat * target_quat, axis=-1))
    rot = np.degrees(2 * np.arccos(np.clip(dots, 0., 1.)))
    sole_height, sole_center = foot_geometry(model, data)
    return pos, quat, target_pos, target_quat, np.linalg.norm(pos - target_pos, axis=-1), rot, sole_height, sole_center


def output_derivatives(model, qposes, dt):
    import mujoco
    velocity = np.empty((len(qposes) - 1, model.nv))
    for j in range(len(velocity)):
        mujoco.mj_differentiatePos(model, velocity[j], dt, qposes[j], qposes[j + 1])
    return velocity, np.diff(velocity, axis=0) / dt


def slide_metrics(centers, contact, dt):
    result = {}
    for side_index, side in enumerate(("left", "right")):
        stance = contact[:, side_index].astype(bool)
        consecutive = stance[:-1] & stance[1:]
        increments = np.linalg.norm(np.diff(centers[:, side_index, :2], axis=0), axis=-1)
        speed = increments[consecutive] / dt
        segment_drifts, segment_lengths = [], []
        begins = np.flatnonzero(stance & ~np.r_[False, stance[:-1]])
        ends = np.flatnonzero(stance & ~np.r_[stance[1:], False])
        for start, end in zip(begins, ends):
            if end - start < 2:
                continue
            segment_drifts.append(float(np.max(np.linalg.norm(
                centers[start:end + 1, side_index, :2] - centers[start, side_index, :2], axis=-1))))
            segment_lengths.append(float((end - start + 1) * dt))
        result[side] = {"contact_frames": int(stance.sum()),
                        "consecutive_contact_steps": int(consecutive.sum()),
                        "stance_speed_m_s": percentiles(speed),
                        "stance_segment_max_displacement_m": percentiles(segment_drifts),
                        "stance_segment_duration_s": percentiles(segment_lengths),
                        "stance_segments": int(len(begins))}
    return result


def quality_metrics(model, qpos, position_error, rotation_error, names,
                    sole_height, sole_center, contacts, dt, limits=None):
    velocity, accel = output_derivatives(model, qpos, dt)
    import mujoco
    joint_ids = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    qadr = model.jnt_qposadr[joint_ids]
    lo, hi = model.jnt_range[joint_ids].T
    limit_error = np.maximum(lo - qpos[:, qadr], 0.) + np.maximum(qpos[:, qadr] - hi, 0.)
    result = {
        "position_error_m": percentiles(position_error),
        "rotation_error_deg": percentiles(rotation_error),
        "per_body": {name: {"position_error_m": percentiles(position_error[:, j]),
                             "rotation_error_deg": percentiles(rotation_error[:, j])}
                     for j, name in enumerate(names)},
        "finite_output": bool(np.isfinite(qpos).all()),
        "root_quaternion_norm_max_error": float(np.max(np.abs(np.linalg.norm(qpos[:, 3:7], axis=1) - 1.))),
        "joint_limit_violations": int(np.count_nonzero(limit_error > 1e-7)),
        "joint_limit_max_excess_rad": float(limit_error.max()),
        "joint_velocity_rad_s": percentiles(np.abs(velocity[:, 6:])),
        "joint_acceleration_rad_s2": percentiles(np.abs(accel[:, 6:])),
        "root_translation_velocity_m_s": percentiles(np.linalg.norm(velocity[:, :3], axis=-1)),
        "root_rotation_velocity_rad_s": percentiles(np.linalg.norm(velocity[:, 3:6], axis=-1)),
        "root_translation_acceleration_m_s2": percentiles(np.linalg.norm(accel[:, :3], axis=-1)),
        "root_rotation_acceleration_rad_s2": percentiles(np.linalg.norm(accel[:, 3:6], axis=-1)),
        "foot_geometry": {
            "definition": "lowest world z over all seven foot collision capsules, exact sphere/endpoints",
            "minimum_sole_height_m": percentiles(sole_height),
            "penetration_over_1mm_frames": int(np.count_nonzero(np.any(sole_height < -.001, axis=1))),
            "penetration_over_5mm_frames": int(np.count_nonzero(np.any(sole_height < -.005, axis=1))),
            "max_penetration_m": float(max(0., -sole_height.min())),
            "per_foot": {s: {"sole_height_m": percentiles(sole_height[:, i]),
                              "contact_sole_height_m": percentiles(sole_height[contacts[:, i], i])}
                         for i, s in enumerate(("left", "right"))}},
        "support_sliding": slide_metrics(sole_center, contacts, dt),
    }
    if limits is not None:
        joint_acceleration_enabled = limits.get('joint_acceleration_enabled', True)
        if not isinstance(joint_acceleration_enabled, bool):
            raise ValueError('joint_acceleration_enabled must be boolean')
        joint_velocity_limits = np.broadcast_to(np.asarray(limits["joint_velocity_rad_s"], dtype=float), (28,))
        joint_acceleration_limits = np.broadcast_to(np.asarray(limits["joint_acceleration_rad_s2"], dtype=float), (28,))
        excess = {
            "joint_velocity_rad_s": np.maximum(np.abs(velocity[:, 6:]) - joint_velocity_limits, 0.),
            "root_linear_velocity_m_s": np.maximum(np.linalg.norm(velocity[:, :3], axis=-1) - limits["root_linear_velocity_limit_m_s"], 0.),
            "root_linear_acceleration_m_s2": np.maximum(np.linalg.norm(accel[:, :3], axis=-1) - limits["root_linear_acceleration_limit_m_s2"], 0.),
            "root_angular_velocity_rad_s": np.maximum(np.linalg.norm(velocity[:, 3:6], axis=-1) - limits["root_angular_velocity_limit_rad_s"], 0.),
            "root_angular_acceleration_rad_s2": np.maximum(np.linalg.norm(accel[:, 3:6], axis=-1) - limits["root_angular_acceleration_limit_rad_s2"], 0.),
        }
        if joint_acceleration_enabled:
            # Keep this field only for an enforced bound. An absent bound is
            # not zero violations or zero limit binding; actual accelerations
            # are always measured in result['joint_acceleration_rad_s2'] above.
            excess["joint_acceleration_rad_s2"] = np.maximum(np.abs(accel[:, 6:]) - joint_acceleration_limits, 0.)
        result["output_bound_violations"] = {
            "definition": "MuJoCo tangent differences between FINAL qpos; joints per component, roots Euclidean norms",
            "tolerance_velocity": 1e-5, "tolerance_acceleration": 1e-3,
            "values": {k: {"violations": int(np.count_nonzero(a > (1e-3 if "acceleration" in k else 1e-5))),
                             "maximum_excess": float(a.max())} for k, a in excess.items()},
            "configured_limits": limits,
            "joint_acceleration_enabled": joint_acceleration_enabled,
            "joint_acceleration_braking_enabled": joint_acceleration_enabled,
            "disabled_bounds": ({} if joint_acceleration_enabled else {
                "joint_acceleration_rad_s2": "not enforced; configured numeric values are inactive; acceleration-dependent hinge braking is also disabled"}),
        }
    return result, velocity, accel


def cpu_model():
    p = Path("/proc/cpuinfo")
    if p.exists():
        for line in p.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def sdk_baseline(recording, indices, reference_qpos, model):
    """Recorded SDK output, with one fixed initial root alignment for comparison."""
    import mujoco
    from scipy.spatial.transform import Rotation
    joint_ids = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in joint_ids]
    source_names = recording["robot_joint_names"].tolist()
    joints = recording["robot_joint_position_rad"][indices][:, [source_names.index(n) for n in joint_names]]
    position = recording["robot_root_position_m"][indices].copy()
    quat = recording["robot_root_rotation_xyzw"][indices].copy()
    sdk_start = Rotation.from_quat(quat[0]).as_matrix()
    new_start = Rotation.from_quat(reference_qpos[0, [4, 5, 6, 3]]).as_matrix()
    sdk_yaw = np.arctan2(sdk_start[1, 0], sdk_start[0, 0])
    new_yaw = np.arctan2(new_start[1, 0], new_start[0, 0])
    align = Rotation.from_rotvec([0., 0., new_yaw - sdk_yaw])
    offset = reference_qpos[0, :3] - align.apply(position[0])
    position = align.apply(position) + offset
    quat = (align * Rotation.from_quat(quat)).as_quat()[:, [3, 0, 1, 2]]
    return np.column_stack((position, quat, joints)), {"yaw_offset_rad": float(new_yaw - sdk_yaw),
         "root_xyz_offset_m": offset.tolist(),
         "alignment": "one constant world yaw+translation, first root matched to new replay; no per-frame fitting",
         "fairness": "SDK used different morphology/calibration; values are a reference baseline, not a fair algorithm ranking"}


def source_parent_indices(recording):
    """Accept recorded topology, or infer the standard PN body hierarchy."""
    names = recording['joint_names'].tolist()
    if 'parent_index' in recording:
        parents = np.asarray(recording['parent_index'], dtype=int)
        if parents.shape != (len(names),):
            raise ValueError('parent_index shape does not match joint_names')
        return parents
    from pn_rp1_calibration import PN_PARENTS
    lookup = {name: index for index, name in enumerate(names)}
    return np.asarray([lookup.get(PN_PARENTS.get(name), -1) for name in names])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--start-s", type=float, default=None)
    ap.add_argument("--duration-s", type=float, default=None)
    ap.add_argument("--rate-hz", type=float, default=50.)
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT / "mocapapi_python"))
    import mujoco
    from pn_rp1_retarget import PNToRP1Retargeter

    recording = load_recording(args.source)
    ticks, indices = causal_samples(recording["timestamps_s"], args.rate_hz, args.start_s, args.duration_s)
    dt = 1. / args.rate_hz
    started = time.perf_counter()
    solver = PNToRP1Retargeter(config_path=args.config)
    initialization_ms = (time.perf_counter() - started) * 1000.
    model = solver.model
    data = mujoco.MjData(model)
    columns = {name: [] for name in ("qpos", "robot_body_position_m", "robot_body_quaternion_wxyz",
                "target_position_m", "target_quaternion_wxyz", "position_error_m", "rotation_error_deg",
                "sole_min_height_m", "sole_center_m", "contact_state", "retarget_ms", "solver_ms",
                "full_body_position_m", "full_body_quaternion_wxyz")}
    names = None
    failure_messages = []
    for step, (tick, index) in enumerate(zip(ticks, indices)):
        pose = world_pose(recording, index)
        stamp = float(recording["timestamps_s"][index])
        started = time.perf_counter()
        try:
            qpos = solver.retarget(pose, dt=dt, source_time_s=stamp)
        except Exception as exc:
            failure = {"output_step": step, "time_s": float(tick),
                       "source_index": int(index), "source_time_s": stamp,
                       "error": f"{type(exc).__name__}: {exc}",
                       "previous_qpos": solver.configuration.q.tolist(),
                       "previous_stats": solver.last_stats}
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "failed_replay.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps({"failure": failure}, ensure_ascii=False), file=sys.stderr)
            raise
        elapsed_ms = (time.perf_counter() - started) * 1000.
        targets = solver.last_targets
        if names is None:
            names = list(targets)
        if list(targets) != names:
            raise RuntimeError("target order changed during recording")
        p, q, tp, tq, pe, re, sh, sc = frame_fk(model, data, qpos, targets)
        stats = solver.last_stats
        contact = stats.get("contacts", stats.get("contact_state", {}))
        if isinstance(contact, dict):
            contact = [bool(contact.get(s, contact.get(f"{s}_foot", False))) for s in ("left", "right")]
        if len(contact) != 2:
            raise ValueError("last_stats.contacts must describe left/right feet")
        for key, value in zip(("qpos", "robot_body_position_m", "robot_body_quaternion_wxyz",
                  "target_position_m", "target_quaternion_wxyz", "position_error_m", "rotation_error_deg",
                  "sole_min_height_m", "sole_center_m", "contact_state", "retarget_ms", "solver_ms"),
                 (qpos.copy(), p, q, tp, tq, pe, re, sh, sc, contact, elapsed_ms, stats.get("solver_ms", elapsed_ms))):
            columns[key].append(value)
        columns["full_body_position_m"].append(data.xpos.copy())
        columns["full_body_quaternion_wxyz"].append(data.xquat.copy())
        if stats.get("status") not in (None, "ok"):
            failure_messages.append({"step": step, "time_s": float(tick), "status": stats.get("status")})
        if step and step % 500 == 0:
            print(f"replayed {step}/{len(ticks)} output frames", flush=True)
    arrays = {k: np.asarray(v) for k, v in columns.items()}
    arrays["contact_state"] = arrays["contact_state"].astype(bool)
    limits = solver.config.get("output_limits")
    quality, velocity, acceleration = quality_metrics(model, arrays["qpos"], arrays["position_error_m"],
        arrays["rotation_error_deg"], names, arrays["sole_min_height_m"], arrays["sole_center_m"],
        arrays["contact_state"], dt, limits)
    source_age = ticks - recording["timestamps_s"][indices]
    sdk_arrays = {}
    sdk_summary = {'available': False, 'reason': 'Recording has no complete SDK robot output.'}
    sdk_fields = {'robot_joint_names', 'robot_joint_position_rad',
                  'robot_root_position_m', 'robot_root_rotation_xyzw'}
    if sdk_fields.issubset(recording):
        sdk_qpos, sdk_align = sdk_baseline(recording, indices, arrays["qpos"], model)
        sdk_errors, sdk_rot_errors, sdk_sole_height, sdk_sole_center = [], [], [], []
        for step, qpos in enumerate(sdk_qpos):
            targets = {n: (arrays["target_position_m"][step, i], arrays["target_quaternion_wxyz"][step, i])
                       for i, n in enumerate(names)}
            _, _, _, _, pe, re, sh, sc = frame_fk(model, data, qpos, targets)
            sdk_errors.append(pe)
            sdk_rot_errors.append(re)
            sdk_sole_height.append(sh)
            sdk_sole_center.append(sc)
        sdk_quality, _, _ = quality_metrics(model, sdk_qpos, np.asarray(sdk_errors), np.asarray(sdk_rot_errors),
            names, np.asarray(sdk_sole_height), np.asarray(sdk_sole_center), arrays["contact_state"], dt)
        sdk_summary = {'available': True, 'transform': sdk_align, 'quality': sdk_quality}
        sdk_arrays = dict(sdk_qpos=sdk_qpos, sdk_position_error_m=np.asarray(sdk_errors),
                          sdk_rotation_error_deg=np.asarray(sdk_rot_errors),
                          sdk_sole_min_height_m=np.asarray(sdk_sole_height))
    result = {
        "scope": "complete offline causal replay; no ROS topics, policy execution, physics stepping, or hardware",
        "source": str(args.source.resolve()), "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "config": str(args.config.resolve()), "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "effective_runtime_parameters": {name: solver.config[name] for name in ("solver", "contact", "output_limits")},
        "robot_model": str(getattr(solver, "model_path", "see configuration")),
        "source_frames": len(recording["timestamps_s"]), "output_frames": len(ticks),
        "source_duration_s": float(recording["timestamps_s"][-1] - recording["timestamps_s"][0]),
        "output_interval_s": [float(ticks[0]), float(ticks[-1])], "output_rate_hz": args.rate_hz,
        "sampling": "causal latest sample: searchsorted(side=right)-1, reception timestamps, no future samples",
        "initialization_ms": initialization_ms,
        "environment": {"cpu": cpu_model(), "python": sys.version.split()[0],
                        "platform": platform.platform(), "mujoco_version": mujoco.__version__},
        "timing": {"all_retarget_ms": percentiles(arrays["retarget_ms"]),
                   "steady_retarget_ms_after_first_50_frames": percentiles(arrays["retarget_ms"][50:]),
                   "solver_ms": percentiles(arrays["solver_ms"]),
                   "over_20ms_output_frames": int((arrays["retarget_ms"] > 20.).sum()),
                   "source_age_ms": percentiles(source_age * 1000.),
                   "offline_reception_age_plus_retarget_ms": percentiles(source_age * 1000. + arrays["retarget_ms"])},
        "quality": quality, "reported_failures": failure_messages,
        "sdk_reference_baseline": sdk_summary,
        "largest_position_residuals": [
            {"time_s": float(ticks[frame]), "body": names[body],
             "position_error_m": float(arrays["position_error_m"][frame, body]),
             "rotation_error_deg": float(arrays["rotation_error_deg"][frame, body])}
            for frame, body in zip(*np.unravel_index(
                np.argsort(arrays["position_error_m"].ravel())[-10:][::-1], arrays["position_error_m"].shape))
        ],
        "limitations": [
            "No device timestamps: source age is elapsed time since recording reception, not sensor capture age.",
            "Position/orientation residuals compare final RP1 FK with scaled, calibrated IK targets; morphology, joint and rate limits prevent exact matches.",
            "Contact is a causal inference from kinematics, not a measured force/contact label.",
            "Stance sliding measures the projected local sole-center point; heel/toe rolling also moves this point.",
            "Sole height and stance sliding are FK reference diagnostics; they do not prove robot balance or tracking stability.",
            "OS jitter, SDK decode, transport, ROS scheduling and policy inference are excluded from solver timing.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "failed_replay.json").unlink(missing_ok=True)
    np.savez_compressed(args.output_dir / "replay.npz", **arrays,
         time_s=ticks, source_index=indices, source_time_s=recording["timestamps_s"][indices],
         source_age_s=source_age, target_names=np.asarray(names), velocity=velocity,
         acceleration=acceleration, **sdk_arrays,
         body_names=np.asarray([mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]),
         body_parent_indices=model.body_parentid.copy(), source_joint_names=recording["joint_names"],
         source_parent_index=source_parent_indices(recording),
         source_position_m=recording["global_position_ros_m"][indices],
         source_quaternion_wxyz=recording["global_rotation_ros_xyzw"][indices][..., [3, 0, 1, 2]])
    (args.output_dir / "evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output_dir": str(args.output_dir), "timing": result["timing"],
                      "quality": {k: v for k, v in quality.items() if k not in ("per_body", "support_sliding")}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
