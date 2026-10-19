"""Deterministic PN Axis/PN-Link -> RP1.3.2 calibration and geometry adapter.

The source reference is an analytical, palms-down T pose. All PN local and
world bind rotations are exactly identity. A recording can supply *scalar
median bone lengths only*: no recorded position direction, orientation, root
height, or selected frame is used as a pose reference. This module has no ROS
or device imports. MuJoCo is needed only to regenerate the robot calibration.

Runtime contract::

    adapter = PNSourceAdapter(config_path)
    targets = adapter.target_frames(pn_world_poses)

``pn_world_poses`` contains all 21 body joints in canonical ROS axes (+X
forward, +Y left, +Z up), metres and quaternion **wxyz**. ``targets`` uses RP1
MuJoCo body names and is ready to set on mink FrameTasks. Do not feed these
targets through GMR's source offsets a second time. The accompanying JSON
retains GMR's mapping/cost fields for task construction and inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping
import xml.etree.ElementTree as ET

import numpy as np

from pn_rp1_paths import default_model_path, resolve_model_path

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "rp1" / "pn_axis_to_rp1_28dof.json"
DEFAULT_MODEL = default_model_path()
DEFAULT_RECORDING = HERE / "recordings" / "dataset.npz"

PN_PARENTS = {
    "Hips": None,
    "RightUpLeg": "Hips", "RightLeg": "RightUpLeg", "RightFoot": "RightLeg",
    "LeftUpLeg": "Hips", "LeftLeg": "LeftUpLeg", "LeftFoot": "LeftLeg",
    "Spine": "Hips", "Spine1": "Spine", "Spine2": "Spine1",
    "Neck": "Spine2", "Neck1": "Neck", "Head": "Neck1",
    "RightShoulder": "Spine2", "RightArm": "RightShoulder",
    "RightForeArm": "RightArm", "RightHand": "RightForeArm",
    "LeftShoulder": "Spine2", "LeftArm": "LeftShoulder",
    "LeftForeArm": "LeftArm", "LeftHand": "LeftForeArm",
}
PN_JOINT_NAMES = tuple(PN_PARENTS)
ROBOT_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_roll_joint", "waist_yaw_joint", "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_yaw_joint", "right_wrist_pitch_joint",
)
IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
NATIVE_TO_CANONICAL = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
DEFAULT_TORSO_COMPENSATION = {"enabled": True, "pitch_limit_deg": 50.0}

# Explicit neutral dimensions, in metres. These are configurable template
# dimensions, not a claim of a single universal human skeleton. A clip may
# replace scalar lengths but never these direction/rotation conventions.
STANDARD_LENGTHS_M = {
    "RightUpLeg": 0.10, "RightLeg": 0.43, "RightFoot": 0.43,
    "LeftUpLeg": 0.10, "LeftLeg": 0.43, "LeftFoot": 0.43,
    "Spine": 0.10, "Spine1": 0.18, "Spine2": 0.15,
    "Neck": 0.18, "Neck1": 0.045, "Head": 0.045,
    "RightShoulder": 0.15, "RightArm": 0.15,
    "RightForeArm": 0.29, "RightHand": 0.27,
    "LeftShoulder": 0.15, "LeftArm": 0.15,
    "LeftForeArm": 0.29, "LeftHand": 0.27,
}


def normalize_quaternion(quaternion):
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("quaternion must contain four finite wxyz values")
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        raise ValueError("zero quaternion is not a valid PN pose")
    return q / norm


def quat_multiply(lhs, rhs):
    aw, ax, ay, az = lhs
    bw, bx, by, bz = rhs
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def rotate_vector(quaternion, vector):
    q = np.asarray(quaternion, dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    cross = 2.0 * np.cross(q[1:], vector)
    return vector + q[0] * cross + np.cross(q[1:], cross)


def native_pose_to_canonical(position_cm, rotation_wxyz):
    """Conjugate the SDK frame, preserving its wxyz quaternion ordering."""
    position = np.asarray(position_cm, dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("native position must contain three finite cm values")
    q = normalize_quaternion(rotation_wxyz)
    return NATIVE_TO_CANONICAL @ position / 100.0, q[[0, 3, 1, 2]]


def _validated_lengths(lengths=None):
    result = dict(STANDARD_LENGTHS_M)
    if lengths is not None:
        unknown = set(lengths) - set(result)
        if unknown:
            raise ValueError(f"unknown PN length keys: {sorted(unknown)}")
        result.update({key: float(value) for key, value in lengths.items()})
    for key, value in result.items():
        if not math.isfinite(value) or not 0.005 < value < 1.0:
            raise ValueError(f"invalid scalar bone length for {key}: {value}")
    return result


def standard_pn_tpose(lengths=None, ankle_height_m=0.05):
    """Return a mathematical PN T pose; rotations and directions are constant.

    Arms point along +/-Y, legs along -Z, torso along +Z, pelvis offsets along
    +/-Y. Clavicles rise with a fixed template direction (1 lateral : 4 up).
    The supplied scalar lengths change dimensions only. Feet are flat and face
    +X; identity hand bind means palms down under the PN Axis convention.
    """
    lengths = _validated_lengths(lengths)
    ankle_height_m = float(ankle_height_m)
    if not math.isfinite(ankle_height_m) or not 0.0 <= ankle_height_m <= 0.2:
        raise ValueError("PN ankle height must be between 0 and 0.2 m")
    leg_height = np.mean([
        lengths[f"{side}Leg"] + lengths[f"{side}Foot"]
        for side in ("Left", "Right")
    ])
    poses = {"Hips": [np.array([0.0, 0.0, leg_height + ankle_height_m]), IDENTITY.copy()]}
    for name, parent in PN_PARENTS.items():
        if parent is None:
            continue
        sign = 1.0 if name.startswith("Left") else -1.0
        if name.endswith("UpLeg"):
            direction = np.array([0.0, sign, 0.0])
        elif name.endswith(("Leg", "Foot")):
            direction = np.array([0.0, 0.0, -1.0])
        elif name.endswith("Shoulder"):
            direction = np.array([0.0, sign, 4.0]) / math.sqrt(17.0)
        elif name.endswith(("Arm", "Hand")):
            direction = np.array([0.0, sign, 0.0])
        else:
            direction = np.array([0.0, 0.0, 1.0])
        poses[name] = [poses[parent][0] + lengths[name] * direction, IDENTITY.copy()]
    return poses


def estimate_actor_lengths(npz_path):
    """Read the whole clip's distance magnitudes only, with no pose selection.

    No quaternion, first-frame offset, root height, or vector direction is
    read. The result depends only on inter-joint Euclidean distances.
    """
    npz_path = Path(npz_path).expanduser().resolve()
    with np.load(npz_path, allow_pickle=False) as recording:
        names = [str(name) for name in recording["joint_names"].tolist()]
        if len(set(names)) != len(names):
            raise ValueError("recording contains duplicate joint names")
        missing = set(PN_JOINT_NAMES) - set(names)
        if missing:
            raise ValueError(f"recording misses PN body joints: {sorted(missing)}")
        positions = recording["global_position_ros_m"]
        if positions.ndim != 3 or positions.shape[1:] != (len(names), 3):
            raise ValueError("recording world positions have an invalid shape")
        if not np.all(np.isfinite(positions)) or len(positions) < 2:
            raise ValueError("recording needs at least two finite frames")
        indices = {name: i for i, name in enumerate(names)}
        lengths = {
            name: float(np.median(np.linalg.norm(
                positions[:, indices[name]] - positions[:, indices[parent]], axis=1
            )))
            for name, parent in PN_PARENTS.items() if parent is not None
        }
        frame_count = len(positions)
    lengths = _validated_lengths(lengths)
    provenance = {
        "method": "median of Euclidean parent-child distances over every frame",
        "recording": str(npz_path),
        "recording_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
        "frame_count": frame_count,
        "pose_values_used": False,
        "rotation_arrays_read": False,
        "selected_reference_frame": None,
        "purpose": "actor dimensions only; pose directions and bind rotations are analytical constants",
    }
    return lengths, provenance


def robot_standard_tpose(model_path=None):
    """Ground RP1's mechanical T pose using its actual collision sole geometry."""
    import mujoco

    model_path = resolve_model_path(model_path)
    if ET.parse(model_path).getroot().get("model") != "rp1.3.2":
        raise ValueError(f"expected rp1.3.2 model, got {model_path}")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    if model.nq != 35:
        raise ValueError(f"expected 28 hinges and free base (nq=35), got nq={model.nq}")
    data.qpos[:] = 0.0
    data.qpos[3] = 1.0
    # This robot's forearm points +X at zero elbow. Its standard T reference
    # therefore needs +/-90 elbow as well as +/-90 shoulder roll.
    standard_angles = {name: 0.0 for name in ROBOT_JOINT_NAMES}
    standard_angles.update({
        "left_shoulder_roll_joint": math.pi / 2.0,
        "right_shoulder_roll_joint": -math.pi / 2.0,
        "left_elbow_joint": -math.pi / 2.0,
        "right_elbow_joint": math.pi / 2.0,
    })
    for name, angle in standard_angles.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"RP1 model misses {name}")
        if model.jnt_qposadr[joint_id] != 7 + ROBOT_JOINT_NAMES.index(name):
            raise ValueError("RP1 model joint order does not match the 35-value reference")
        lo, hi = model.jnt_range[joint_id]
        if model.jnt_limited[joint_id] and not lo <= angle <= hi:
            raise ValueError(f"standard T pose exceeds {name} range")
        data.qpos[model.jnt_qposadr[joint_id]] = angle
    mujoco.mj_forward(model, data)

    # RP1 soles use horizontal capsules. Store the bottom centre in the ankle
    # frame and verify that the capsule radius is included (toe body itself is
    # 1 cm above the collision bottom).
    sole_local = {}
    lowest_z = math.inf
    for side in ("left", "right"):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link")
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_center_collision")
        if body_id < 0 or geom_id < 0:
            raise ValueError(f"RP1 {side} foot collision geometry is missing")
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_CAPSULE:
            raise ValueError("RP1 sole calibration expects the capsule foot geometry")
        radius = float(model.geom_size[geom_id, 0])
        center_bottom = data.geom_xpos[geom_id] - np.array([0.0, 0.0, radius])
        local = data.xmat[body_id].reshape(3, 3).T @ (center_bottom - data.xpos[body_id])
        sole_local[side] = local.tolist()
        first, last = model.body_geomadr[body_id], model.body_geomadr[body_id] + model.body_geomnum[body_id]
        for gid in range(first, last):
            if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_CAPSULE:
                continue
            geom_axes = data.geom_xmat[gid].reshape(3, 3)
            capsule_z_extent = abs(geom_axes[2, 2]) * model.geom_size[gid, 1] + model.geom_size[gid, 0]
            lowest_z = min(lowest_z, float(data.geom_xpos[gid, 2] - capsule_z_extent))
    data.qpos[2] = -lowest_z
    mujoco.mj_forward(model, data)
    poses = {}
    for body_id in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        poses[name] = [data.xpos[body_id].copy(), data.xquat[body_id].copy()]
    return {
        "model": model, "qpos": data.qpos.copy(), "poses": poses,
        "joint_angles_rad": standard_angles, "sole_offsets_local_m": sole_local,
        "model_path": str(model_path),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }


def _json_poses(poses):
    return {name: [pose[0].tolist(), pose[1].tolist()] for name, pose in poses.items()}


def build_calibration(model_path=None, recording_path=None, actor_lengths=None,
                      *, mapping_profile="yaw", root_position_scale=None):
    """Build a GMR-compatible mapping with explicit analytical source provenance.

    ``yaw`` retains the original distal hip/shoulder task frames. ``pitch``
    selects the proximal pitch frames requested for the alternative mapping;
    their origins and bind orientations come from RP1 FK, not renamed yaw
    offsets. Distal knee/elbow/foot/hand geometry remains the same.
    ``root_position_scale`` overrides displacement scale only; it does not
    stretch CAD limb dimensions or change the analytical reference posture.
    """
    if mapping_profile not in ("yaw", "pitch"):
        raise ValueError("mapping profile must be 'yaw' or 'pitch'")
    if root_position_scale is not None:
        root_position_scale = float(root_position_scale)
        if not math.isfinite(root_position_scale) or root_position_scale <= 0.0:
            raise ValueError("root position scale must be positive and finite")
    if recording_path is not None and actor_lengths is not None:
        raise ValueError("provide either a recording for length statistics or scalar actor lengths")
    if recording_path is not None:
        lengths, provenance = estimate_actor_lengths(recording_path)
    else:
        lengths = _validated_lengths(actor_lengths)
        provenance = {"method": "explicit scalar dimensions" if actor_lengths else "fixed neutral template dimensions",
                      "pose_values_used": False, "selected_reference_frame": None}
    source_ref = standard_pn_tpose(lengths)
    robot = robot_standard_tpose(model_path)
    ref = robot["poses"]
    anthropometric_root_scale = float(ref["base_link"][0][2] / source_ref["Hips"][0][2])
    root_scale = (anthropometric_root_scale if root_position_scale is None
                  else root_position_scale)

    # Rules form an acyclic position graph. An anatomical segment is scaled
    # to the RP1 segment, then receives only the exact CAD rest residual.
    # Thus standard PN T -> exact RP1 T, including RP1's small CAD offsets.
    rules = [{"name": "base_link", "kind": "root"}]

    def offset(name, anchor, driver, target_delta):
        rules.append({"name": name, "kind": "offset", "anchor": anchor,
                      "driver": driver, "reference_delta_m": np.asarray(target_delta).tolist()})

    def segment(name, anchor, driver, source_parent, source_child, target_delta):
        source_delta = source_ref[source_child][0] - source_ref[source_parent][0]
        source_direction = source_delta / np.linalg.norm(source_delta)
        target_delta = np.asarray(target_delta)
        target_length = float(np.linalg.norm(target_delta))
        rules.append({"name": name, "kind": "segment", "anchor": anchor,
                      "driver": driver, "source_parent": source_parent,
                      "source_child": source_child,
                      "source_direction_tpose": source_direction.tolist(),
                      "source_length_m": float(lengths[source_child]),
                      "target_length_m": target_length,
                      "cad_residual_m": (target_delta - target_length * source_direction).tolist(),
                      "reference_delta_m": target_delta.tolist()})

    offset("waist_yaw_link", "base_link", "Hips", ref["waist_yaw_link"][0] - ref["base_link"][0])
    target_sources = {"base_link": "Hips", "waist_yaw_link": "Spine2"}
    for side, lower in (("Left", "left"), ("Right", "right")):
        hip_pivot = f"{lower}_hip_pivot"
        shoulder_pivot = f"{lower}_shoulder_pivot"
        hip = f"{lower}_hip_{mapping_profile}_link"
        knee = f"{lower}_knee_link"
        ankle = f"{lower}_ankle_roll_link"
        shoulder = f"{lower}_shoulder_{mapping_profile}_link"
        elbow = f"{lower}_elbow_link"
        wrist = f"{lower}_wrist_pitch_link"
        hip_roll = ref[f"{lower}_hip_roll_link"][0]
        shoulder_roll = ref[f"{lower}_shoulder_roll_link"][0]
        offset(hip_pivot, "base_link", "Hips", hip_roll - ref["base_link"][0])
        if mapping_profile == "pitch":
            # This body's origin is fixed in the base, before the pitch
            # hinge. Upper-leg rotation must not rotate its position around
            # the downstream hip-roll pivot.
            offset(hip, "base_link", "Hips", ref[hip][0] - ref["base_link"][0])
        else:
            offset(hip, hip_pivot, f"{side}UpLeg", ref[hip][0] - hip_roll)
        segment(knee, hip_pivot, f"{side}UpLeg", f"{side}UpLeg", f"{side}Leg", ref[knee][0] - hip_roll)
        segment(ankle, knee, f"{side}Leg", f"{side}Leg", f"{side}Foot", ref[ankle][0] - ref[knee][0])
        offset(shoulder_pivot, "waist_yaw_link", "Spine2", shoulder_roll - ref["waist_yaw_link"][0])
        if mapping_profile == "pitch":
            # Shoulder pitch is attached directly to the waist. Its T-pose
            # frame also precedes the +/-90-degree shoulder-roll rotation.
            offset(shoulder, "waist_yaw_link", "Spine2", ref[shoulder][0] - ref["waist_yaw_link"][0])
        else:
            offset(shoulder, shoulder_pivot, f"{side}Arm", ref[shoulder][0] - shoulder_roll)
        segment(elbow, shoulder_pivot, f"{side}Arm", f"{side}Arm", f"{side}ForeArm", ref[elbow][0] - shoulder_roll)
        segment(wrist, elbow, f"{side}ForeArm", f"{side}ForeArm", f"{side}Hand", ref[wrist][0] - ref[elbow][0])
        target_sources.update({hip: f"{side}UpLeg", knee: f"{side}Leg", ankle: f"{side}Foot",
                               shoulder: f"{side}Arm", elbow: f"{side}ForeArm", wrist: f"{side}Hand"})

    frames, table1, table2 = {}, {}, {}
    for body, source in target_sources.items():
        source_q = source_ref[source][1]
        inverse = source_q * np.array([1.0, -1.0, -1.0, -1.0])
        rotation_offset = normalize_quaternion(quat_multiply(inverse, ref[body][1]))
        if body == "base_link":
            position_cost, orientation_cost = 30.0, 12.0
        elif body.endswith("ankle_roll_link"):
            position_cost, orientation_cost = 30.0, 3.0
        elif body.endswith("wrist_pitch_link"):
            position_cost, orientation_cost = 6.0, 1.0
        elif body.endswith("elbow_link"):
            position_cost, orientation_cost = 6.0, 0.6
        elif body == "waist_yaw_link":
            position_cost, orientation_cost = 0.0, 3.0
        else:
            position_cost, orientation_cost = 1.0, 0.8
        frames[body] = {
            "source_joint": source,
            "position_rule": body,
            "orientation_offset_wxyz": rotation_offset.tolist(),
            "robot_reference_position_m": ref[body][0].tolist(),
            "robot_reference_rotation_wxyz": ref[body][1].tolist(),
        }
        table1[body] = [source, position_cost, orientation_cost, [0.0, 0.0, 0.0], rotation_offset.tolist()]
        table2[body] = [source, position_cost, orientation_cost, [0.0, 0.0, 0.0], rotation_offset.tolist()]

    config = {
        "robot_root_name": "base_link", "human_root_name": "Hips", "ground_height": 0.0,
        "human_height_assumption": 1.8,
        "use_ik_match_table1": True, "use_ik_match_table2": False,
        "human_scale_table": {name: 1.0 for name in PN_JOINT_NAMES},
        "ik_match_table1": table1, "ik_match_table2": table2,
        "torso_compensation": dict(DEFAULT_TORSO_COMPENSATION),
        "source_calibration": {
            "schema_version": "pn-rp1-standard-tpose-v1",
            "source_name": "PN Axis / PN-Link 21 body joints",
            "source_pose_definition": "analytical standard palms-down T pose; identity PN local bind rotations; no recorded pose",
            "source_axes": "+X forward, +Y left, +Z up; metres; quaternion wxyz",
            "native_axes": "+X left, +Y up, +Z forward; centimetres; quaternion wxyz",
            "native_to_canonical_matrix": NATIVE_TO_CANONICAL.tolist(),
            "required_joints": list(PN_JOINT_NAMES),
            "source_parents": PN_PARENTS,
            "source_tpose_world": _json_poses(source_ref),
            "source_tpose_local_rotation_wxyz": {name: IDENTITY.tolist() for name in PN_JOINT_NAMES},
            "actor_median_lengths_m": lengths,
            "length_provenance": provenance,
            "source_ankle_height_m": 0.05,
            "source_sole_offsets_local_m": {"left": [0.0, 0.0, -0.05], "right": [0.0, 0.0, -0.05]},
            "robot_name": "rp1.3.2", "robot_model_path": robot["model_path"],
            "robot_model_sha256": robot["model_sha256"],
            "robot_tpose_definition": "all hinge angles zero except shoulder roll +90/-90 and elbow -90/+90 degrees; base yaw zero; capsule soles on z=0",
            "robot_tpose_qpos": robot["qpos"].tolist(),
            "robot_joint_names": list(ROBOT_JOINT_NAMES),
            "robot_tpose_joint_angles_rad": robot["joint_angles_rad"],
            "robot_tpose_world": _json_poses(ref),
            "robot_sole_offsets_local_m": robot["sole_offsets_local_m"],
            "mapping_profile": mapping_profile,
            "mapping_definition": ("PN UpLeg -> hip_pitch_link; PN Arm -> shoulder_pitch_link; proximal frame origins use parent-driven CAD offsets"
                                   if mapping_profile == "pitch" else
                                   "PN UpLeg -> hip_yaw_link; PN Arm -> shoulder_yaw_link"),
            "root_position_scale": root_scale,
            "anthropometric_root_position_scale": anthropometric_root_scale,
            "root_position_scale_method": ("robot/source standard-T root height ratio"
                                            if root_position_scale is None else "explicit displacement-scale override"),
            "root_reference_position_m": ref["base_link"][0].tolist(),
            "source_root_reference_position_m": source_ref["Hips"][0].tolist(),
            "root_transform": "robot_ref + root_position_scale * (source_Hips_xyz - source_ref)",
            "orientation_transform": "source_world_quat * inverse(source_T_world_quat) * robot_T_world_quat",
            "target_adapter_required": True,
            "position_rules": rules, "target_frames": frames,
            "limitations": [
                "The 5 cm human ankle-to-floor distance is an explicit template dimension; PN has no measured sole or toe in this 21-joint stream.",
                "RP1's standard joint-angle T reference preserves actual CAD offsets, so mechanical segment endpoints are not a perfectly horizontal human line.",
                "This source adapter creates geometric targets; it does not enforce robot dynamics, contacts, balance, or self-collision.",
            ],
        },
    }
    check = PNSourceAdapter(config).target_frames(source_ref)
    max_pos = max(float(np.linalg.norm(check[body][0] - ref[body][0])) for body in frames)
    max_rot = max(2.0 * math.acos(min(1.0, abs(float(np.dot(check[body][1], ref[body][1]))))) for body in frames)
    if max_pos > 1e-9 or max_rot > 1e-7:
        raise AssertionError("analytical PN T pose failed to reproduce the robot T reference")
    config["source_calibration"]["standard_tpose_verification"] = {
        "max_target_position_error_m": max_pos,
        "max_target_orientation_error_rad": max_rot,
        "recorded_pose_reference_used": False,
    }
    return config


class PNSourceAdapter:
    """Stateless PN geometry adapter; neither input frames nor config are mutated."""

    def __init__(self, config=DEFAULT_CONFIG):
        if isinstance(config, (str, Path)):
            config = json.loads(Path(config).read_text(encoding="utf-8"))
        self.calibration = config["source_calibration"]
        if self.calibration["schema_version"] != "pn-rp1-standard-tpose-v1":
            raise ValueError("unsupported PN-RP1 calibration schema")
        if self.calibration["robot_name"] != "rp1.3.2":
            raise ValueError("PN calibration requires the RP1.3.2 target model")
        self.required_joints = tuple(self.calibration["required_joints"])
        self.rules = tuple(self.calibration["position_rules"])
        self.frames = self.calibration["target_frames"]
        self.root_scale = float(self.calibration["root_position_scale"])
        self.source_root_ref = np.array(self.calibration["source_root_reference_position_m"])
        self.robot_root_ref = np.array(self.calibration["root_reference_position_m"])
        self.source_tpose = {name: [np.array(pose[0]), np.array(pose[1])]
                             for name, pose in self.calibration["source_tpose_world"].items()}
        self.torso_compensation = {
            **DEFAULT_TORSO_COMPENSATION, **config.get("torso_compensation", {})
        }
        if not isinstance(self.torso_compensation["enabled"], bool):
            raise ValueError("torso compensation enabled must be a boolean")
        limit = float(self.torso_compensation["pitch_limit_deg"])
        if not math.isfinite(limit) or not 0.0 < limit < 90.0:
            raise ValueError("torso compensation pitch limit must be between 0 and 90 degrees")
        self.torso_pitch_limit = math.radians(limit)
        # RP1 has waist roll/yaw, but no waist pitch. Rotate the virtual pelvis
        # about the midpoint of its mechanical hip pivots so pitching the
        # floating base does not also translate the planted leg targets.
        hip_pivots = [np.asarray(rule["reference_delta_m"], dtype=float)
                      for rule in self.rules
                      if rule["name"] in ("left_hip_pivot", "right_hip_pivot")]
        if len(hip_pivots) != 2:
            raise ValueError("PN calibration needs both mechanical hip pivots")
        self.hip_pivot_local = np.mean(hip_pivots, axis=0)

    def torso_compensation_angle(self, poses: Mapping):
        """Return missing waist pitch transferred to the virtual pelvis.

        Sagittal tilt is measured from Spine2's +Z axis in the Hips frame;
        pelvis yaw is retained. The standard identity T pose maps to zero.
        No recorded pose offset, state, or input mutation is involved.
        """
        hips = normalize_quaternion(poses["Hips"][1])
        torso = normalize_quaternion(poses["Spine2"][1])
        inverse_hips = hips * np.array([1.0, -1.0, -1.0, -1.0])
        torso_axis = rotate_vector(quat_multiply(inverse_hips, torso), [0.0, 0.0, 1.0])
        pitch = math.atan2(float(torso_axis[0]), float(torso_axis[2]))
        if not self.torso_compensation["enabled"]:
            return 0.0
        return float(np.clip(pitch, -self.torso_pitch_limit, self.torso_pitch_limit))

    def target_frames(self, poses: Mapping):
        missing = set(self.required_joints) - set(poses)
        if missing:
            raise ValueError(f"PN frame misses body joints: {sorted(missing)}")
        positions, rotations = {}, {}
        for name in self.required_joints:
            pose = poses[name]
            if len(pose) != 2:
                raise ValueError(f"PN {name} pose must be [xyz_m, quaternion_wxyz]")
            position = np.asarray(pose[0], dtype=np.float64)
            if position.shape != (3,) or not np.all(np.isfinite(position)):
                raise ValueError(f"PN {name} has an invalid world position")
            positions[name] = position
            rotations[name] = normalize_quaternion(pose[1])
        pitch = self.torso_compensation_angle(poses)
        hips_rotation = rotations["Hips"]
        virtual_rotation = normalize_quaternion(quat_multiply(
            hips_rotation, [math.cos(pitch / 2.0), 0.0, math.sin(pitch / 2.0), 0.0]
        ))
        root_correction = (rotate_vector(hips_rotation, self.hip_pivot_local)
                           - rotate_vector(virtual_rotation, self.hip_pivot_local))
        rotations["Hips"] = virtual_rotation
        mapped = {}
        for rule in self.rules:
            name, kind = rule["name"], rule["kind"]
            if kind == "root":
                mapped[name] = (self.robot_root_ref
                                + self.root_scale * (positions["Hips"] - self.source_root_ref)
                                + root_correction)
            elif kind == "offset":
                mapped[name] = mapped[rule["anchor"]] + rotate_vector(rotations[rule["driver"]], rule["reference_delta_m"])
            elif kind == "segment":
                delta = positions[rule["source_child"]] - positions[rule["source_parent"]]
                length = float(np.linalg.norm(delta))
                if length < 0.005:
                    raise ValueError(f"PN segment {rule['source_parent']} -> {rule['source_child']} is collapsed")
                mapped[name] = (mapped[rule["anchor"]] + rule["target_length_m"] * delta / length
                                + rotate_vector(rotations[rule["driver"]], rule["cad_residual_m"]))
            else:
                raise ValueError(f"unknown PN target position rule: {kind}")
        return {
            body: [mapped[frame["position_rule"]].copy(),
                   normalize_quaternion(quat_multiply(rotations[frame["source_joint"]], frame["orientation_offset_wxyz"]))]
            for body, frame in self.frames.items()
        }

    def source_to_targets(self, poses: Mapping):
        """Alias suitable for integrations using a functional adapter name."""
        return self.target_frames(poses)


def source_to_targets(poses, config=DEFAULT_CONFIG):
    return PNSourceAdapter(config).target_frames(poses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None,
                        help='RP1 MJCF path; defaults to PN_RP1_MODEL or a discovered model')
    parser.add_argument("--recording-lengths", type=Path, help="use all recording frames' scalar median bone lengths only")
    parser.add_argument("--actor-lengths", type=Path, help="JSON of scalar child-joint bone lengths in metres")
    parser.add_argument("--mapping-profile", choices=("yaw", "pitch"), default="yaw",
                        help="hip/shoulder frame profile; pitch uses exact proximal RP1 frames")
    parser.add_argument("--root-position-scale", type=float,
                        help="override root XYZ displacement scale; robot limb lengths are unchanged")
    parser.add_argument("--output", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    actor_lengths = None
    if args.actor_lengths is not None:
        actor_lengths = json.loads(args.actor_lengths.read_text(encoding="utf-8"))
    config = build_calibration(args.model, args.recording_lengths, actor_lengths,
                               mapping_profile=args.mapping_profile,
                               root_position_scale=args.root_position_scale)
    # Regenerating standard geometry must not discard separately tuned online
    # limits or contact/solver settings from the existing deployment config.
    if args.output.is_file():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        for key in ("output_limits", "solver", "contact", "torso_compensation"):
            if key in previous:
                config[key] = previous[key]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    calibration = config["source_calibration"]
    print(json.dumps({"output": str(args.output.resolve()), "robot": calibration["robot_name"],
                      "root_height_m": calibration["robot_tpose_qpos"][2],
                      "mapping_profile": calibration["mapping_profile"],
                      "root_position_scale": calibration["root_position_scale"],
                      "source_pose": calibration["source_pose_definition"],
                      "length_method": calibration["length_provenance"]["method"],
                      "verification": calibration["standard_tpose_verification"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
