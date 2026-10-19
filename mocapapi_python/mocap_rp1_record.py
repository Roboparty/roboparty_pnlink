"""Record the PN-Link body skeleton for a later RP1 retarget fit.

Run this instead of ``mocap_rp1_reference.py``. Both open the same PN-Link
port, so only one of them can run.

    cd mocapapi_python
    python mocap_rp1_record.py

Press N to start device capture, S to start a take, and E to stop and save.
ESC saves an open take and quits. Each take is written to ``recordings/``:

* ``raw_skeleton.bvh`` — the body skeleton in the SDK frame: centimetres,
  Y up, X left, Z forward. Rotation order is YXZ. Position channels are the
  absolute local translation. This is not the reference ROS axis order.
* ``dataset.npz`` — the same clip as arrays, plus the solver's raw joint
  angles and the current retarget output. This is the dataset for a
  position / orientation loss.
* ``dataset.csv`` — the same rows in text.

The 21 joints are the body used by ``/raw_skeleton`` in
``mocap_rp1_reference.py``. Finger bones are not part of the RP1 retarget.
BVH row ``i`` is NPZ row ``i``. ``timestamps_s`` is the reception clock;
the BVH frame time is the median interval.

Native positions are the SDK axes in centimetres, Y up. The ROS world used
by the reference publisher is metres with ``(x, y, z) = (z, x, y) / 100``
and quaternion ``xyzw = (z, x, y, w)`` from native ``wxyz``. Global arrays
in that world are the keypoint targets for a retarget loss.

``robot_raw_joint_angle`` is ``GetRobotRawJointAngle`` in SDK units. Index
``i`` is ``EMCPRobotJointType`` and the ``rawJoint`` field in
``rp1/retarget.json``. ``robot_joint_position_rad`` is the current retarget,
radians, in ``urdfJointNames`` order. The robot root is the solver root in
metres and quaternion xyzw, without the teleop ground-height bias and
without the A-key planar alignment.
"""

import argparse
import asyncio
import csv
import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from pn_sdk import load_sdk

_sdk, MCPBase = load_sdk()
MCPAvatar, MCPRobot, MCPRobotJointType = _sdk.MCPAvatar, _sdk.MCPRobot, _sdk.MCPRobotJointType

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RETARGET_CONFIG = SCRIPT_DIR / "rp1" / "retarget.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "recordings"

# Parent of each body joint. The same map draws /raw_skeleton.
RAW_PARENTS = {
    "Hips": "world",
    "RightUpLeg": "Hips",
    "RightLeg": "RightUpLeg",
    "RightFoot": "RightLeg",
    "LeftUpLeg": "Hips",
    "LeftLeg": "LeftUpLeg",
    "LeftFoot": "LeftLeg",
    "Spine": "Hips",
    "Spine1": "Spine",
    "Spine2": "Spine1",
    "Neck": "Spine2",
    "Neck1": "Neck",
    "Head": "Neck1",
    "RightShoulder": "Spine2",
    "RightArm": "RightShoulder",
    "RightForeArm": "RightArm",
    "RightHand": "RightForeArm",
    "LeftShoulder": "Spine2",
    "LeftArm": "LeftShoulder",
    "LeftForeArm": "LeftArm",
    "LeftHand": "LeftForeArm",
}

# Depth-first order, so a BVH written from RAW_PARENTS lists joints in
# this sequence.
JOINT_NAMES = (
    "Hips",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Neck1",
    "Head",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
)
JOINT_INDEX = {name: index for index, name in enumerate(JOINT_NAMES)}
ROBOT_RAW_JOINTS = tuple(
    name for name in MCPRobotJointType._fields if name != "JointCount"
)

_BVH_CHANNELS = (
    "Xposition Yposition Zposition Yrotation Xrotation Zrotation"
)


def _parent_index():
    parents = np.empty(len(JOINT_NAMES), dtype=np.int32)
    for index, name in enumerate(JOINT_NAMES):
        parent = RAW_PARENTS[name]
        parents[index] = -1 if parent == "world" else JOINT_INDEX[parent]
    return parents


PARENT_INDEX = _parent_index()


def _normalize_wxyz(quat):
    quat = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return quat / norm


def _quat_mul_wxyz(lhs, rhs):
    aw, ax, ay, az = (float(value) for value in lhs)
    bw, bx, by, bz = (float(value) for value in rhs)
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def _rotate_wxyz(quat, vector):
    w, x, y, z = (float(value) for value in quat)
    vx, vy, vz = (float(value) for value in vector)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return np.array(
        [
            vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx),
        ]
    )


def _mocap_pose_to_ros(position_cm, rotation_wxyz):
    """Match mocap_rp1_reference.py: SDK centimetres, wxyz to ROS metres, xyzw."""
    w, x, y, z = (float(value) for value in rotation_wxyz)
    translation = np.array(
        [
            float(position_cm[2]) / 100.0,
            float(position_cm[0]) / 100.0,
            float(position_cm[1]) / 100.0,
        ]
    )
    rotation_xyzw = np.array([z, x, y, w])
    return translation, rotation_xyzw


def _xyzw_to_wxyz(quat):
    x, y, z, w = (float(value) for value in quat)
    return np.array([w, x, y, z])


def _wxyz_to_xyzw(quat):
    w, x, y, z = (float(value) for value in quat)
    return np.array([x, y, z, w])


def _matrix_from_quat(quat):
    w, x, y, z = _normalize_wxyz(quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def _quat_from_axis_angle(axis, angle):
    half = float(angle) * 0.5
    scale = math.sin(half)
    return np.array(
        [
            math.cos(half),
            float(axis[0]) * scale,
            float(axis[1]) * scale,
            float(axis[2]) * scale,
        ]
    )


def _matrix_from_yxz(yaw, pitch, roll):
    """R = Ry(yaw) @ Rx(pitch) @ Rz(roll), radians."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cx, sx = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(roll), math.sin(roll)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return ry @ rx @ rz


def _quat_from_yxz(yaw, pitch, roll):
    """Quaternion for R = Ry @ Rx @ Rz. Angles are radians."""
    quat = _quat_mul_wxyz(
        _quat_from_axis_angle((0.0, 1.0, 0.0), yaw),
        _quat_from_axis_angle((1.0, 0.0, 0.0), pitch),
    )
    return _quat_mul_wxyz(quat, _quat_from_axis_angle((0.0, 0.0, 1.0), roll))


def _yxz_from_matrix(matrix):
    """Inverse of ``_matrix_from_yxz``. Pitch is kept in [-90, 90] degrees."""
    sin_pitch = max(-1.0, min(1.0, -float(matrix[1, 2])))
    pitch = math.asin(sin_pitch)
    if abs(math.cos(pitch)) > 1e-6:
        yaw = math.atan2(float(matrix[0, 2]), float(matrix[2, 2]))
        roll = math.atan2(float(matrix[1, 0]), float(matrix[1, 1]))
    else:
        yaw = math.atan2(-float(matrix[2, 0]), float(matrix[0, 0]))
        roll = 0.0
    return yaw, pitch, roll


def quat_to_yxz_degrees(quat):
    """BVH channels Yrotation, Xrotation, Zrotation, in degrees."""
    yaw, pitch, roll = _yxz_from_matrix(_matrix_from_quat(quat))
    return (
        math.degrees(yaw),
        math.degrees(pitch),
        math.degrees(roll),
    )


def forward_kinematics(local_position, local_rotation_wxyz, parents):
    """Local parent-frame transforms to global transforms. One frame."""
    count = len(parents)
    global_position = np.zeros((count, 3), dtype=np.float64)
    global_rotation = np.zeros((count, 4), dtype=np.float64)
    for index, parent in enumerate(parents):
        rotation = _normalize_wxyz(local_rotation_wxyz[index])
        position = np.asarray(local_position[index], dtype=np.float64)
        if int(parent) < 0:
            global_position[index] = position
            global_rotation[index] = rotation
            continue
        parent_rotation = global_rotation[int(parent)]
        global_rotation[index] = _quat_mul_wxyz(parent_rotation, rotation)
        global_position[index] = global_position[int(parent)] + _rotate_wxyz(
            parent_rotation, position
        )
    return global_position, global_rotation


def globals_from_locals(local_position_cm, local_rotation_wxyz, parents):
    """Native centimetre globals, and the ROS-metre globals used in rviz."""
    frames, joints, _ = local_position_cm.shape
    global_cm = np.empty((frames, joints, 3), dtype=np.float64)
    global_wxyz = np.empty((frames, joints, 4), dtype=np.float64)
    global_ros_m = np.empty((frames, joints, 3), dtype=np.float64)
    global_ros_xyzw = np.empty((frames, joints, 4), dtype=np.float64)
    for frame in range(frames):
        ros_local_m = np.empty((joints, 3), dtype=np.float64)
        ros_local_wxyz = np.empty((joints, 4), dtype=np.float64)
        for joint in range(joints):
            translation, rotation_xyzw = _mocap_pose_to_ros(
                local_position_cm[frame, joint],
                local_rotation_wxyz[frame, joint],
            )
            ros_local_m[joint] = translation
            ros_local_wxyz[joint] = _xyzw_to_wxyz(rotation_xyzw)
        position_cm, rotation_wxyz = forward_kinematics(
            local_position_cm[frame], local_rotation_wxyz[frame], parents
        )
        position_m, rotation_ros = forward_kinematics(
            ros_local_m, ros_local_wxyz, parents
        )
        global_cm[frame] = position_cm
        global_wxyz[frame] = rotation_wxyz
        global_ros_m[frame] = position_m
        global_ros_xyzw[frame] = np.stack(
            [_wxyz_to_xyzw(rotation_ros[joint]) for joint in range(joints)]
        )
    return global_cm, global_wxyz, global_ros_m, global_ros_xyzw


def _children(parents):
    children = [[] for _ in parents]
    for index, parent in enumerate(parents):
        if int(parent) >= 0:
            children[int(parent)].append(index)
    return children


def _bvh_offsets(offset_cm):
    """Root OFFSET is zero so its position channels stay absolute."""
    offsets = np.array(offset_cm, dtype=np.float64, copy=True)
    offsets[0] = 0.0
    return offsets


def _write_bvh_joint(lines, index, joint_names, children, offsets, depth):
    indent = "\t" * depth
    kind = "ROOT" if depth == 0 else "JOINT"
    ox, oy, oz = (float(value) for value in offsets[index])
    lines.append(f"{indent}{kind} {joint_names[index]}")
    lines.append(f"{indent}{{")
    lines.append(f"{indent}\tOFFSET {ox:.6f} {oy:.6f} {oz:.6f}")
    lines.append(f"{indent}\tCHANNELS 6 {_BVH_CHANNELS}")
    if children[index]:
        for child in children[index]:
            _write_bvh_joint(
                lines, child, joint_names, children, offsets, depth + 1
            )
    else:
        # Display stub only. It has no channels and is not a recorded joint.
        lines.append(f"{indent}\tEnd Site")
        lines.append(f"{indent}\t{{")
        lines.append(f"{indent}\t\tOFFSET 0.000000 5.000000 0.000000")
        lines.append(f"{indent}\t}}")
    lines.append(f"{indent}}}")


def write_bvh(
    path, joint_names, parents, offset_cm, local_position_cm, local_rotation_wxyz, frame_time
):
    offsets = _bvh_offsets(offset_cm)
    lines = ["HIERARCHY"]
    _write_bvh_joint(lines, 0, joint_names, _children(parents), offsets, 0)
    lines.append("MOTION")
    lines.append(f"Frames: {len(local_position_cm)}")
    lines.append(f"Frame Time: {float(frame_time):.6f}")
    for frame in range(len(local_position_cm)):
        values = []
        for joint in range(len(joint_names)):
            # Absolute local translation. Blender treats the channel as the
            # current translation and subtracts OFFSET, so a delta here would
            # pull every bone back by its rest length.
            position = local_position_cm[frame, joint]
            yaw, pitch, roll = quat_to_yxz_degrees(local_rotation_wxyz[frame, joint])
            values.extend(
                (
                    float(position[0]),
                    float(position[1]),
                    float(position[2]),
                    yaw,
                    pitch,
                    roll,
                )
            )
        lines.append(" ".join(f"{value:.6f}" for value in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_bvh(path):
    """Read a clip written by ``write_bvh``.

    Returns joint names, OFFSET rows, absolute local positions in
    centimetres, and YXZ Euler angles in degrees. Leaf end sites are
    ignored. Position channels are the absolute local translation.
    """
    joint_names = []
    offsets = []
    expect_offset = False
    motion_rows = []
    frame_time = None
    in_motion = False
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if not in_motion:
            if parts[0] in ("ROOT", "JOINT"):
                joint_names.append(parts[1])
                expect_offset = True
            elif parts[0] == "End":
                expect_offset = False
            elif parts[0] == "OFFSET" and expect_offset:
                offsets.append([float(parts[1]), float(parts[2]), float(parts[3])])
                expect_offset = False
            elif parts[0] == "MOTION":
                in_motion = True
            continue
        if parts[0] == "Frames:":
            continue
        if parts[0] == "Frame":
            frame_time = float(parts[2])
            continue
        motion_rows.append([float(value) for value in parts])

    offsets = np.asarray(offsets, dtype=np.float64)
    motion = np.asarray(motion_rows, dtype=np.float64)
    joints = len(joint_names)
    if offsets.shape != (joints, 3):
        raise ValueError(f"{path} has {joints} joints and {len(offsets)} offsets")
    if motion.ndim != 2 or motion.shape[1] != joints * 6:
        raise ValueError(f"{path} motion width is not 6 channels per joint")
    local_position = np.empty((len(motion), joints, 3), dtype=np.float64)
    euler_yxz_deg = np.empty((len(motion), joints, 3), dtype=np.float64)
    for joint in range(joints):
        start = joint * 6
        local_position[:, joint, :] = motion[:, start : start + 3]
        euler_yxz_deg[:, joint, :] = motion[:, start + 3 : start + 6]
    return {
        "joint_names": joint_names,
        "offset_cm": offsets,
        "local_position_cm": local_position,
        "euler_yxz_deg": euler_yxz_deg,
        "frame_time_s": frame_time,
    }


def _fps(timestamps_s):
    deltas = np.diff(np.asarray(timestamps_s, dtype=np.float64))
    deltas = deltas[deltas > 1e-4]
    if len(deltas) == 0:
        return 100.0
    return float(1.0 / np.median(deltas))


def _append_index(path, row):
    fieldnames = (
        "take",
        "frames",
        "duration_s",
        "fps_hz",
        "bvh",
        "npz",
        "csv",
    )
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _write_csv(
    path,
    timestamps_s,
    joint_names,
    local_position_cm,
    local_rotation_wxyz,
    global_position_ros_m,
    robot_raw_names,
    robot_raw_angle,
    robot_joint_names,
    robot_joint_position_rad,
    robot_root_position_m,
    robot_root_rotation_xyzw,
):
    header = ["time_s"]
    for name in joint_names:
        header.extend(
            (
                f"{name}_local_x_cm",
                f"{name}_local_y_cm",
                f"{name}_local_z_cm",
                f"{name}_local_qw",
                f"{name}_local_qx",
                f"{name}_local_qy",
                f"{name}_local_qz",
                f"{name}_global_ros_x_m",
                f"{name}_global_ros_y_m",
                f"{name}_global_ros_z_m",
            )
        )
    header.extend(f"raw_{name}" for name in robot_raw_names)
    header.extend(f"retarget_{name}_rad" for name in robot_joint_names)
    header.extend(
        (
            "robot_root_x_m",
            "robot_root_y_m",
            "robot_root_z_m",
            "robot_root_qx",
            "robot_root_qy",
            "robot_root_qz",
            "robot_root_qw",
        )
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        handle.write(
            "# local columns are the SDK frame, centimetres and quaternion wxyz. "
            "global_ros columns are metres in the /raw_skeleton world. "
            "raw_* is GetRobotRawJointAngle. retarget_*_rad is the current retarget.\n"
        )
        writer = csv.writer(handle)
        writer.writerow(header)
        for frame, time_s in enumerate(timestamps_s):
            row = [f"{float(time_s):.7f}"]
            for joint in range(len(joint_names)):
                row.extend(f"{float(value):.7f}" for value in local_position_cm[frame, joint])
                row.extend(
                    f"{float(value):.7f}" for value in local_rotation_wxyz[frame, joint]
                )
                row.extend(
                    f"{float(value):.7f}" for value in global_position_ros_m[frame, joint]
                )
            row.extend(f"{float(value):.7f}" for value in robot_raw_angle[frame])
            row.extend(f"{float(value):.7f}" for value in robot_joint_position_rad[frame])
            row.extend(f"{float(value):.7f}" for value in robot_root_position_m[frame])
            row.extend(f"{float(value):.7f}" for value in robot_root_rotation_xyzw[frame])
            writer.writerow(row)


def save_take(
    directory,
    timestamps_s,
    offset_cm,
    local_position_cm,
    local_rotation_wxyz,
    sdk_euler_xyz,
    robot_raw_angle,
    robot_joint_names,
    robot_joint_position_rad,
    robot_root_position_m,
    robot_root_rotation_xyzw,
    retarget_config,
    retarget_sha256,
):
    """Write one take. ``directory`` is created and must not already exist."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    timestamps_s = np.asarray(timestamps_s, dtype=np.float64)
    timestamps_s = timestamps_s - float(timestamps_s[0])
    offset_cm = np.asarray(offset_cm, dtype=np.float64)
    local_position_cm = np.asarray(local_position_cm, dtype=np.float64)
    local_rotation_wxyz = np.asarray(local_rotation_wxyz, dtype=np.float64)
    sdk_euler_xyz = np.asarray(sdk_euler_xyz, dtype=np.float64)
    robot_raw_angle = np.asarray(robot_raw_angle, dtype=np.float64)
    robot_joint_position_rad = np.asarray(robot_joint_position_rad, dtype=np.float64)
    robot_root_position_m = np.asarray(robot_root_position_m, dtype=np.float64)
    robot_root_rotation_xyzw = np.asarray(robot_root_rotation_xyzw, dtype=np.float64)

    frames = len(timestamps_s)
    if frames < 2:
        raise ValueError("a take needs at least 2 frames")
    if local_position_cm.shape != (frames, len(JOINT_NAMES), 3):
        raise ValueError("local positions do not match the body skeleton")

    fps = _fps(timestamps_s)
    global_cm, global_wxyz, global_ros_m, global_ros_xyzw = globals_from_locals(
        local_position_cm, local_rotation_wxyz, PARENT_INDEX
    )
    meta = {
        "format": "pnmocap-raw-skeleton-v1",
        "joint_count": len(JOINT_NAMES),
        "fps_hz": fps,
        "frames": frames,
        "bvh_units": "centimetres",
        "bvh_axes": "SDK axes, Y up",
        "bvh_rotation": "Yrotation Xrotation Zrotation",
        "bvh_composition": "R = Ry(Y) @ Rx(X) @ Rz(Z)",
        "bvh_translation": "position channels are the absolute SDK local translation in centimetres. OFFSET is the rest translation. Blender subtracts OFFSET from the channel.",
        "bvh_end_site": "5 cm display stub on leaf joints, not a recorded joint",
        "native_quaternion": "wxyz",
        "ros_position_m": "(x, y, z) = (native_z, native_x, native_y) / 100",
        "ros_quaternion_xyzw": "(x, y, z, w) = (native_z, native_x, native_y, native_w)",
        "robot_raw_joint_angle": "GetRobotRawJointAngle in SDK units; column i is EMCPRobotJointType and retarget.json rawJoint",
        "robot_joint_position_rad": "joint_positions from get_robot_ros_frame_json, radians, urdfJointNames order",
        "robot_root": "solver root, metres and quaternion xyzw, without LIVE_ROOT_Z_BIAS and without teleop planar alignment",
        "sdk_euler_xyz": "GetJointLocalRotationByEuler values, unmodified",
        "retarget_config": str(retarget_config),
        "retarget_sha256": retarget_sha256,
        "alignment": "NPZ row i is BVH frame i. timestamps_s is the reception clock. Frame Time is the median interval.",
    }
    bvh_path = directory / "raw_skeleton.bvh"
    npz_path = directory / "dataset.npz"
    csv_path = directory / "dataset.csv"
    write_bvh(
        bvh_path,
        JOINT_NAMES,
        PARENT_INDEX,
        offset_cm,
        local_position_cm,
        local_rotation_wxyz,
        1.0 / fps,
    )
    np.savez_compressed(
        npz_path,
        timestamps_s=timestamps_s,
        joint_names=np.array(JOINT_NAMES),
        parent_index=PARENT_INDEX,
        parent_name=np.array(
            [
                "" if PARENT_INDEX[index] < 0 else JOINT_NAMES[int(PARENT_INDEX[index])]
                for index in range(len(JOINT_NAMES))
            ]
        ),
        offset_cm=offset_cm,
        local_position_cm=local_position_cm,
        local_rotation_wxyz=local_rotation_wxyz,
        global_position_cm=global_cm,
        global_rotation_wxyz=global_wxyz,
        global_position_ros_m=global_ros_m,
        global_rotation_ros_xyzw=global_ros_xyzw,
        sdk_euler_xyz=sdk_euler_xyz,
        robot_raw_joint_names=np.array(ROBOT_RAW_JOINTS),
        robot_raw_joint_angle=robot_raw_angle,
        robot_joint_names=np.array(robot_joint_names),
        robot_joint_position_rad=robot_joint_position_rad,
        robot_root_position_m=robot_root_position_m,
        robot_root_rotation_xyzw=robot_root_rotation_xyzw,
        fps_hz=np.float64(fps),
        meta_json=np.array(json.dumps(meta)),
    )
    _write_csv(
        csv_path,
        timestamps_s,
        JOINT_NAMES,
        local_position_cm,
        local_rotation_wxyz,
        global_ros_m,
        ROBOT_RAW_JOINTS,
        robot_raw_angle,
        robot_joint_names,
        robot_joint_position_rad,
        robot_root_position_m,
        robot_root_rotation_xyzw,
    )
    duration = float(timestamps_s[-1] - timestamps_s[0])
    _append_index(
        directory.parent / "index.csv",
        {
            "take": directory.name,
            "frames": frames,
            "duration_s": f"{duration:.6f}",
            "fps_hz": f"{fps:.6f}",
            "bvh": str(bvh_path),
            "npz": str(npz_path),
            "csv": str(csv_path),
        },
    )
    return {
        "directory": directory,
        "frames": frames,
        "duration_s": duration,
        "fps_hz": fps,
        "bvh": bvh_path,
        "npz": npz_path,
        "csv": csv_path,
    }


def _finite_stack(frames, key, shape):
    values = np.stack([frame[key] for frame in frames]).astype(np.float64)
    if values.shape[1:] != shape:
        raise ValueError(f"{key} has shape {values.shape}, expected (*, {shape})")
    return values


class MocapRP1Recorder(MCPBase):
    def __init__(self, output_dir, retarget_config):
        super().__init__()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.retarget_config = Path(retarget_config).expanduser().resolve()
        if not self.retarget_config.is_file():
            raise FileNotFoundError(f"RP1 retarget config not found: {self.retarget_config}")
        config_text = self.retarget_config.read_text(encoding="utf-8")
        joint_names = json.loads(config_text)["urdfJointNames"]
        if len(joint_names) != 28:
            raise ValueError(
                f"{self.retarget_config} must list 28 urdfJointNames, got {len(joint_names)}"
            )
        self.robot_joint_names = tuple(joint_names)
        self.retarget_sha256 = hashlib.sha256(config_text.encode("utf-8")).hexdigest()
        self.robot = MCPRobot(config_text)
        self._recording = False
        self._frames = []
        self._offsets = None
        self._dropped = 0
        self._missing_reported = False
        self._robot_error_reported = False
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(
            "录制身体骨骼。先按 N 开始采集，S 开始一条，E 结束并保存。"
            "ESC 退出时如果还在录，会先保存。"
        )
        print("不要同时运行 mocap_rp1_reference.py，两者都会占用 PN-Link。")
        print(f"文件写到 {self.output_dir}")

    def dispatch_key(self, key_name):
        if key_name == "s":
            self._start_take()
            return
        if key_name == "e":
            self._finish_take("E")
            return
        super().dispatch_key(key_name)

    def _start_take(self):
        if self._recording:
            print(f"已经在录，当前 {len(self._frames)} 帧。按 E 结束并保存。")
            return
        self._frames = []
        self._offsets = None
        self._dropped = 0
        self._missing_reported = False
        self._robot_error_reported = False
        self._recording = True
        if self.capture_key:
            print("S：开始录制。按 E 结束并保存。")
        else:
            print("S：开始录制。先按 N 启动采集，动作数据进来之后才会写入帧。")

    def _finish_take(self, reason):
        if not self._recording and not self._frames:
            print("当前没有未保存的录制。按 S 开始。")
            return
        self._recording = False
        frames = self._frames
        self._frames = []
        if len(frames) < 2:
            print(f"{reason}：只有 {len(frames)} 帧，没有保存。")
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory = self.output_dir / stamp
        suffix = 2
        while directory.exists():
            directory = self.output_dir / f"{stamp}_{suffix}"
            suffix += 1
        try:
            summary = save_take(
                directory,
                _finite_stack(frames, "time_s", ()),
                self._offsets,
                _finite_stack(frames, "local_position_cm", (len(JOINT_NAMES), 3)),
                _finite_stack(frames, "local_rotation_wxyz", (len(JOINT_NAMES), 4)),
                _finite_stack(frames, "sdk_euler_xyz", (len(JOINT_NAMES), 3)),
                _finite_stack(frames, "robot_raw_angle", (len(ROBOT_RAW_JOINTS),)),
                self.robot_joint_names,
                _finite_stack(
                    frames, "robot_joint_position_rad", (len(self.robot_joint_names),)
                ),
                _finite_stack(frames, "robot_root_position_m", (3,)),
                _finite_stack(frames, "robot_root_rotation_xyzw", (4,)),
                self.retarget_config,
                self.retarget_sha256,
            )
        except Exception as exc:
            self._frames = frames
            self._recording = True
            print(f"保存失败，录制还在继续: {exc}")
            return
        self._offsets = None
        print(
            f"已保存 {summary['frames']} 帧, {summary['duration_s']:.2f} 秒, "
            f"{summary['fps_hz']:.1f} Hz。丢弃不完整帧 {self._dropped}。"
        )
        print(f"  动捕文件 {summary['bvh']}")
        print(f"  数据集   {summary['npz']}")
        print(f"  表格     {summary['csv']}")

    def _report_robot_error(self, exc):
        if self._robot_error_reported:
            return
        print(f"这一帧没有完整的机器人关节，骨骼仍会保存: {exc}")
        self._robot_error_reported = True

    def handleAvatar(self, avatar_handle):
        if not self._recording:
            return
        try:
            frame = self._capture_frame(MCPAvatar(avatar_handle))
        except Exception as exc:
            self._dropped += 1
            if not self._missing_reported:
                print(f"跳过一帧: {exc}")
                self._missing_reported = True
            return
        if frame is None:
            return
        self._frames.append(frame)
        count = len(self._frames)
        if count == 1 or count % 100 == 0:
            hips = frame["local_position_cm"][0]
            print(
                f"录制中 {count} 帧, Hips cm "
                f"{hips[0]:.1f} {hips[1]:.1f} {hips[2]:.1f}"
            )

    def _capture_frame(self, avatar):
        found = {}
        eulers = {}
        defaults = {}
        for joint in avatar.get_joints():
            name = joint.get_name()
            if name not in JOINT_INDEX:
                continue
            position = joint.get_local_position()
            if position is None:
                continue
            rotation = joint.get_local_rotation()
            found[name] = (
                np.asarray(position, dtype=np.float64),
                _normalize_wxyz(rotation),
            )
            try:
                eulers[name] = tuple(float(value) for value in joint.get_local_rotation_by_euler())
            except Exception:
                eulers[name] = (math.nan, math.nan, math.nan)
            try:
                defaults[name] = np.asarray(
                    joint.get_default_local_position(), dtype=np.float64
                )
            except Exception:
                defaults[name] = None

        missing = [name for name in JOINT_NAMES if name not in found]
        if missing:
            self._dropped += 1
            if not self._missing_reported:
                print(f"跳过不完整帧，缺少 {', '.join(missing)}")
                self._missing_reported = True
            return None

        local_position = np.stack([found[name][0] for name in JOINT_NAMES])
        local_rotation = np.stack([found[name][1] for name in JOINT_NAMES])
        sdk_euler = np.asarray([eulers[name] for name in JOINT_NAMES], dtype=np.float64)
        if self._offsets is None:
            offsets = []
            for name in JOINT_NAMES:
                default = defaults[name]
                if default is None or not np.all(np.isfinite(default)):
                    default = found[name][0]
                offsets.append(np.asarray(default, dtype=np.float64))
            self._offsets = np.stack(offsets)

        robot_raw = np.full(len(ROBOT_RAW_JOINTS), np.nan, dtype=np.float64)
        robot_joints = np.full(len(self.robot_joint_names), np.nan, dtype=np.float64)
        robot_root_position = np.full(3, np.nan, dtype=np.float64)
        robot_root_rotation = np.full(4, np.nan, dtype=np.float64)
        robot_ready = False
        try:
            self.robot.update_robot(avatar)
            self.robot.run_robot_step()
            robot_ready = True
        except Exception as exc:
            self._report_robot_error(exc)
        if robot_ready:
            for index in range(len(ROBOT_RAW_JOINTS)):
                try:
                    robot_raw[index] = float(self.robot.get_robot_raw_joint_angle(index))
                except Exception as exc:
                    self._report_robot_error(exc)
            try:
                positions = json.loads(self.robot.get_robot_ros_frame_json()[0]).get(
                    "joint_positions", {}
                )
                robot_joints[:] = [
                    float(positions.get(name, math.nan)) for name in self.robot_joint_names
                ]
                robot_root_position[:] = self.robot.get_robot_root_position()
                robot_root_rotation[:] = self.robot.get_robot_root_rotataion()
            except Exception as exc:
                self._report_robot_error(exc)

        return {
            "time_s": np.float64(time.perf_counter()),
            "local_position_cm": local_position,
            "local_rotation_wxyz": local_rotation,
            "sdk_euler_xyz": sdk_euler,
            "robot_raw_angle": robot_raw,
            "robot_joint_position_rad": robot_joints,
            "robot_root_position_m": robot_root_position,
            "robot_root_rotation_xyzw": robot_root_rotation,
        }

    async def main_async(self):
        try:
            await super().main_async()
        finally:
            if self._recording or self._frames:
                self._finish_take("退出")

    def main(self):
        asyncio.run(self.main_async())


def _assert_close(name, actual, expected, tolerance=1e-6):
    if not np.allclose(actual, expected, atol=tolerance, rtol=tolerance):
        raise AssertionError(f"{name}: {actual} != {expected}")


def self_test():
    """Check the Euler convention, kinematics, and the three output files."""
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    turned = _quat_from_axis_angle((0.0, 1.0, 0.0), math.pi / 2.0)
    _assert_close("Ry vector", _rotate_wxyz(turned, (0.0, 0.0, 1.0)), (1.0, 0.0, 0.0))
    _assert_close(
        "Ry matrix",
        _matrix_from_quat(turned) @ np.array([0.0, 0.0, 1.0]),
        (1.0, 0.0, 0.0),
    )

    rng = np.random.default_rng(0)
    for _ in range(40):
        yaw, pitch, roll = rng.uniform(-2.2, 2.2, 3)
        if abs(abs(pitch) - math.pi / 2.0) < 0.15:
            continue
        matrix = _matrix_from_yxz(yaw, pitch, roll)
        _assert_close(
            "quat matrix",
            _matrix_from_quat(_quat_from_yxz(yaw, pitch, roll)),
            matrix,
        )
        recovered = _yxz_from_matrix(matrix)
        _assert_close("matrix roundtrip", _matrix_from_yxz(*recovered), matrix)
        # asin keeps Xrotation inside [-90, 90], so the angles match there.
        if abs(pitch) < 1.2:
            _assert_close("euler", recovered, (yaw, pitch, roll))
    singular = _matrix_from_yxz(0.4, math.pi / 2.0, 0.25)
    singular_back = _matrix_from_yxz(*_yxz_from_matrix(singular))
    _assert_close("gimbal matrix", singular_back, singular, tolerance=1e-6)

    ros_translation, ros_rotation = _mocap_pose_to_ros((1.0, 2.0, 3.0), identity)
    _assert_close("ros position", ros_translation, (0.03, 0.01, 0.02))
    _assert_close("ros rotation", ros_rotation, (0.0, 0.0, 0.0, 1.0))

    joints = len(JOINT_NAMES)
    offset_cm = np.zeros((joints, 3), dtype=np.float64)
    offset_cm[0] = (0.0, 90.0, 0.0)
    for index, parent in enumerate(PARENT_INDEX):
        if index == 0:
            continue
        offset_cm[index] = (0.0, 10.0, 0.0) if JOINT_NAMES[index] == "Spine" else (5.0, 0.0, 0.0)
        if int(parent) < 0:
            raise AssertionError("only Hips is the root")

    def pose(hips, spine_yaw):
        position = np.array(offset_cm, copy=True)
        position[0] = hips
        rotation = np.tile(identity, (joints, 1))
        rotation[JOINT_INDEX["Spine"]] = _quat_from_yxz(spine_yaw, 0.0, 0.0)
        return position, rotation

    positions = []
    rotations = []
    for hips, yaw in (((0.0, 100.0, 0.0), 0.0), ((10.0, 100.0, 5.0), math.radians(30.0))):
        position, rotation = pose(hips, yaw)
        positions.append(position)
        rotations.append(rotation)
    local_position_cm = np.stack(positions)
    local_rotation_wxyz = np.stack(rotations)
    timestamps = np.array([10.0, 10.02], dtype=np.float64)
    robot_names = tuple(f"joint_{index}" for index in range(28))
    robot_raw = np.arange(2 * len(ROBOT_RAW_JOINTS), dtype=np.float64).reshape(
        2, len(ROBOT_RAW_JOINTS)
    )
    robot_joints = np.full((2, 28), 0.25, dtype=np.float64)
    robot_root_position = np.zeros((2, 3), dtype=np.float64)
    robot_root_rotation = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (2, 1))
    sdk_euler = np.zeros((2, joints, 3), dtype=np.float64)

    import tempfile

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        summary = save_take(
            root / "take",
            timestamps,
            offset_cm,
            local_position_cm,
            local_rotation_wxyz,
            sdk_euler,
            robot_raw,
            robot_names,
            robot_joints,
            robot_root_position,
            robot_root_rotation,
            DEFAULT_RETARGET_CONFIG,
            "test",
        )
        with np.load(summary["npz"]) as dataset:
            _assert_close(
                "stored hips",
                dataset["global_position_cm"][0, 0],
                (0.0, 100.0, 0.0),
            )
            _assert_close(
                "stored spine",
                dataset["global_position_cm"][0, JOINT_INDEX["Spine"]],
                (0.0, 110.0, 0.0),
            )
            _assert_close(
                "stored ros hips",
                dataset["global_position_ros_m"][0, 0],
                (0.0, 0.0, 1.0),
            )
            _assert_close(
                "stored ros spine",
                dataset["global_position_ros_m"][0, JOINT_INDEX["Spine"]],
                (0.0, 0.0, 1.1),
            )
            _assert_close("raw angle", dataset["robot_raw_joint_angle"][1, 3], robot_raw[1, 3])
            meta = json.loads(dataset["meta_json"].item())
            if meta["format"] != "pnmocap-raw-skeleton-v1":
                raise AssertionError(meta["format"])
            if abs(float(dataset["fps_hz"]) - 50.0) > 1e-6:
                raise AssertionError(dataset["fps_hz"])

        loaded = load_bvh(summary["bvh"])
        if loaded["joint_names"] != list(JOINT_NAMES):
            raise AssertionError(loaded["joint_names"])
        _assert_close("bvh root offset", loaded["offset_cm"][0], (0.0, 0.0, 0.0))
        _assert_close("bvh locals", loaded["local_position_cm"], local_position_cm)
        _assert_close(
            "bvh spine offset",
            loaded["offset_cm"][JOINT_INDEX["Spine"]],
            offset_cm[JOINT_INDEX["Spine"]],
        )
        yaw, pitch, roll = (math.radians(value) for value in loaded["euler_yxz_deg"][1, JOINT_INDEX["Spine"]])
        recovered = _quat_from_yxz(yaw, pitch, roll)
        if abs(abs(float(np.dot(recovered, local_rotation_wxyz[1, JOINT_INDEX["Spine"]]))) - 1.0) > 1e-5:
            raise AssertionError((recovered, local_rotation_wxyz[1, JOINT_INDEX["Spine"]]))

        with summary["csv"].open(encoding="utf-8") as handle:
            next(handle)
            rows = list(csv.DictReader(handle))
        if len(rows) != 2:
            raise AssertionError(len(rows))
        if abs(float(rows[1]["Spine_global_ros_z_m"]) - float(dataset_spine_z(summary["npz"]))) > 1e-5:
            raise AssertionError(rows[1]["Spine_global_ros_z_m"])
        index_path = root / "index.csv"
        if not index_path.is_file():
            raise AssertionError("index.csv was not written")
    print("self-test passed")


def dataset_spine_z(path):
    with np.load(path) as dataset:
        return float(dataset["global_position_ros_m"][1, JOINT_INDEX["Spine"], 2])


def main():
    parser = argparse.ArgumentParser(
        description="Record the PN-Link body skeleton as BVH, NPZ, and CSV."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for takes (default: mocapapi_python/recordings)",
    )
    parser.add_argument(
        "--retarget",
        type=Path,
        default=DEFAULT_RETARGET_CONFIG,
        help="retarget.json used to store the current robot joint angles",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="check the writer without opening PN-Link",
    )
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    MocapRP1Recorder(args.output, args.retarget).main()


if __name__ == "__main__":
    main()
