"""Publish a live PN-Link retarget as the RP1 motion-tracking reference.

The inference node in rp1_deploy_new subscribes to
``/tracking/current_reference`` as a 35-float array:

    [root_x, root_y, root_z, root_qw, root_qx, root_qy, root_qz, joint_0 .. joint_27]

Root position is metres. The quaternion is wxyz. Joints are radians in the
28-DoF order stored in ``rp1/retarget.json``, which matches the inference
``joint_default_angle`` order. The message is ``std_msgs/Float32MultiArray``,
published with the topic, rate, and QoS in ``rp1/publish.yaml``, the same
message contract as ``roboparty_teleop`` ``scripts/rp1_deploy/run_local.sh``.
Raw finger poses are ``geometry_msgs/PoseArray`` plus a ``std_msgs/Bool`` active
flag. Each hand contains up to 19 PN finger joints, excluding the wrist, using
the same message types as ``/pico/hand/*/poses``.

Press N in this terminal to start capture. A follows the live retarget, X
returns to the default stand. Both blend over two seconds. X keeps the
current ground position and yaw. A starts x, y, and yaw from the pose being
published now, then follows the mocap's changes. Do not run another publisher
on the same topic, and do not run ``mocap_main_rp1_ros2.py`` at the same time:
both open the PN-Link connection.
"""

import asyncio
import json
import math
from pathlib import Path

import rclpy
from geometry_msgs.msg import Point, Pose, PoseArray, TransformStamped
from rclpy.node import Node
from visualization_msgs.msg import Marker
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32MultiArray
from tf2_ros import TransformBroadcaster

from pn_sdk import load_sdk

_sdk, MCPBase = load_sdk()
MCPAvatar, MCPRobot = _sdk.MCPAvatar, _sdk.MCPRobot

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RETARGET_CONFIG = SCRIPT_DIR / "rp1" / "retarget.json"
DEFAULT_PUBLISH_CONFIG = SCRIPT_DIR / "rp1" / "publish.yaml"
REFERENCE_TOPIC = "/tracking/current_reference"
PUBLISH_RATE_HZ = 50.0
JOINT_COUNT = 28
VECTOR_SIZE = 7 + JOINT_COUNT
SOURCE_TIMEOUT_S = 1.0
TRANSITION_STEPS = 100
# This actor's standing root was 0.922 m. The URDF sole is 0.776 m below
# base_link, so the live root is published 0.146 m lower. Idle stand is unchanged.
LIVE_ROOT_Z_BIAS = 0.146
RAW_SKELETON_TOPIC = "/raw_skeleton"
# Parent of each mocap joint. Frame ids are prefixed so they do not collide
# with the RP1 robot links.
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


def _hand_joint_order(side):
    """Thumb, then four fingers; 19 PN joints without the wrist."""
    names = [
        f"{side}HandThumb1",
        f"{side}HandThumb2",
        f"{side}HandThumb3",
    ]
    for finger in ("Index", "Middle", "Ring", "Pinky"):
        names.append(f"{side}InHand{finger}")
        names.extend(f"{side}Hand{finger}{index}" for index in (1, 2, 3))
    return tuple(names)


HAND_JOINT_ORDER = {
    "left": _hand_joint_order("Left"),
    "right": _hand_joint_order("Right"),
}


def _with_hand_parents(parents):
    extended = dict(parents)
    for side in ("Left", "Right"):
        hand = f"{side}Hand"
        extended[f"{side}HandThumb1"] = hand
        extended[f"{side}HandThumb2"] = f"{side}HandThumb1"
        extended[f"{side}HandThumb3"] = f"{side}HandThumb2"
        for finger in ("Index", "Middle", "Ring", "Pinky"):
            root = f"{side}InHand{finger}"
            extended[root] = hand
            previous = root
            for index in (1, 2, 3):
                name = f"{side}Hand{finger}{index}"
                extended[name] = previous
                previous = name
    return extended


POSE_PARENTS = _with_hand_parents(RAW_PARENTS)

# Same stand as roboparty_teleop configs/rp1_deploy/teleop_ros2_28dof.yaml
# and rp1_deploy_new motion_tracking_teleop joint_default_angle.
# Quaternion is wxyz.
IDLE_REFERENCE = [
    0.0, 0.0, 0.78,
    1.0, 0.0, 0.0, 0.0,
    -0.1, 0.0, 0.0, -0.3, -0.2, 0.0,
    0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    0.0, 0.0,
    0.2, 0.2, 0.0, -1.2, 0.0, 0.0, 0.0,
    -0.2, -0.2, 0.0, 1.2, 0.0, 0.0, 0.0,
]

REFERENCE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=21,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


def _mocap_pose_to_ros(position_cm, rotation_wxyz):
    """Match the existing stickman axis map: cm, (w, x, y, z) to metres, xyzw."""
    w, x, y, z = rotation_wxyz
    translation = (
        float(position_cm[2]) / 100.0,
        float(position_cm[0]) / 100.0,
        float(position_cm[1]) / 100.0,
    )
    rotation = (float(z), float(x), float(y), float(w))
    return translation, rotation


def _quat_multiply(lhs, rhs):
    x1, y1, z1, w1 = lhs
    x2, y2, z2, w2 = rhs
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _rotate_vector(rotation, vector):
    x, y, z, w = rotation
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def _normalize_xyzw(quat):
    x, y, z, w = (float(value) for value in quat)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-8:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / norm, y / norm, z / norm, w / norm)


def _compose_world(local_poses, names):
    world = {}

    def walk(name):
        if name in world:
            return world[name]
        if name not in local_poses:
            return None
        translation, rotation = local_poses[name]
        parent = POSE_PARENTS.get(name, "world")
        if parent == "world":
            pose = (translation, _normalize_xyzw(rotation))
        else:
            parent_pose = walk(parent)
            if parent_pose is None:
                return None
            parent_translation, parent_rotation = parent_pose
            rotated = _rotate_vector(parent_rotation, translation)
            pose = (
                tuple(a + b for a, b in zip(parent_translation, rotated)),
                _normalize_xyzw(_quat_multiply(parent_rotation, rotation)),
            )
        world[name] = pose
        return pose

    return {name: walk(name) for name in names}


def _normalize_wxyz(quat):
    w, x, y, z = (float(value) for value in quat)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-8:
        return (1.0, 0.0, 0.0, 0.0)
    return (w / norm, x / norm, y / norm, z / norm)


def _slerp_wxyz(start, end, alpha):
    q0 = list(_normalize_wxyz(start))
    q1 = list(_normalize_wxyz(end))
    if sum(a * b for a, b in zip(q0, q1)) < 0.0:
        q1 = [-value for value in q1]
    return _normalize_wxyz(
        [(1.0 - alpha) * a + alpha * b for a, b in zip(q0, q1)]
    )


def _blend(start, target, alpha):
    mixed = [(1.0 - alpha) * a + alpha * b for a, b in zip(start, target)]
    mixed[3:7] = list(_slerp_wxyz(start[3:7], target[3:7], alpha))
    return mixed


def _wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _yaw_from_wxyz(quat):
    w, x, y, z = _normalize_wxyz(quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _roll_pitch_from_wxyz(quat):
    w, x, y, z = _normalize_wxyz(quat)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    return roll, math.asin(sin_pitch)


def _wxyz_from_rpy(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _stand_at(x, y, yaw):
    """Default stand with the ground position and yaw held fixed."""
    target = list(IDLE_REFERENCE)
    target[0] = float(x)
    target[1] = float(y)
    target[3:7] = _wxyz_from_rpy(0.0, 0.0, float(yaw))
    return target


def _align_planar(vector, origin):
    """Shift live x, y, and yaw so they start at the robot pose in origin."""
    mocap_x, mocap_y, mocap_yaw, robot_x, robot_y, robot_yaw = origin
    live_yaw = _yaw_from_wxyz(vector[3:7])
    roll, pitch = _roll_pitch_from_wxyz(vector[3:7])
    delta_yaw = _wrap_angle(live_yaw - mocap_yaw)
    heading = _wrap_angle(robot_yaw - mocap_yaw)
    cosine = math.cos(heading)
    sine = math.sin(heading)
    dx = float(vector[0]) - mocap_x
    dy = float(vector[1]) - mocap_y
    aligned = list(vector)
    aligned[0] = robot_x + cosine * dx - sine * dy
    aligned[1] = robot_y + sine * dx + cosine * dy
    aligned[3:7] = _wxyz_from_rpy(roll, pitch, _wrap_angle(robot_yaw + delta_yaw))
    return aligned


def _reference_vector(position, rotation_xyzw, joint_positions) -> list[float]:
    """Pack solver output into the 35-D inference vector."""
    if len(position) != 3 or len(rotation_xyzw) != 4:
        raise ValueError("root pose must be xyz and quaternion xyzw")
    if len(joint_positions) != JOINT_COUNT:
        raise ValueError(f"expected {JOINT_COUNT} joint positions")
    x, y, z, w = (float(value) for value in rotation_xyzw)
    vector = [float(position[0]), float(position[1]), float(position[2]), w, x, y, z]
    vector.extend(float(value) for value in joint_positions)
    if len(vector) != VECTOR_SIZE or not all(math.isfinite(value) for value in vector):
        raise ValueError("reference vector must contain 35 finite values")
    return vector


def _parse_scalar(value):
    if value in ("true", "True"):
        return True
    if value in ("false", "False"):
        return False
    try:
        if any(mark in value for mark in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_simple_yaml(path):
    """Load the two-level publish.yaml without a third-party YAML package."""
    root = {}
    stack = [(-1, root)]
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        stripped = raw.strip()
        if ":" not in stripped:
            raise ValueError(f"{path}:{line_number}: expected key: value")
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.split("#", 1)[0].strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if not isinstance(parent, dict):
            raise ValueError(f"{path}:{line_number}: cannot nest under a value")
        if value == "":
            child = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _require_topic(config, section, key):
    topic = str(config.get(section, {}).get(key, "")).strip()
    if not topic.startswith("/"):
        raise ValueError(f"{section}.{key} must be an absolute ROS topic")
    return topic


def _require_rate(config, section):
    rate = float(config.get(section, {}).get("rate_hz", 0))
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError(f"{section}.rate_hz must be greater than zero")
    return rate


def _qos_from(section):
    reliability = str(section.get("reliability", "reliable")).lower()
    if reliability not in ("reliable", "best_effort"):
        raise ValueError("reliability must be reliable or best_effort")
    depth = int(section.get("qos_depth", 1))
    if depth < 1:
        raise ValueError("qos_depth must be at least 1")
    policy = (
        ReliabilityPolicy.RELIABLE
        if reliability == "reliable"
        else ReliabilityPolicy.BEST_EFFORT
    )
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=policy,
        durability=DurabilityPolicy.VOLATILE,
    )


def _load_publish_config(path):
    if not path.is_file():
        raise FileNotFoundError(f"publish config not found: {path}")
    raw = _load_simple_yaml(path)
    reference = raw.get("reference", {})
    hands = raw.get("hands", {})
    transition_steps = int(reference.get("transition_steps", TRANSITION_STEPS))
    source_timeout_ms = float(reference.get("source_timeout_ms", SOURCE_TIMEOUT_S * 1000.0))
    if transition_steps < 1:
        raise ValueError("reference.transition_steps must be at least 1")
    if not math.isfinite(source_timeout_ms) or source_timeout_ms <= 0.0:
        raise ValueError("reference.source_timeout_ms must be greater than zero")
    return {
        "path": path,
        "reference_topic": _require_topic(raw, "reference", "topic"),
        "reference_rate_hz": _require_rate(raw, "reference"),
        "reference_qos": _qos_from(reference),
        "transition_steps": transition_steps,
        "source_timeout_s": source_timeout_ms / 1000.0,
        "hands_enabled": bool(hands.get("enabled", True)),
        "hand_rate_hz": _require_rate(raw, "hands") if hands.get("enabled", True) else 0.0,
        "hand_frame_id": str(hands.get("frame_id", "world")),
        "hand_qos": _qos_from(hands) if hands.get("enabled", True) else None,
        "hand_topics": {
            "left_poses": _require_topic(raw, "hands", "left_poses"),
            "right_poses": _require_topic(raw, "hands", "right_poses"),
            "left_active": _require_topic(raw, "hands", "left_active"),
            "right_active": _require_topic(raw, "hands", "right_active"),
        },
    }


class MocapRP1Reference(MCPBase):
    def __init__(self):
        super().__init__()
        rclpy.init()
        self.node = Node("mocap_rp1_reference")
        config_path = Path(
            self.node.declare_parameter(
                "retarget_config", str(DEFAULT_RETARGET_CONFIG)
            ).value
        ).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"RP1 retarget config not found: {config_path}")

        config_text = config_path.read_text(encoding="utf-8")
        joint_names = json.loads(config_text)["urdfJointNames"]
        if len(joint_names) != JOINT_COUNT:
            raise ValueError(
                f"{config_path} must list {JOINT_COUNT} urdfJointNames, "
                f"got {len(joint_names)}"
            )
        self.joint_names = joint_names
        self.robot = MCPRobot(config_text)
        self._latest = None
        self._latest_time = 0.0
        self._mode = "idle"
        self._output = list(IDLE_REFERENCE)
        self._blend_from = None
        self._blend_steps = 0
        self._planar_hold = (0.0, 0.0, 0.0)
        self._live_origin = None
        self._published = 0
        self._hand_snapshot = {"left": None, "right": None}
        publish_config_path = Path(
            self.node.declare_parameter(
                "publish_config", str(DEFAULT_PUBLISH_CONFIG)
            ).value
        ).expanduser().resolve()
        self._publish_config = _load_publish_config(publish_config_path)
        self._reference_topic = self._publish_config["reference_topic"]
        self._reference_rate_hz = self._publish_config["reference_rate_hz"]
        self._transition_steps = self._publish_config["transition_steps"]
        self._source_timeout_s = self._publish_config["source_timeout_s"]
        self._hands_enabled = self._publish_config["hands_enabled"]
        self._hand_rate_hz = self._publish_config["hand_rate_hz"]
        self._hand_frame_id = self._publish_config["hand_frame_id"]
        self.publisher = self.node.create_publisher(
            Float32MultiArray,
            self._reference_topic,
            self._publish_config["reference_qos"],
        )
        if self._hands_enabled:
            topics = self._publish_config["hand_topics"]
            hand_qos = self._publish_config["hand_qos"]
            self._hand_pose_publishers = {
                "left": self.node.create_publisher(PoseArray, topics["left_poses"], hand_qos),
                "right": self.node.create_publisher(PoseArray, topics["right_poses"], hand_qos),
            }
            self._hand_active_publishers = {
                "left": self.node.create_publisher(Bool, topics["left_active"], hand_qos),
                "right": self.node.create_publisher(Bool, topics["right_active"], hand_qos),
            }
        self.joint_publisher = self.node.create_publisher(JointState, "/joint_states", 10)
        self.raw_skeleton_publisher = self.node.create_publisher(
            Marker, RAW_SKELETON_TOPIC, 10
        )
        self.tf_broadcaster = TransformBroadcaster(self.node)
        self.world_frame_id = "world"
        self.root_frame_id = "base_link"
        print(
            "Reference gate: A follows the live retarget from the current "
            "x, y, and yaw. X keeps x, y, and yaw and returns to the stand. "
            "Starts on the default stand."
        )
        self.node.get_logger().info(
            f"Retarget {config_path}; publishing {VECTOR_SIZE}-D "
            f"{self._reference_topic} at {self._reference_rate_hz:.0f} Hz "
            f"from {publish_config_path}"
        )
        if self._hands_enabled:
            topics = self._publish_config["hand_topics"]
            self.node.get_logger().info(
                "Raw hands at "
                f"{self._hand_rate_hz:.0f} Hz: {topics['left_poses']} and "
                f"{topics['right_poses']}"
            )

    def handleAvatar(self, avatar_handle):
        try:
            avatar = MCPAvatar(avatar_handle)
            self._publish_raw_skeleton(avatar)
            if self._hands_enabled:
                self._capture_hands(avatar)
            self.robot.update_robot(avatar)
            self.robot.run_robot_step()
            frame = json.loads(self.robot.get_robot_ros_frame_json()[0])
            positions = frame.get("joint_positions", {})
            joints = [float(positions.get(name, 0.0)) for name in self.joint_names]
            position = list(self.robot.get_robot_root_position())
            position[2] -= LIVE_ROOT_Z_BIAS
            self._latest = _reference_vector(
                position,
                self.robot.get_robot_root_rotataion(),
                joints,
            )
            self._latest_time = self.node.get_clock().now().nanoseconds * 1e-9
        except Exception as exc:
            self.node.get_logger().error(f"RP1 retarget frame failed: {exc}")

    def _publish_raw_skeleton(self, avatar):
        local_poses = {}
        for joint in avatar.get_joints():
            name = joint.get_name()
            if name not in RAW_PARENTS:
                continue
            position = joint.get_local_position()
            if position is None:
                continue
            local_poses[name] = _mocap_pose_to_ros(position, joint.get_local_rotation())
        if "Hips" not in local_poses:
            return

        world_poses = {}

        def world_pose(name):
            if name in world_poses:
                return world_poses[name]
            translation, rotation = local_poses[name]
            parent = RAW_PARENTS[name]
            if parent == "world":
                pose = (translation, _normalize_xyzw(rotation))
            else:
                if parent not in local_poses:
                    return None
                parent_pose = world_pose(parent)
                if parent_pose is None:
                    return None
                parent_translation, parent_rotation = parent_pose
                rotated = _rotate_vector(parent_rotation, translation)
                pose = (
                    tuple(a + b for a, b in zip(parent_translation, rotated)),
                    _normalize_xyzw(_quat_multiply(parent_rotation, rotation)),
                )
            world_poses[name] = pose
            return pose

        stamp = self.node.get_clock().now().to_msg()
        transforms = []
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.world_frame_id
        marker.ns = "raw_skeleton"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.02
        marker.color.r = 0.1
        marker.color.g = 0.9
        marker.color.b = 0.3
        marker.color.a = 1.0
        marker.pose.orientation.w = 1.0

        for name in RAW_PARENTS:
            if name not in local_poses:
                continue
            pose = world_pose(name)
            if pose is None:
                continue
            translation, rotation = pose
            parent = RAW_PARENTS[name]
            transform = TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = (
                self.world_frame_id if parent == "world" else f"raw_{parent}"
            )
            transform.child_frame_id = f"raw_{name}"
            local_translation, local_rotation = local_poses[name]
            transform.transform.translation.x = local_translation[0]
            transform.transform.translation.y = local_translation[1]
            transform.transform.translation.z = local_translation[2]
            transform.transform.rotation.x = local_rotation[0]
            transform.transform.rotation.y = local_rotation[1]
            transform.transform.rotation.z = local_rotation[2]
            transform.transform.rotation.w = local_rotation[3]
            transforms.append(transform)
            if parent == "world":
                continue
            parent_pose = world_poses.get(parent)
            if parent_pose is None:
                continue
            parent_translation = parent_pose[0]
            marker.points.append(Point(
                x=parent_translation[0], y=parent_translation[1], z=parent_translation[2]
            ))
            marker.points.append(Point(
                x=translation[0], y=translation[1], z=translation[2]
            ))

        if transforms:
            self.tf_broadcaster.sendTransform(transforms)
        if marker.points:
            self.raw_skeleton_publisher.publish(marker)

    def dispatch_key(self, key_name):
        if key_name == "a":
            self._request_live()
            return
        if key_name == "x":
            self._request_idle()
            return
        super().dispatch_key(key_name)

    def _now(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    def _live_age(self):
        if self._latest is None:
            return None
        return self._now() - self._latest_time

    def _published_xyyaw(self):
        return (
            float(self._output[0]),
            float(self._output[1]),
            _yaw_from_wxyz(self._output[3:7]),
        )

    def _request_live(self):
        age = self._live_age()
        if self._latest is None or age is None or age > self._source_timeout_s:
            print("A ignored: no fresh retarget frame. Press N and wait for mocap.")
            return
        robot_x, robot_y, robot_yaw = self._published_xyyaw()
        self._planar_hold = (robot_x, robot_y, robot_yaw)
        self._live_origin = (
            float(self._latest[0]),
            float(self._latest[1]),
            _yaw_from_wxyz(self._latest[3:7]),
            robot_x,
            robot_y,
            robot_yaw,
        )
        self._begin_blend("live")
        print("A: x, y, and yaw stay with the robot. Blending to the live retarget.")

    def _request_idle(self):
        self._latch_idle()
        print("X: keeping x, y, and yaw. Blending the rest to the default stand.")

    def _latch_idle(self):
        self._planar_hold = self._published_xyyaw()
        self._live_origin = None
        self._begin_blend("idle")

    def _begin_blend(self, mode):
        self._mode = mode
        self._blend_from = list(self._output)
        self._blend_steps = 0

    def _idle_target(self):
        x, y, yaw = self._planar_hold
        return _stand_at(x, y, yaw)

    def _target(self):
        if self._mode != "live":
            return self._idle_target()
        age = self._live_age()
        if self._latest is None or age is None or age > self._source_timeout_s or self._live_origin is None:
            print("Live retarget went stale. Keeping x, y, and yaw, returning to the stand.")
            self._latch_idle()
            return self._idle_target()
        return _align_planar(self._latest, self._live_origin)

    def _capture_hands(self, avatar):
        local_poses = {}
        for joint in avatar.get_joints():
            name = joint.get_name()
            if name not in POSE_PARENTS:
                continue
            position = joint.get_local_position()
            if position is None:
                continue
            local_poses[name] = _mocap_pose_to_ros(position, joint.get_local_rotation())
        snapshot = {}
        for side, order in HAND_JOINT_ORDER.items():
            world = _compose_world(local_poses, order)
            snapshot[side] = [world[name] for name in order if world.get(name) is not None] or None
        self._hand_snapshot = snapshot

    def _publish_hands(self):
        stamp = self.node.get_clock().now().to_msg()
        for side in ("left", "right"):
            poses = self._hand_snapshot.get(side)
            active = Bool()
            active.data = bool(poses)
            self._hand_active_publishers[side].publish(active)
            message = PoseArray()
            message.header.stamp = stamp
            message.header.frame_id = self._hand_frame_id
            for translation, rotation in poses or ():
                pose = Pose()
                pose.position.x = translation[0]
                pose.position.y = translation[1]
                pose.position.z = translation[2]
                pose.orientation.x = rotation[0]
                pose.orientation.y = rotation[1]
                pose.orientation.z = rotation[2]
                pose.orientation.w = rotation[3]
                message.poses.append(pose)
            self._hand_pose_publishers[side].publish(message)

    async def _hand_loop(self):
        period = 1.0 / self._hand_rate_hz
        while True:
            self._publish_hands()
            await asyncio.sleep(period)

    async def _publish_loop(self):
        period = 1.0 / self._reference_rate_hz
        while True:
            target = self._target()
            if self._blend_from is not None:
                self._blend_steps += 1
                alpha = min(1.0, self._blend_steps / self._transition_steps)
                self._output = _blend(self._blend_from, target, alpha)
                if alpha >= 1.0:
                    self._blend_from = None
            elif self._mode == "live":
                self._output = target
            message = Float32MultiArray()
            message.data = self._output
            self.publisher.publish(message)
            self._publish_rviz(self._output)
            self._published += 1
            if self._published == 1:
                self.node.get_logger().info(
                    f"Sending the default stand on {REFERENCE_TOPIC}. "
                    "Press A for the live retarget."
                )
            await asyncio.sleep(period)

    def _publish_rviz(self, vector):
        """Mirror the published reference so RobotModel can draw it."""
        stamp = self.node.get_clock().now().to_msg()
        joint_state = JointState()
        joint_state.header.stamp = stamp
        joint_state.name = self.joint_names
        joint_state.position = [float(value) for value in vector[7:]]
        self.joint_publisher.publish(joint_state)

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self.world_frame_id
        transform.child_frame_id = self.root_frame_id
        transform.transform.translation.x = float(vector[0])
        transform.transform.translation.y = float(vector[1])
        transform.transform.translation.z = float(vector[2])
        transform.transform.rotation.w = float(vector[3])
        transform.transform.rotation.x = float(vector[4])
        transform.transform.rotation.y = float(vector[5])
        transform.transform.rotation.z = float(vector[6])
        self.tf_broadcaster.sendTransform(transform)

    async def main_async(self):
        publish_task = asyncio.create_task(self._publish_loop())
        hand_task = (
            asyncio.create_task(self._hand_loop()) if self._hands_enabled else None
        )
        try:
            await super().main_async()
        finally:
            publish_task.cancel()
            if hand_task is not None:
                hand_task.cancel()
            try:
                await publish_task
            except asyncio.CancelledError:
                pass
            if hand_task is not None:
                try:
                    await hand_task
                except asyncio.CancelledError:
                    pass

    def main(self):
        asyncio.run(self.main_async())


if __name__ == "__main__":
    MocapRP1Reference().main()
