"""Finger-only hand topics, without importing ROS or opening PN/IK devices."""

import copy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "mocapapi_python" / "mocap_rp1_gmr_reference.py"
SPEC = importlib.util.spec_from_file_location("hand_topics_bridge", SCRIPT)
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


def finger_order(side):
    side = side.title()
    return ([f"{side}HandThumb{i}" for i in (1, 2, 3)] +
            [name for finger in ("Index", "Middle", "Ring", "Pinky")
             for name in (f"{side}InHand{finger}",
                          *(f"{side}Hand{finger}{i}" for i in (1, 2, 3)))])


def xyz():
    return SimpleNamespace(x=0., y=0., z=0.)


def quaternion():
    return SimpleNamespace(w=0., x=0., y=0., z=0.)


class Header:
    def __init__(self):
        self.stamp, self.frame_id = None, ""


class Pose:
    def __init__(self):
        self.position, self.orientation = xyz(), quaternion()


class PoseArray:
    def __init__(self):
        self.header, self.poses = Header(), []


class TransformStamped:
    def __init__(self):
        self.header, self.child_frame_id = Header(), ""
        self.transform = SimpleNamespace(translation=xyz(), rotation=quaternion())


class JointState:
    def __init__(self):
        self.header, self.name, self.position = Header(), [], []


class Bool:
    def __init__(self):
        self.data = False


class Reference:
    def __init__(self):
        self.data = []


class Publisher:
    def __init__(self, message_type, topic, qos):
        self.message_type, self.topic, self.qos = message_type, topic, qos
        self.messages = []

    def publish(self, message):
        self.messages.append(copy.deepcopy(message))


class Node:
    def __init__(self, name):
        self.name, self.context, self.publishers = name, object(), {}

    def create_publisher(self, message_type, topic, qos):
        publisher = Publisher(message_type, topic, qos)
        self.publishers[topic] = publisher
        return publisher

    def get_clock(self):
        return SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: "hand-test-stamp"))


class Executor:
    def __init__(self, **kwargs):
        self.nodes, self.spins = [], 0

    def add_node(self, node):
        self.nodes.append(node)

    def spin_once(self, **kwargs):
        self.spins += 1


class Broadcaster:
    def __init__(self, node):
        self.transforms = []

    def sendTransform(self, values):
        self.transforms.extend(values)


def ros_modules():
    def module(name, **attributes):
        value = ModuleType(name)
        value.__dict__.update(attributes)
        return value

    return {
        "rclpy": module("rclpy", init=mock.Mock()),
        "rclpy.node": module("rclpy.node", Node=Node),
        "rclpy.executors": module("rclpy.executors", SingleThreadedExecutor=Executor),
        "rclpy.qos": module("rclpy.qos", QoSProfile=lambda **kwargs: SimpleNamespace(**kwargs),
                            HistoryPolicy=SimpleNamespace(KEEP_LAST="keep_last"),
                            ReliabilityPolicy=SimpleNamespace(RELIABLE="reliable", BEST_EFFORT="best_effort"),
                            DurabilityPolicy=SimpleNamespace(VOLATILE="volatile")),
        "sensor_msgs": module("sensor_msgs"),
        "sensor_msgs.msg": module("sensor_msgs.msg", JointState=JointState),
        "std_msgs": module("std_msgs"),
        "std_msgs.msg": module("std_msgs.msg", Float32MultiArray=Reference, Bool=Bool),
        "geometry_msgs": module("geometry_msgs"),
        "geometry_msgs.msg": module("geometry_msgs.msg", TransformStamped=TransformStamped,
                                    PoseArray=PoseArray, Pose=Pose,
                                    Point=lambda **kwargs: SimpleNamespace(**kwargs)),
        "visualization_msgs": module("visualization_msgs"),
        "visualization_msgs.msg": module("visualization_msgs.msg", Marker=type("Marker", (), {})),
        "tf2_ros": module("tf2_ros", TransformBroadcaster=Broadcaster),
    }


def make_publisher(enabled=True):
    config = bridge.load_publish_config(SCRIPT.parent / "rp1" / "publish.yaml")
    config["hands"]["enabled"] = enabled
    with mock.patch.dict(sys.modules, ros_modules()):
        return bridge.ROSPublisher(config, [f"joint_{i}" for i in range(28)], source_viz=False)


def source_frame():
    poses = {}
    for side_index, side in enumerate(("right", "left")):
        # Dict insertion order must have no effect on the published order.
        for index, name in reversed(list(enumerate(finger_order(side)))):
            poses[name] = [[1.2+side_index+.013*index, -.23-.017*index, .84+.019*index],
                           [.5, -.5, .5, -.5]]
        # These distinctive wrist positions must never appear in the message.
        poses[f"{side.title()}Hand"] = [[-99., -98., -97.], [1., 0., 0., 0.]]
    return {"seq": 81, "source_time_s": 15.02, "received_monotonic_s": 24., "poses": poses}


class HandTopicTests(unittest.TestCase):
    def message(self, publisher, side, channel):
        return publisher.node.publishers[f"/mocap/hand/{side}/{channel}"].messages[-1]

    def assertInactive(self, publisher, side):
        self.assertFalse(self.message(publisher, side, "active").data)
        self.assertEqual(self.message(publisher, side, "poses").poses, [])

    def test_hand_order_has_nineteen_finger_points_and_no_wrist(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                self.assertEqual(bridge.HAND_ORDERS[side], finger_order(side))
                self.assertEqual(len(bridge.HAND_ORDERS[side]), 19)
                self.assertNotIn(f"{side.title()}Hand", bridge.HAND_ORDERS[side])

    def test_existing_topics_types_fifty_hertz_and_qos_are_preserved(self):
        config = bridge.load_publish_config(SCRIPT.parent / "rp1" / "publish.yaml")
        self.assertEqual(config["hands"]["rate_hz"], 50)
        self.assertEqual(config["reference"]["rate_hz"], 50)
        publisher = make_publisher()
        self.assertEqual(set(publisher.hand_publishers), {"left", "right"})
        for side in ("left", "right"):
            pose_pub, active_pub = publisher.hand_publishers[side]
            self.assertEqual(pose_pub.topic, f"/mocap/hand/{side}/poses")
            self.assertEqual(active_pub.topic, f"/mocap/hand/{side}/active")
            self.assertIs(pose_pub.message_type, PoseArray)
            self.assertIs(active_pub.message_type, Bool)
            for channel in (pose_pub, active_pub):
                self.assertEqual(channel.qos.depth, 1)
                self.assertEqual(channel.qos.reliability, "best_effort")
                self.assertEqual(channel.qos.history, "keep_last")

    def test_complete_hand_has_nineteen_ordered_world_poses_without_scale_or_alignment(self):
        publisher = make_publisher()
        source = source_frame()
        original = copy.deepcopy(source)
        reference = list(bridge.IDLE_REFERENCE)
        reference[:3] = [-8., 5., .58]
        reference[7:] = [index*.017 for index in range(28)]
        publisher.publish(reference, source)
        self.assertEqual(source, original)
        self.assertEqual(publisher.publisher.messages[-1].data, reference)
        self.assertEqual(publisher.joint_publisher.messages[-1].position, reference[7:])
        self.assertEqual(publisher.joint_publisher.messages[-1].name, publisher.joint_names)
        for side in ("left", "right"):
            message = self.message(publisher, side, "poses")
            self.assertEqual(message.header.frame_id, "world")
            self.assertEqual(message.header.stamp, "hand-test-stamp")
            self.assertEqual(message.header.stamp, publisher.joint_publisher.messages[-1].header.stamp)
            self.assertTrue(self.message(publisher, side, "active").data)
            self.assertEqual(len(message.poses), 19)
            for name, pose in zip(finger_order(side), message.poses):
                position, rotation = source["poses"][name]
                self.assertEqual([pose.position.x, pose.position.y, pose.position.z], position)
                self.assertEqual([pose.orientation.w, pose.orientation.x,
                                  pose.orientation.y, pose.orientation.z], rotation)
                self.assertNotEqual([pose.position.x, pose.position.y, pose.position.z], [-99., -98., -97.])
        self.assertEqual(publisher.executor.spins, 1)

    def test_only_wrist_data_is_empty_and_inactive(self):
        publisher = make_publisher()
        source = source_frame()
        source["poses"] = {name: value for name, value in source["poses"].items()
                           if name in ("LeftHand", "RightHand")}
        publisher.publish(list(bridge.IDLE_REFERENCE), source)
        for side in ("left", "right"):
            self.assertInactive(publisher, side)

    def test_partial_fingers_remain_active_and_keep_existing_omission_order(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                publisher = make_publisher()
                source = source_frame()
                missing = f"{side.title()}HandThumb2"
                del source["poses"][missing]
                publisher.publish(list(bridge.IDLE_REFERENCE), source)
                message = self.message(publisher, side, "poses")
                self.assertEqual(len(message.poses), 18)
                self.assertTrue(self.message(publisher, side, "active").data)
                expected = [source["poses"][name][0] for name in finger_order(side) if name != missing]
                self.assertEqual([[p.position.x, p.position.y, p.position.z] for p in message.poses], expected)
                other = "right" if side == "left" else "left"
                self.assertEqual(len(self.message(publisher, other, "poses").poses), 19)

    def test_one_index_joint_can_be_published_without_thumb_or_wrist(self):
        publisher = make_publisher()
        source = source_frame()
        name = "LeftHandIndex2"
        source["poses"] = {name: source["poses"][name]}
        publisher.publish(list(bridge.IDLE_REFERENCE), source)
        self.assertTrue(self.message(publisher, "left", "active").data)
        self.assertEqual(len(self.message(publisher, "left", "poses").poses), 1)
        self.assertInactive(publisher, "right")

    def test_none_and_empty_source_clear_previous_finger_messages(self):
        for missing in (None, {"poses": {}}):
            with self.subTest(missing=missing):
                publisher = make_publisher()
                publisher.publish(list(bridge.IDLE_REFERENCE), source_frame())
                publisher.publish(list(bridge.IDLE_REFERENCE), missing)
                for side in ("left", "right"):
                    self.assertInactive(publisher, side)
                    self.assertEqual(len(publisher.hand_publishers[side][0].messages), 2)

    def test_disabled_hands_do_not_create_hand_topics_or_change_reference(self):
        publisher = make_publisher(enabled=False)
        publisher.publish(list(bridge.IDLE_REFERENCE), source_frame())
        self.assertEqual(publisher.hand_publishers, {})
        self.assertFalse(any(topic.startswith("/mocap/hand/") for topic in publisher.node.publishers))
        self.assertEqual(publisher.publisher.messages[-1].data, bridge.IDLE_REFERENCE)

    def test_wrist_parent_is_retained_for_sdk_fk_body_ik_and_raw_display(self):
        for side in ("Left", "Right"):
            with self.subTest(side=side):
                wrist = f"{side}Hand"
                self.assertEqual(bridge.BODY_PARENTS[wrist], f"{side}ForeArm")
                self.assertEqual(bridge.ALL_PARENTS[wrist], f"{side}ForeArm")
                self.assertEqual(bridge.ALL_PARENTS[f"{side}HandThumb1"], wrist)
                self.assertEqual(bridge.ALL_PARENTS[f"{side}InHandIndex"], wrist)
                locals_ = {name: [[0., 0., 0.], [1., 0., 0., 0.]] for name in bridge.ALL_PARENTS}
                locals_["Hips"][0] = [100., 200., 300.]
                locals_[wrist][0] = [10., 20., 30.]
                locals_[f"{side}HandThumb1"][0] = [1., 2., 3.]
                world = bridge.world_poses_from_sdk_locals(locals_, bridge.ALL_PARENTS)
                self.assertIn(wrist, world)
                for actual, expected in zip(world[wrist][0], [3.3, 1.1, 2.2]):
                    self.assertAlmostEqual(actual, expected)
                for actual, expected in zip(world[f"{side}HandThumb1"][0], [3.33, 1.11, 2.22]):
                    self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
