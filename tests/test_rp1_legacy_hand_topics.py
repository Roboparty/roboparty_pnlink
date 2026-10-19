"""Legacy hand topics contain fingers only, tested without importing ROS/SDK."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest


def load_hand_io():
    directory = Path(__file__).resolve().parent
    candidates = (
        directory / "mocap_rp1_reference.py",
        directory.parent / "mocapapi_python" / "mocap_rp1_reference.py",
    )
    source = next(path for path in candidates if path.is_file())
    tree = ast.parse(source.read_text(), filename=str(source))
    assignments = {"RAW_PARENTS", "HAND_JOINT_ORDER", "POSE_PARENTS"}
    functions = {
        "_hand_joint_order", "_with_hand_parents", "_mocap_pose_to_ros",
        "_quat_multiply", "_rotate_vector", "_normalize_xyzw", "_compose_world",
    }
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in assignments
            for target in node.targets
        ):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "MocapRP1Reference":
            methods = [method for method in node.body
                       if isinstance(method, ast.FunctionDef)
                       and method.name in ("_capture_hands", "_publish_hands")]
            selected.append(ast.ClassDef(name="HandIO", bases=[], keywords=[],
                                         body=methods, decorator_list=[]))
    namespace = {"math": math, "Bool": Bool, "PoseArray": PoseArray, "Pose": Pose}
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace


class Bool:
    def __init__(self):
        self.data = False


class Pose:
    def __init__(self):
        self.position = SimpleNamespace(x=0., y=0., z=0.)
        self.orientation = SimpleNamespace(x=0., y=0., z=0., w=0.)


class PoseArray:
    def __init__(self):
        self.header = SimpleNamespace(stamp=None, frame_id="")
        self.poses = []


class Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, value):
        self.messages.append(value)


class Joint:
    def __init__(self, name, position, rotation=(1., 0., 0., 0.)):
        self.name, self.position, self.rotation = name, position, rotation

    def get_name(self):
        return self.name

    def get_local_position(self):
        return self.position

    def get_local_rotation(self):
        return self.rotation


HAND_IO = load_hand_io()


def make_io():
    io = HAND_IO["HandIO"]()
    io.node = SimpleNamespace(get_clock=lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(to_msg=lambda: "test-stamp")))
    io._hand_frame_id = "world"
    io._hand_snapshot = {"left": None, "right": None}
    io._hand_pose_publishers = {side: Publisher() for side in ("left", "right")}
    io._hand_active_publishers = {side: Publisher() for side in ("left", "right")}
    return io


def make_avatar(exclude=(), fingers=True, wrist_rotation=(1., 0., 0., 0.)):
    joints = []
    for name in HAND_IO["POSE_PARENTS"]:
        if name in exclude or (not fingers and name not in HAND_IO["RAW_PARENTS"]):
            continue
        position = (0., 0., 0.)
        rotation = (1., 0., 0., 0.)
        if name == "Hips":
            position = (100., 200., 300.)
        elif name in ("LeftHand", "RightHand"):
            position = (10., 20., 30.)
            rotation = wrist_rotation
        elif name not in HAND_IO["RAW_PARENTS"]:
            position = (1., 2., 3.)
        joints.append(Joint(name, position, rotation))
    return SimpleNamespace(get_joints=lambda: joints)


class LegacyFingerTopicTests(unittest.TestCase):
    def test_fixed_order_excludes_wrist_and_retains_fk_parents(self):
        for side, prefix in (("left", "Left"), ("right", "Right")):
            order = HAND_IO["HAND_JOINT_ORDER"][side]
            expected = [f"{prefix}HandThumb{index}" for index in (1, 2, 3)]
            for finger in ("Index", "Middle", "Ring", "Pinky"):
                expected.extend([f"{prefix}InHand{finger}", *(
                    f"{prefix}Hand{finger}{index}" for index in (1, 2, 3))])
            self.assertEqual(order, tuple(expected))
            self.assertEqual(len(order), 19)
            self.assertNotIn(f"{prefix}Hand", order)
            self.assertIn(f"{prefix}Hand", HAND_IO["RAW_PARENTS"])
            self.assertIn(f"{prefix}Hand", HAND_IO["POSE_PARENTS"])
            self.assertEqual(HAND_IO["POSE_PARENTS"][order[0]], f"{prefix}Hand")

    def test_complete_capture_publishes_nineteen_world_poses_and_active(self):
        io = make_io()
        io._capture_hands(make_avatar())
        io._publish_hands()
        for side in ("left", "right"):
            message = io._hand_pose_publishers[side].messages[-1]
            self.assertEqual(len(message.poses), 19)
            self.assertTrue(io._hand_active_publishers[side].messages[-1].data)
            self.assertEqual(message.header.stamp, "test-stamp")
            self.assertEqual(message.header.frame_id, "world")
            pose = message.poses[0]
            for actual, expected in zip((pose.position.x, pose.position.y, pose.position.z),
                                        (3.33, 1.11, 2.22)):
                self.assertAlmostEqual(actual, expected)
            self.assertEqual((pose.orientation.x, pose.orientation.y,
                              pose.orientation.z, pose.orientation.w), (0., 0., 0., 1.))

    def test_missing_thumb_root_keeps_other_fingers_and_other_hand(self):
        io = make_io()
        io._capture_hands(make_avatar(exclude=("LeftHandThumb1",)))
        io._publish_hands()
        self.assertEqual(len(io._hand_pose_publishers["left"].messages[-1].poses), 16)
        self.assertEqual(len(io._hand_pose_publishers["right"].messages[-1].poses), 19)
        self.assertTrue(io._hand_active_publishers["left"].messages[-1].data)

    def test_wrist_only_and_none_snapshot_publish_empty_inactive(self):
        io = make_io()
        io._capture_hands(make_avatar(fingers=False))
        self.assertEqual(io._hand_snapshot, {"left": None, "right": None})
        for snapshot in (io._hand_snapshot, {"left": None, "right": None}, {}):
            with self.subTest(snapshot=snapshot):
                io._hand_snapshot = snapshot
                io._publish_hands()
                for side in ("left", "right"):
                    self.assertEqual(io._hand_pose_publishers[side].messages[-1].poses, [])
                    self.assertFalse(io._hand_active_publishers[side].messages[-1].data)

    def test_fingers_still_depend_on_wrist_fk_and_rotation(self):
        io = make_io()
        half = math.sqrt(.5)
        io._capture_hands(make_avatar(wrist_rotation=(half, 0., half, 0.)))
        io._publish_hands()
        first = io._hand_pose_publishers["left"].messages[-1].poses[0]
        for actual, expected in zip((first.position.x, first.position.y, first.position.z),
                                    (3.29, 1.13, 2.22)):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip((first.orientation.x, first.orientation.y,
                                    first.orientation.z, first.orientation.w),
                                   (0., 0., half, half)):
            self.assertAlmostEqual(actual, expected)
        io._capture_hands(make_avatar(exclude=("LeftHand",)))
        io._publish_hands()
        self.assertEqual(io._hand_pose_publishers["left"].messages[-1].poses, [])
        self.assertFalse(io._hand_active_publishers["left"].messages[-1].data)
        self.assertEqual(len(io._hand_pose_publishers["right"].messages[-1].poses), 19)


if __name__ == "__main__":
    unittest.main()
