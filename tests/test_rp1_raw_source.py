"""Raw PN data remains independent of retarget and publication alignment.

The display pipe and ROS messages are tested with in-memory fakes only. These
tests do not import ROS, open a PN connection, or create network sockets.
"""

import asyncio
import copy
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "mocapapi_python" / "mocap_rp1_gmr_reference.py"
SPEC = importlib.util.spec_from_file_location("raw_source_bridge", SCRIPT)
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


def source_frame(sequence=42):
    """Nontrivial orientations expose a mistaken world/local TF conversion."""
    poses = {}
    for index, (name, parent) in enumerate(bridge.BODY_PARENTS.items()):
        if parent == "world":
            position = [1.3, -.4, 1.1]
            rotation = bridge.normalize_quat([.8, .1, -.2, .4])
        else:
            offset = [.017*(index+1), -.011*(index+1), .023*(index+1)]
            local_rotation = bridge.normalize_quat([1., .013*index, -.019*index, .007*index])
            parent_position, parent_rotation = poses[parent]
            position = [a+b for a, b in zip(parent_position, bridge.rotate_vector(parent_rotation, offset))]
            rotation = bridge.normalize_quat(bridge.quat_multiply(parent_rotation, local_rotation))
        poses[name] = [position, rotation]
    return {"seq": sequence, "source_time_s": 15.02,
            "received_monotonic_s": 1234.5, "poses": poses}


def expected_pn(source):
    return {"joint_names": list(bridge.BODY_PARENTS),
            "positions_m": [list(source["poses"][name][0]) for name in bridge.BODY_PARENTS],
            "seq": source["seq"], "source_time_s": source["source_time_s"],
            "received_monotonic_s": source["received_monotonic_s"]}


class RawSourcePipeTests(unittest.TestCase):
    def test_reference_only_payload_stays_compatible(self):
        viewer = bridge.IntegratedViser(["unused-viewer"])
        vector = list(bridge.IDLE_REFERENCE)
        viewer.offer(vector)
        self.assertEqual(viewer.pending, {"qpos": vector})

    def test_full_source_is_sent_in_raw_metres_without_scale_or_alignment(self):
        viewer = bridge.IntegratedViser(["unused-viewer"])
        source = source_frame()
        vector = list(bridge.IDLE_REFERENCE)
        # The published root is deliberately unrelated to the PN root.
        vector[:3] = [-3., 2., .78]
        viewer.offer(vector, source, reference_source_time_s=14.98)
        self.assertEqual(viewer.pending["qpos"], vector)
        self.assertEqual(viewer.pending["pn"], expected_pn(source))
        self.assertEqual(viewer.pending["reference_source_time_s"], 14.98)
        self.assertEqual(viewer.pending["pn"]["positions_m"][0], [1.3, -.4, 1.1])

    def test_missing_source_or_body_joint_keeps_the_robot_reference(self):
        for source in (None, {"poses": {}}, source_frame()):
            if source and source.get("poses"):
                del source["poses"]["LeftFoot"]
            with self.subTest(source=source is not None):
                viewer = bridge.IntegratedViser(["unused-viewer"])
                viewer.offer(bridge.IDLE_REFERENCE, source)
                self.assertEqual(viewer.pending["qpos"], bridge.IDLE_REFERENCE)
                self.assertNotIn("pn", viewer.pending)

    def test_invalid_raw_positions_do_not_replace_the_valid_robot_reference(self):
        for invalid in ([float("nan"), 0., 0.], [0., 0.], None):
            source = source_frame()
            source["poses"]["LeftFoot"][0] = invalid
            with self.subTest(position=invalid):
                viewer = bridge.IntegratedViser(["unused-viewer"])
                viewer.offer(bridge.IDLE_REFERENCE, source)
                self.assertEqual(viewer.pending["qpos"], bridge.IDLE_REFERENCE)
                self.assertNotIn("pn", viewer.pending)

    def test_pending_payload_is_a_snapshot_and_remains_latest_only(self):
        viewer = bridge.IntegratedViser(["unused-viewer"])
        source = source_frame()
        vector = list(bridge.IDLE_REFERENCE)
        viewer.offer(vector, source)
        snapshot = copy.deepcopy(viewer.pending)
        vector[0] = 1000.
        source["poses"]["Hips"][0][0] = 1000.
        source["seq"] = 1000
        self.assertEqual(viewer.pending, snapshot)
        newer = source_frame(sequence=43)
        viewer.offer(bridge.IDLE_REFERENCE, newer)
        self.assertEqual(viewer.pending["pn"], expected_pn(newer))
        self.assertEqual(viewer.status["offered"], 2)
        self.assertEqual(viewer.status["replaced"], 1)


class RawSourceSerializedPipeTests(unittest.IsolatedAsyncioTestCase):
    async def test_serialized_pipe_contains_the_raw_snapshot_and_exact_reference(self):
        writes = []

        class Pipe:
            def write(self, data):
                writes.append(json.loads(data))

            async def drain(self):
                pass

        process = SimpleNamespace(returncode=None, stdin=Pipe())
        viewer = bridge.IntegratedViser(["unused-viewer"])
        source = source_frame()
        with mock.patch.object(bridge.asyncio, "create_subprocess_exec",
                               new=mock.AsyncMock(return_value=process)), \
                mock.patch.object(bridge, "terminate_subprocess", new=mock.AsyncMock()):
            await viewer.start()
            try:
                viewer.offer(bridge.IDLE_REFERENCE, source, reference_source_time_s=15.)
                for _ in range(30):
                    if viewer.status["sent"]:
                        break
                    await asyncio.sleep(0)
                self.assertEqual(len(writes), 1)
                self.assertEqual(writes[0]["qpos"], bridge.IDLE_REFERENCE)
                self.assertEqual(writes[0]["pn"], expected_pn(source))
                self.assertEqual(writes[0]["reference_source_time_s"], 15.)
            finally:
                await viewer.close()


def xyz():
    return SimpleNamespace(x=0., y=0., z=0.)


def quaternion():
    return SimpleNamespace(w=0., x=0., y=0., z=0.)


class Header:
    def __init__(self):
        self.stamp = None
        self.frame_id = ""


class Transform:
    def __init__(self):
        self.header = Header()
        self.child_frame_id = ""
        self.transform = SimpleNamespace(translation=xyz(), rotation=quaternion())


class Marker:
    LINE_LIST, ADD = 5, 0

    def __init__(self):
        self.header = Header()
        self.ns = ""
        self.id = self.type = self.action = 0
        self.scale = xyz()
        self.color = SimpleNamespace(r=0., g=0., b=0., a=0.)
        self.pose = SimpleNamespace(position=xyz(), orientation=quaternion())
        self.points = []


class RecordingPublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class RecordingTF:
    def __init__(self):
        self.transforms = []

    def sendTransform(self, value):
        self.transforms.extend(value if isinstance(value, (list, tuple)) else [value])


def fake_ros_publisher():
    publisher = bridge.ROSPublisher.__new__(bridge.ROSPublisher)
    publisher.node = SimpleNamespace(get_clock=lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(to_msg=lambda: "synthetic-stamp")))
    publisher.Reference = lambda: SimpleNamespace(data=[])
    publisher.JointState = lambda: SimpleNamespace(header=Header(), name=[], position=[])
    publisher.TransformStamped, publisher.Marker = Transform, Marker
    publisher.Point = lambda **kwargs: SimpleNamespace(**kwargs)
    publisher.joint_names = [f"joint_{index}" for index in range(28)]
    publisher.publisher = RecordingPublisher()
    publisher.joint_publisher = RecordingPublisher()
    publisher.raw_publisher = RecordingPublisher()
    publisher.tf = RecordingTF()
    publisher.hands, publisher.hand_publishers = {}, {}
    publisher.executor = SimpleNamespace(spin_once=lambda **kwargs: None)
    return publisher


class RawSourceROSTests(unittest.TestCase):
    def assertVectorClose(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b, places=9)

    def assertQuaternionClose(self, actual, expected):
        # q and -q describe the same orientation.
        sign = 1. if sum(a*b for a, b in zip(actual, expected)) >= 0. else -1.
        self.assertVectorClose(actual, [sign*v for v in expected])

    def test_raw_source_is_enabled_by_default_with_explicit_opt_out(self):
        self.assertTrue(bridge.parse_args([]).source_viz)
        self.assertTrue(bridge.parse_args(["--source-viz"]).source_viz)
        self.assertFalse(bridge.parse_args(["--no-source-viz"]).source_viz)
        self.assertIs(inspect.signature(bridge.ROSPublisher).parameters["source_viz"].default, True)

    def test_marker_uses_original_world_positions_and_legacy_style(self):
        publisher = fake_ros_publisher()
        source = source_frame()
        publisher.publish(list(bridge.IDLE_REFERENCE), source)
        self.assertEqual(len(publisher.raw_publisher.messages), 1)
        marker = publisher.raw_publisher.messages[0]
        self.assertEqual(marker.header.frame_id, "world")
        self.assertEqual(marker.ns, "raw_skeleton")
        self.assertEqual((marker.id, marker.type, marker.action), (0, Marker.LINE_LIST, Marker.ADD))
        self.assertEqual((marker.scale.x, marker.color.r, marker.color.g,
                          marker.color.b, marker.color.a), (.02, .1, .9, .3, 1.))
        expected = []
        for name, parent in bridge.BODY_PARENTS.items():
            if parent != "world":
                expected.extend([source["poses"][parent][0], source["poses"][name][0]])
        actual = [[point.x, point.y, point.z] for point in marker.points]
        self.assertEqual(len(actual), 40)
        for point, raw_point in zip(actual, expected):
            self.assertVectorClose(point, raw_point)
        self.assertEqual(publisher.publisher.messages[0].data, bridge.IDLE_REFERENCE)

    def test_raw_tf_parent_chain_reconstructs_unscaled_original_world_poses(self):
        publisher = fake_ros_publisher()
        source = source_frame()
        publisher.publish(list(bridge.IDLE_REFERENCE), source)
        raw = {transform.child_frame_id: transform for transform in publisher.tf.transforms
               if transform.child_frame_id.startswith("raw_")}
        self.assertEqual(set(raw), {"raw_"+name for name in bridge.BODY_PARENTS})
        reconstructed = {}
        for name, parent in bridge.BODY_PARENTS.items():
            transform = raw["raw_"+name]
            self.assertEqual(transform.header.frame_id, "world" if parent == "world" else "raw_"+parent)
            self.assertEqual(transform.header.stamp, "synthetic-stamp")
            translation, rotation = transform.transform.translation, transform.transform.rotation
            position, orientation = [translation.x, translation.y, translation.z], \
                                    [rotation.w, rotation.x, rotation.y, rotation.z]
            if parent != "world":
                pp, pq = reconstructed[parent]
                position = [a+b for a, b in zip(pp, bridge.rotate_vector(pq, position))]
                orientation = bridge.quat_multiply(pq, orientation)
            reconstructed[name] = [position, orientation]
            self.assertVectorClose(position, source["poses"][name][0])
            self.assertQuaternionClose(orientation, source["poses"][name][1])
        robots = [transform for transform in publisher.tf.transforms
                  if transform.child_frame_id == "base_link"]
        self.assertEqual(len(robots), 1)
        self.assertEqual(robots[0].header.frame_id, "world")

    def test_raw_opt_out_or_missing_frame_does_not_change_robot_publication(self):
        for source, enabled in ((source_frame(), False), (None, True)):
            publisher = fake_ros_publisher()
            raw_publisher = publisher.raw_publisher
            if not enabled:
                publisher.raw_publisher = None
            publisher.publish(list(bridge.IDLE_REFERENCE), source)
            self.assertEqual(publisher.publisher.messages[0].data, bridge.IDLE_REFERENCE)
            self.assertEqual(raw_publisher.messages, [])
            self.assertFalse(any(transform.child_frame_id.startswith("raw_")
                                 for transform in publisher.tf.transforms))


if __name__ == "__main__":
    unittest.main()
