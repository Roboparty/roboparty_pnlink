"""Reference publication preserves retarget IK after alignment and transition.

These tests use synthetic poses and a fake in-process source/IK worker. They
never open PN, load a robot model, or publish a ROS message.
"""

import asyncio
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "mocapapi_python" / "mocap_rp1_gmr_reference.py"
SPEC = importlib.util.spec_from_file_location("passthrough_bridge", SCRIPT)
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


def pose(x=18., y=-22., z=.83, heading=.4, roll=.2, joint_delta=.4):
    q = list(bridge.IDLE_REFERENCE)
    q[:3] = [x, y, z]
    # Rz(heading) Rx(roll), in wxyz order.
    c, s = math.cos(heading/2.), math.sin(heading/2.)
    cr, sr = math.cos(roll/2.), math.sin(roll/2.)
    q[3:7] = [c*cr, c*sr, s*sr, s*cr]
    q[7:] = [value+joint_delta*(i+1)/28. for i, value in enumerate(q[7:])]
    return q


def result(qpos, seq=1, now=10.):
    return {"seq": seq, "qpos": list(qpos), "received_monotonic_s": now,
            "source_time_s": seq*.02, "worker_roundtrip_ms": .1,
            "stats": {"solver_ms": .05}}


def aligned_pose(qpos, origin):
    sx, sy, syaw, rx, ry, ryaw = origin
    heading = ryaw-syaw
    c, s = math.cos(heading), math.sin(heading)
    dx, dy = qpos[0]-sx, qpos[1]-sy
    expected = list(qpos)
    expected[:2] = [rx+c*dx-s*dy, ry+s*dx+c*dy]
    hcos, hsin = math.cos(heading/2.), math.sin(heading/2.)
    w, x, y, z = qpos[3:7]
    expected[3:7] = [hcos*w-hsin*z, hcos*x-hsin*y,
                     hcos*y+hsin*x, hcos*z+hsin*w]
    return expected


class ReferencePassthroughTests(unittest.TestCase):
    def assertPoseEqual(self, actual, expected):
        self.assertEqual(len(actual), 35)
        for actual_value, expected_value in zip(actual, expected):
            self.assertAlmostEqual(actual_value, expected_value, places=12)

    def test_steady_output_preserves_root_height_and_all_28_ik_angles(self):
        with mock.patch.object(bridge, "OutputLimiter", side_effect=AssertionError("no final limiter")):
            gate = bridge.ReferenceGate(timeout_s=10.)
            gate.accept(result(pose()))
            self.assertTrue(gate.activate(10.))
            gate.tick(12.)
            # A deliberately large 20 ms change distinguishes passthrough
            # from either the old velocity limiter or acceleration limiter.
            target = pose(x=19.2, y=-21.3, z=.72, heading=.9, roll=-.3, joint_delta=1.2)
            gate.accept(result(target, seq=2, now=12.02))
            output = gate.tick(12.02)
        self.assertPoseEqual(output, aligned_pose(target, gate.origin))
        self.assertEqual(output[2], target[2])
        self.assertEqual(output[7:], target[7:])
        self.assertIsNone(gate.blend_from)

    def test_two_second_transition_starts_at_previous_output_and_finishes_at_ik(self):
        gate = bridge.ReferenceGate(timeout_s=10., blend_s=2.)
        start = list(gate.output)
        target = pose()
        gate.accept(result(target))
        self.assertTrue(gate.activate(10.))
        self.assertPoseEqual(gate.tick(10.), start)
        aligned = aligned_pose(target, gate.origin)
        halfway = gate.tick(11.)
        self.assertAlmostEqual(halfway[2], (start[2]+aligned[2])/2., places=12)
        for actual, first, last in zip(halfway[7:], start[7:], aligned[7:]):
            self.assertAlmostEqual(actual, (first+last)/2., places=12)
        # Initial yaw is removed, leaving Rx(.2); at halfway this is Rx(.1).
        self.assertPoseEqual(halfway[:3]+[math.cos(.05), math.sin(.05), 0., 0.]+halfway[7:], halfway)
        self.assertPoseEqual(gate.tick(12.), aligned)
        self.assertIsNone(gate.blend_from)

    def test_idle_and_reactivation_use_current_reference_as_new_alignment_origin(self):
        gate = bridge.ReferenceGate(timeout_s=10., blend_s=2.)
        gate.accept(result(pose()))
        self.assertTrue(gate.activate(10.))
        gate.tick(12.)
        target = pose(x=19., y=-21., heading=.8)
        gate.accept(result(target, seq=2, now=12.02))
        moving = gate.tick(12.02)
        gate.idle(12.02)
        self.assertPoseEqual(gate.tick(12.02), moving)
        resting = gate.tick(14.02)
        self.assertEqual(resting[:2], moving[:2])
        self.assertEqual(resting[7:], bridge.IDLE_REFERENCE[7:])
        new_source = pose(x=-7., y=35., heading=-1.1, z=.91)
        gate.accept(result(new_source, seq=3, now=14.03))
        self.assertTrue(gate.activate(14.03))
        self.assertPoseEqual(gate.tick(14.03), resting)
        self.assertEqual(gate.origin[:2], tuple(new_source[:2]))
        self.assertEqual(gate.origin[3:5], tuple(resting[:2]))
        self.assertPoseEqual(gate.tick(16.03), aligned_pose(new_source, gate.origin))

    def test_timeout_starts_idle_transition_and_new_results_do_not_auto_activate(self):
        gate = bridge.ReferenceGate(timeout_s=.25, blend_s=2.)
        gate.accept(result(pose()))
        gate.activate(10.)
        previous = gate.tick(10.1)
        self.assertPoseEqual(gate.tick(10.3), previous)
        self.assertEqual(gate.mode, "idle")
        self.assertEqual(gate.stale_count, 1)
        gate.accept(result(pose(x=30.), seq=2, now=10.31))
        gate.tick(10.32)
        self.assertEqual(gate.mode, "idle")
        self.assertEqual(gate.stale_count, 1)
        self.assertTrue(gate.activate(10.33))


class PublicationPassthroughTests(unittest.IsolatedAsyncioTestCase):
    async def test_publisher_never_constructs_or_calls_final_constraint_processing(self):
        joint_names = json.loads(bridge.DEFAULT_JOINT_CONFIG.read_text())["urdfJointNames"]

        class FakeWorker:
            def __init__(self, command, accept, timeout, startup_timeout, on_ready):
                self.accept, self.on_ready = accept, on_ready
                self.ready, self.process = asyncio.Event(), None
                self.metadata = {"joint_names": joint_names, "model": "test fixture"}
                self.startup_ms = [.1]
                self.solved = self.errors = self.restarts = 0
                self.last_error = None
                self.mailbox = bridge.LatestFrameMailbox()

            async def run(self):
                self.on_ready(self.metadata)
                self.ready.set()
                while True:
                    frame = await self.mailbox.take()
                    i = frame["seq"]
                    self.solved += 1
                    sample = result(pose(x=18.+i*.08, y=-22.+i*.03,
                                         z=.83-i*.002, heading=.4+i*.03,
                                         joint_delta=.4+i*.2), seq=i,
                                    now=frame["received_monotonic_s"])
                    sample["source_time_s"] = frame["source_time_s"]
                    self.accept(sample)

        class FakeReplay:
            times, speed, skipped = [0., .14], 1., 0

            def __init__(self, path, speed):
                pass

            async def run(self, submit, stop):
                for i in range(8):
                    submit({"seq": i, "source_time_s": i*.02,
                            "received_monotonic_s": time.monotonic(), "poses": {}})
                    await asyncio.sleep(.02)

        original_gate = bridge.ReferenceGate

        def short_transition(timeout_s, blend_s, *args, **kwargs):
            return original_gate(timeout_s, .02)

        with tempfile.TemporaryDirectory() as directory, ExitStack() as patches:
            directory = Path(directory)
            args = bridge.parse_args(["--source", "replay", "--npz", "unused.npz", "--dry-run",
                                      "--autostart", "--replay-tail-s", "0", "--max-duration-s", ".18",
                                      "--output", str(directory / "reference.jsonl"),
                                      "--metrics", str(directory / "metrics.json")])
            patches.enter_context(mock.patch.object(bridge, "SolverWorker", FakeWorker))
            patches.enter_context(mock.patch.object(bridge, "ReplaySource", FakeReplay))
            patches.enter_context(mock.patch.object(bridge, "ReferenceGate", short_transition))
            for forbidden in ("ReferenceSafetyWorker", "OutputLimiter", "recover_publication_timing",
                              "check_published_motion"):
                patches.enter_context(mock.patch.object(bridge, forbidden,
                    side_effect=AssertionError(f"{forbidden} must never run in the reference publisher")))
            patches.enter_context(mock.patch.object(bridge, "start_cleanup_watchdog", return_value=None))
            patches.enter_context(mock.patch.object(bridge.sys.stdin, "isatty", return_value=False))
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                metrics = await bridge.run_bridge(args)
            rows = [json.loads(line) for line in (directory / "reference.jsonl").read_text().splitlines()]
            saved = json.loads((directory / "metrics.json").read_text())

        self.assertGreaterEqual(len(rows), 5)
        stable = [row for row in rows if row["mode"] == "live" and not row["blend_active"]]
        self.assertGreaterEqual(len(stable), 2)
        for row in rows:
            self.assertEqual(row["reference_pipeline"], "alignment_transition_only")
            self.assertFalse(row["final_constraints_enabled"])
            self.assertEqual(row["qpos"], row["desired_qpos"])
        for row in stable:
            expected = aligned_pose(row["solver_qpos"], row["alignment_se2"])
            for actual, value in zip(row["qpos"], expected):
                self.assertAlmostEqual(actual, value, places=12)
            self.assertEqual(row["qpos"][2], row["solver_qpos"][2])
            self.assertEqual(row["qpos"][7:], row["solver_qpos"][7:])
        self.assertEqual(metrics["reference_pipeline"], "alignment_transition_only")
        self.assertFalse(metrics["final_constraints_enabled"])
        self.assertIsNone(metrics["reference_fault"])
        self.assertEqual(metrics["process_cleanup"]["status"], "complete")
        self.assertEqual(saved["published"], len(rows))
        self.assertFalse(any(key.startswith("safety_") for key in metrics))


if __name__ == "__main__":
    unittest.main()
