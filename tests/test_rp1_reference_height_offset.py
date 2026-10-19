"""Fixed publication root-Z offset, with no PN, ROS or robot side effects.

The offset belongs to the live reference target, before the existing SE(2)
alignment and transition. It must not feed back into IK or the standing pose.
"""

import asyncio
import copy
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
SPEC = importlib.util.spec_from_file_location("height_offset_bridge", SCRIPT)
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


def pose():
    qpos = list(bridge.IDLE_REFERENCE)
    qpos[:3] = [8., -3., .85]
    # Rz(.4) Rx(.2), keeping a nontrivial orientation for the passthrough test.
    cy, sy, cr, sr = math.cos(.2), math.sin(.2), math.cos(.1), math.sin(.1)
    qpos[3:7] = [cy*cr, cy*sr, sy*sr, sy*cr]
    qpos[7:] = [value+.01*(index+1) for index, value in enumerate(qpos[7:])]
    return qpos


def solution(qpos, seq=1, now=10.):
    return {"seq": seq, "qpos": list(qpos), "received_monotonic_s": now,
            "source_time_s": seq*.02, "worker_roundtrip_ms": .1,
            "stats": {"solver_ms": .05}}


def publish_yaml(offset=None, *, rate=50, steps=100):
    offset_line = "" if offset is None else f"  root_z_offset_m: {offset}\n"
    return ("reference:\n"
            "  topic: /tracking/current_reference\n"
            f"  rate_hz: {rate}\n"
            "  qos_depth: 21\n"
            "  reliability: reliable\n"
            f"  transition_steps: {steps}\n"
            "  source_timeout_ms: 1000\n" + offset_line)


class HeightOffsetGateTests(unittest.TestCase):
    def assertPoseAlmostEqual(self, actual, expected):
        self.assertEqual(len(actual), 35)
        self.assertEqual(len(expected), 35)
        for index, (value, target) in enumerate(zip(actual, expected)):
            self.assertAlmostEqual(value, target, places=12, msg=f"qpos[{index}]")

    def test_default_zero_preserves_existing_reference(self):
        implicit = bridge.ReferenceGate(timeout_s=10.)
        explicit = bridge.ReferenceGate(timeout_s=10., root_z_offset_m=0.)
        for gate in (implicit, explicit):
            gate.accept(solution(pose()))
            gate.activate(10.)
        for now in (10., 10.5, 11., 12., 12.02):
            self.assertPoseAlmostEqual(implicit.tick(now), explicit.tick(now))
        self.assertAlmostEqual(implicit.output[2], pose()[2], places=12)

    def test_positive_and_negative_offsets_only_change_live_root_z(self):
        for offset in (.08, -.146, 1, -1):
            with self.subTest(offset=offset):
                baseline = bridge.ReferenceGate(timeout_s=10.)
                shifted = bridge.ReferenceGate(timeout_s=10., root_z_offset_m=offset)
                for gate in (baseline, shifted):
                    gate.accept(solution(pose()))
                    gate.activate(10.)
                expected = baseline.tick(12.)
                actual = shifted.tick(12.)
                self.assertAlmostEqual(actual[2], expected[2]+offset, places=12)
                self.assertPoseAlmostEqual(actual[:2]+[expected[2]]+actual[3:], expected)
                # This is a translation, not a new IK/limiter pass: even an
                # intentionally below-ground output is retained unchanged.
                self.assertEqual(actual[7:], expected[7:])
                if offset == -1:
                    self.assertLess(actual[2], 0.)

    def test_repeated_proposals_and_duplicate_sequences_never_accumulate_offset(self):
        raw_pose = list(bridge.IDLE_REFERENCE)
        raw_pose[:3] = [8., -3., .85]
        raw = solution(raw_pose)
        original = copy.deepcopy(raw)
        gate = bridge.ReferenceGate(timeout_s=10., root_z_offset_m=.075)
        gate.accept(raw)
        gate.activate(10.)
        first = gate.tick(12.)
        for index in range(20):
            self.assertPoseAlmostEqual(gate.proposal(12.+index*.02), first)
            self.assertPoseAlmostEqual(gate.tick(12.+index*.02), first)
        # A duplicate sequence cannot replace the accepted raw IK solution.
        gate.accept(solution([*raw_pose[:2], 9., *raw_pose[3:]], seq=1, now=12.5))
        self.assertPoseAlmostEqual(gate.tick(12.5), first)
        self.assertEqual(raw, original)
        self.assertEqual(raw_pose, original["qpos"])
        self.assertEqual(gate.latest["qpos"], original["qpos"])
        self.assertAlmostEqual(first[2], .925, places=12)

    def test_two_second_entry_and_exit_blend_includes_offset_only_in_live_target(self):
        gate = bridge.ReferenceGate(timeout_s=10., blend_s=2., root_z_offset_m=.12)
        stand = list(gate.output)
        raw = pose()
        gate.accept(solution(raw))
        gate.activate(10.)
        self.assertPoseAlmostEqual(gate.tick(10.), stand)
        self.assertAlmostEqual(gate.tick(11.)[2], (stand[2]+raw[2]+.12)/2., places=12)
        live = gate.tick(12.)
        self.assertAlmostEqual(live[2], raw[2]+.12, places=12)
        gate.idle(12.)
        self.assertPoseAlmostEqual(gate.tick(12.), live)
        self.assertAlmostEqual(gate.tick(13.)[2], (live[2]+stand[2])/2., places=12)
        rested = gate.tick(14.)
        self.assertEqual(rested[2], bridge.IDLE_REFERENCE[2])
        self.assertEqual(rested[7:], bridge.IDLE_REFERENCE[7:])
        self.assertEqual(rested[:2], live[:2])
        # Reactivation starts from the current stand pose, and applies once.
        gate.accept(solution(raw, seq=2, now=14.01))
        gate.activate(14.01)
        self.assertPoseAlmostEqual(gate.tick(14.01), rested)
        self.assertAlmostEqual(gate.tick(16.01)[2], raw[2]+.12, places=12)

    def test_stale_source_returns_to_unshifted_stand_without_reactivation(self):
        gate = bridge.ReferenceGate(timeout_s=.25, blend_s=2., root_z_offset_m=-.08)
        raw = pose()
        gate.accept(solution(raw))
        gate.activate(10.)
        # Refresh the same synthetic pose during its two-second entry.
        for index in range(1, 11):
            now = 10.+index*.2
            gate.accept(solution(raw, seq=index+1, now=now))
            gate.tick(now)
        live = list(gate.output)
        self.assertAlmostEqual(live[2], raw[2]-.08, places=12)
        self.assertPoseAlmostEqual(gate.tick(12.26), live)
        self.assertEqual(gate.mode, "idle")
        self.assertEqual(gate.stale_count, 1)
        self.assertAlmostEqual(gate.tick(13.26)[2], (live[2]+bridge.IDLE_REFERENCE[2])/2., places=12)
        self.assertEqual(gate.tick(14.26)[2], bridge.IDLE_REFERENCE[2])
        gate.accept(solution(raw, seq=20, now=14.27))
        self.assertEqual(gate.tick(14.28)[2], bridge.IDLE_REFERENCE[2])
        self.assertEqual(gate.mode, "idle")
        self.assertEqual(gate.stale_count, 1)

    def test_constructor_rejects_nonfinite_and_non_numeric_offsets(self):
        for offset in (float("nan"), float("inf"), -float("inf"), True, False,
                       ".1", "0", None, [], {}):
            with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, "root_z_offset_m"):
                bridge.ReferenceGate(root_z_offset_m=offset)


class HeightOffsetConfigTests(unittest.TestCase):
    def load(self, offset=None, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "publish.yaml"
            path.write_text(publish_yaml(offset, **kwargs))
            return bridge.load_publish_config(path)

    def test_missing_key_defaults_to_zero_and_finite_yaml_numbers_are_accepted(self):
        self.assertEqual(self.load()["reference"]["root_z_offset_m"], 0.)
        for text, expected in (("0", 0.), ("1", 1.), ("-.146", -.146),
                               ("0.075", .075), ("1e-2", .01), ("-2E-2", -.02)):
            with self.subTest(text=text):
                self.assertEqual(self.load(text)["reference"]["root_z_offset_m"], expected)

    def test_yaml_rejects_nonfinite_strings_bools_and_empty_values(self):
        for text in ("nan", ".nan", "inf", ".inf", "-.inf", "1e309", "-1e309",
                     '"0.1"', "'0.1'", "true", "false", "null", "[]", "{}", ""):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "root_z_offset_m"):
                self.load(text)

    def test_offset_does_not_relax_fixed_50_hz_interface(self):
        self.assertEqual(self.load(".05")["reference"]["rate_hz"], 50)
        for rate in (49, 100):
            with self.subTest(rate=rate), self.assertRaisesRegex(ValueError, "50"):
                self.load(".05", rate=rate)


class HeightOffsetPublicationTests(unittest.IsolatedAsyncioTestCase):
    async def test_mocked_replay_sends_same_shifted_reference_to_ros_viser_and_jsonl(self):
        joint_names = json.loads(bridge.DEFAULT_JOINT_CONFIG.read_text())["urdfJointNames"]
        publishers, viewers, solver_inputs = [], [], []

        class FakeWorker:
            def __init__(self, command, accept, timeout, startup_timeout, on_ready):
                self.accept, self.on_ready = accept, on_ready
                self.ready, self.process = asyncio.Event(), None
                self.metadata = {"joint_names": joint_names, "model": "height-offset fixture"}
                self.startup_ms = [.1]
                self.solved = self.errors = self.restarts = 0
                self.last_error = None
                self.mailbox = bridge.LatestFrameMailbox()

            async def run(self):
                self.on_ready(self.metadata)
                self.ready.set()
                while True:
                    frame = await self.mailbox.take()
                    qpos = list(bridge.IDLE_REFERENCE)
                    qpos[:3] = [2.+frame["seq"]*.01, -3., .85+frame["seq"]*.001]
                    qpos[7] += frame["seq"]*.02
                    sample = {**frame, "qpos": qpos, "worker_roundtrip_ms": .1,
                              "stats": {"solver_ms": .05}}
                    solver_inputs.append((sample, copy.deepcopy(sample)))
                    self.solved += 1
                    self.accept(sample)

        class FakeReplay:
            times, speed, skipped = [0., .28], 1., 0

            def __init__(self, path, speed):
                pass

            async def run(self, submit, stop):
                for index in range(15):
                    submit({"seq": index, "source_time_s": index*.02,
                            "received_monotonic_s": time.monotonic(), "poses": {}})
                    await asyncio.sleep(.02)

        class FakeROS:
            reference_topic = "/tracking/current_reference"

            def __init__(self, config, *args, **kwargs):
                self.config = config
                self.vectors = []
                publishers.append(self)

            def reference_contract(self):
                return {"topic": self.reference_topic, "rate_hz": self.config["reference"]["rate_hz"],
                        "size": 35, "reliability": self.config["reference"]["reliability"],
                        "depth": self.config["reference"]["qos_depth"]}

            def publish(self, vector, source):
                self.vectors.append(list(vector))

            def close(self, progress):
                return {"status": "complete"}

        class FakeViewer:
            def __init__(self, *args):
                self.status = {"enabled": True, "started": False, "failure": None}
                self.vectors = []
                viewers.append(self)

            async def start(self):
                self.status["started"] = True

            def offer(self, vector, **kwargs):
                self.vectors.append(list(vector))

            async def close(self):
                self.status["closed"] = True

        with tempfile.TemporaryDirectory() as directory, ExitStack() as patches:
            directory = Path(directory)
            config = directory / "publish.yaml"
            config.write_text(publish_yaml(".075", steps=1))
            args = bridge.parse_args(["--source", "replay", "--npz", "unused.npz", "--autostart",
                                      "--viser", "--replay-tail-s", "0", "--max-duration-s", ".32",
                                      "--publish-config", str(config),
                                      "--output", str(directory / "reference.jsonl"),
                                      "--metrics", str(directory / "metrics.json")])
            for name, replacement in (("SolverWorker", FakeWorker), ("ReplaySource", FakeReplay),
                                      ("ROSPublisher", FakeROS), ("IntegratedViser", FakeViewer)):
                patches.enter_context(mock.patch.object(bridge, name, replacement))
            for forbidden in ("ReferenceSafetyWorker", "OutputLimiter", "create_live_source",
                              "recover_publication_timing", "check_published_motion"):
                patches.enter_context(mock.patch.object(bridge, forbidden,
                    side_effect=AssertionError(f"{forbidden} must not run in this test")))
            patches.enter_context(mock.patch.object(bridge, "start_cleanup_watchdog", return_value=None))
            patches.enter_context(mock.patch.object(bridge, "stop_cleanup_watchdog"))
            patches.enter_context(mock.patch.object(bridge.sys.stdin, "isatty", return_value=False))
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                metrics = await bridge.run_bridge(args)
            rows = [json.loads(line) for line in (directory / "reference.jsonl").read_text().splitlines()]
            saved_metrics = json.loads((directory / "metrics.json").read_text())

        self.assertEqual(len(publishers), 1)
        self.assertEqual(len(viewers), 1)
        self.assertGreaterEqual(len(rows), 8)
        self.assertEqual(publishers[0].vectors, viewers[0].vectors)
        self.assertEqual(publishers[0].vectors, [row["qpos"] for row in rows])
        self.assertTrue(viewers[0].status["closed"])
        stable = [row for row in rows if row["mode"] == "live" and not row["blend_active"]]
        self.assertGreaterEqual(len(stable), 4)
        for row in rows:
            self.assertEqual(len(row["qpos"]), 35)
            self.assertEqual(row["root_z_offset_m"], .075)
            self.assertFalse(row["final_constraints_enabled"])
            self.assertEqual(row["qpos"], row["desired_qpos"])
        for row in stable:
            self.assertAlmostEqual(row["qpos"][2], row["solver_qpos"][2]+.075, places=12)
            self.assertEqual(row["qpos"][7:], row["solver_qpos"][7:])
        for sample, before in solver_inputs:
            self.assertEqual(sample, before)
        self.assertEqual(metrics["root_z_offset_m"], .075)
        self.assertEqual(saved_metrics["root_z_offset_m"], .075)
        self.assertIn("fixed_root_z_offset", metrics["reference_stages"])
        self.assertEqual(metrics["reference_pipeline"], "alignment_transition_only")
        self.assertFalse(metrics["final_constraints_enabled"])
        self.assertIsNone(metrics["reference_fault"])
        self.assertEqual(metrics["process_cleanup"]["status"], "complete")
        self.assertEqual(metrics["ros_reference_publisher"]["rate_hz"], 50)
        # Short mocked wall-clock replay checks the existing 20 ms scheduler,
        # allowing normal CI jitter without pretending to measure PN latency.
        self.assertGreater(metrics["publish_hz"], 40.)
        self.assertLess(metrics["publish_hz"], 60.)


if __name__ == "__main__":
    unittest.main()
