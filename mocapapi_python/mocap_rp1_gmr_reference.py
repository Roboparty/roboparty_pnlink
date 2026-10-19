"""PN body world poses -> isolated GMR/mink worker -> 50 Hz RP1 reference.

The main process never imports MuJoCo/mink or calls IK. Only one PN worker
request can be in flight; newer PN samples replace a one-frame mailbox.
The reference stage applies the configured fixed root-height offset to live
retarget poses, initial planar alignment, and the activation or stand transition.
The offset/aligned/blended retarget pose is published directly;
it does not run another solver, limit motion, or acquire foot anchors.
Replay --dry-run does not import ROS or open a PN connection. Live input
opens one MCPBase connection and retains N/C/F/R/0/O/Z keyboard commands.
A activates a fresh solution with planar alignment and a two-second blend;
X, stale input, or solver failure returns to stand and requires a new A.

Worker stdin/stdout are JSON lines. Input poses are world metres and wxyz.
Output qpos is [root xyz, root wxyz, 28 joints in rp1/retarget.json order].

Examples (replay opens neither PN nor hardware):
  python mocap_rp1_gmr_reference.py --source replay --npz TAKE/dataset.npz \
      --dry-run --autostart --max-duration-s 8 --metrics /tmp/timing.json
  python mocap_rp1_gmr_reference.py --source replay --npz TAKE/dataset.npz \
      --reference-topic /pn_retarget_test/reference --autostart --source-viz
  python mocap_rp1_gmr_reference.py --source live
  python mocap_rp1_gmr_reference.py --source live --no-viser

Use the ROS Python runtime for publishing; --solver-python selects the
independent GMR environment. Run replay ROS checks in an isolated ROS_DOMAIN_ID.
--max-duration-s measures source playback after the worker ready handshake;
50 Hz stand messages continue while the retarget worker initializes. --output JSONL
records time_s (bridge clock), source_time_s (clip/PN clock), seq, mode,
source_age_ms, worker_roundtrip_ms, solver_stats, blend_active, raw solver_qpos,
and desired_qpos / qpos (identical offset/aligned/blended vectors). Both logs and
metrics identify reference_pipeline=alignment_transition_only and
final_constraints_enabled=false, together with root_z_offset_m (metres, positive
raises the live target; the idle stand is unchanged). Retarget constraints remain in the core IK;
the reference stage does not enforce motion or floor limits during transitions.
Live publishing also starts a Viser model at http://127.0.0.1:8891/ by default.
It receives the exact published reference and latest raw PN body over a separate latest-pose pipe;
--no-viser disables it. Replay/dry-run starts no viewer unless --viser is set.
The /mocap/hand/* PoseArray messages contain PN finger bones only (up to 19
per side, excluding the wrist). Wrist input is retained for body/FK processing.
Live UDP endpoints are read from the selected publish.yaml's pn_network section;
explicit --pn-* options override YAML, followed by legacy PN_* environment values.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import signal
import subprocess
import sys
import termios
import time
import tty
from typing import Callable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from pn_network import PNNetworkConfig, PNNetworkError, load_pn_network, preflight_pn_network, validate_pn_network
from pn_rp1_paths import resolve_python_runtime

DEFAULT_PUBLISH_CONFIG = SCRIPT_DIR / "rp1" / "publish.yaml"
DEFAULT_JOINT_CONFIG = SCRIPT_DIR / "rp1" / "retarget.json"
DEFAULT_CALIBRATION = SCRIPT_DIR / "rp1" / "pn_axis_to_rp1_28dof_current.json"
DEFAULT_WORKER_SCRIPT = SCRIPT_DIR / "pn_rp1_retarget.py"
DEFAULT_SOLVER_PYTHON = resolve_python_runtime('solver')
DEFAULT_VISER_PYTHON = resolve_python_runtime('viser')
DEFAULT_VISER_SCRIPT = SCRIPT_DIR / "view_rp1_gmr_3d.py"
REFERENCE_PIPELINE = "alignment_transition_only"
_FINAL_CLEANUP_WATCHDOG = None
BODY_PARENTS = {
    "Hips": "world", "RightUpLeg": "Hips", "RightLeg": "RightUpLeg",
    "RightFoot": "RightLeg", "LeftUpLeg": "Hips", "LeftLeg": "LeftUpLeg",
    "LeftFoot": "LeftLeg", "Spine": "Hips", "Spine1": "Spine",
    "Spine2": "Spine1", "Neck": "Spine2", "Neck1": "Neck",
    "Head": "Neck1", "RightShoulder": "Spine2", "RightArm": "RightShoulder",
    "RightForeArm": "RightArm", "RightHand": "RightForeArm",
    "LeftShoulder": "Spine2", "LeftArm": "LeftShoulder",
    "LeftForeArm": "LeftArm", "LeftHand": "LeftForeArm",
}
IDLE_REFERENCE = [
    0., 0., .78, 1., 0., 0., 0.,
    -.1, 0., 0., -.3, -.2, 0., .1, 0., 0., .3, -.2, 0., 0., 0.,
    .2, .2, 0., -1.2, 0., 0., 0., -.2, -.2, 0., 1.2, 0., 0., 0.,
]


def normalize_quat(q):
    q = [float(v) for v in q]
    norm = math.sqrt(sum(v * v for v in q))
    if len(q) != 4 or not math.isfinite(norm) or norm < 1e-10:
        raise ValueError("quaternion must have four finite, nonzero components")
    return [v / norm for v in q]


def quat_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
            aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw]


def rotate_vector(q, v):
    p = quat_multiply(quat_multiply(q, [0., *v]), [q[0], -q[1], -q[2], -q[3]])
    return p[1:]


def published_world_velocity(previous, current, dt):
    """MuJoCo free-joint velocity convention, with angular velocity in world."""
    relative = quat_multiply(current[3:7], [previous[3], -previous[4], -previous[5], -previous[6]])
    if relative[0] < 0:
        relative = [-v for v in relative]
    sine = math.sqrt(sum(v*v for v in relative[1:]))
    angle = 2*math.atan2(sine, max(0., relative[0]))
    angular = [v*angle/max(sine*dt, 1e-15) for v in relative[1:]]
    return [*( (b-a)/dt for a, b in zip(previous[:3], current[:3]) ), *angular,
            *( (b-a)/dt for a, b in zip(previous[7:], current[7:]) )]


def check_published_motion(velocity, previous_velocity, dt, limiter):
    """Hard check at the actual publication time; never clip a guarded pose."""
    acceleration = [(a-b)/dt for a, b in zip(velocity, previous_velocity)]
    joint_v = limiter.joint_speed if isinstance(limiter.joint_speed, list) else [limiter.joint_speed]*28
    joint_a = limiter.joint_acceleration if isinstance(limiter.joint_acceleration, list) else [limiter.joint_acceleration]*28
    for actual, maximum in zip(velocity[6:], joint_v):
        if abs(actual) > maximum+1e-5:
            raise RuntimeError("actual publication time would exceed a joint speed limit")
    for actual, maximum in zip(acceleration[6:], joint_a):
        if abs(actual) > maximum+1e-4:
            raise RuntimeError("actual publication time would exceed a joint acceleration limit")
    for actual, maximum in ((_norm(velocity[:3]), limiter.root_speed),
                            (_norm(velocity[3:6]), limiter.angular_speed),
                            (_norm(acceleration[:3]), limiter.root_acceleration),
                            (_norm(acceleration[3:6]), limiter.angular_acceleration)):
        if actual > maximum+1e-4:
            raise RuntimeError("actual publication time would exceed a root motion limit")


async def recover_publication_timing(safety, seq, gate, previous, previous_velocity,
                                    previous_tick, limiter, vector, stats, source,
                                    *, clock=time.monotonic, max_retries=2, budget_s=.08):
    """Re-solve a late candidate from the unchanged actual publication state.

    No component clipping or prediction integration is used. Each candidate
    comes from a new coherent QP and must pass the actual-time hard check.
    """
    recovery_started = clock()
    attempts = 0
    while True:
        now = clock()
        dt = now-previous_tick
        achieved = published_world_velocity(previous, vector, dt)
        try:
            check_published_motion(achieved, previous_velocity, dt, limiter)
            stats = {**stats, "timing_reprojections": attempts}
            return vector, stats, source, now, achieved, attempts
        except RuntimeError:
            if attempts >= max_retries:
                raise RuntimeError("actual-time safety check failed after bounded timing reprojection")
        remaining = budget_s-(clock()-recovery_started)
        if remaining <= 0:
            raise RuntimeError("timing reprojection exceeded its 80 ms recovery budget")
        measured = safety.roundtrips_ms[-20:]
        estimate = sorted(measured)[len(measured)//2]/1000. if measured else .0025
        estimate = min(.010, max(.001, estimate))
        predicted_publish = clock()+estimate
        predicted_dt = predicted_publish-previous_tick
        if not .001 <= predicted_dt <= .1:
            raise RuntimeError("timing reprojection period is outside the 1–100 ms safety range")
        proposal = gate.proposal(predicted_publish)
        source = gate.latest
        safety.timing_reprojections = getattr(safety, "timing_reprojections", 0)+1
        result = await asyncio.wait_for(safety.project(seq, previous, previous_velocity,
                                                     proposal, predicted_dt), remaining)
        vector, stats = result["qpos"], result["stats"]
        attempts += 1


def slerp(a, b, fraction):
    a, b = normalize_quat(a), normalize_quat(b)
    dot = sum(x*y for x, y in zip(a, b))
    if dot < 0:
        b, dot = [-v for v in b], -dot
    dot = min(1., max(-1., dot))
    if dot > .9995:
        return normalize_quat([(1-fraction)*x+fraction*y for x, y in zip(a, b)])
    angle = math.acos(dot)
    u, v = math.sin((1-fraction)*angle)/math.sin(angle), math.sin(fraction*angle)/math.sin(angle)
    return [u*x+v*y for x, y in zip(a, b)]


def yaw(q):
    w, x, y, z = normalize_quat(q)
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def yaw_quat(angle):
    return [math.cos(angle/2), 0., 0., math.sin(angle/2)]


def validated_qpos(values):
    if len(values) != 35:
        raise ValueError("worker qpos must contain 35 values")
    qpos = [float(v) for v in values]
    if not all(math.isfinite(v) for v in qpos):
        raise ValueError("worker qpos contains a nonfinite value")
    qpos[3:7] = normalize_quat(qpos[3:7])
    return qpos


def world_poses_from_sdk_locals(local_poses, parents=BODY_PARENTS):
    """Same FK/axis map as mocap_rp1_record.globals_from_locals.

    SDK local translations are absolute parent-frame centimetres, so no
    BVH OFFSET is added. ROS xyz = native zxy/100, ROS wxyz = native wzxy.
    """
    world = {}

    def visit(name):
        if name in world:
            return world[name]
        if name not in local_poses:
            return None
        p, q = local_poses[name]
        p = [float(p[2])/100., float(p[0])/100., float(p[1])/100.]
        q = normalize_quat([q[0], q[3], q[1], q[2]])
        parent = parents.get(name, "world")
        if parent != "world":
            parent_pose = visit(parent)
            if parent_pose is None:
                return None
            pp, pq = parent_pose
            p = [a+b for a, b in zip(pp, rotate_vector(pq, p))]
            q = normalize_quat(quat_multiply(pq, q))
        if not all(math.isfinite(v) for v in p):
            raise ValueError(f"{name} position is not finite")
        world[name] = [p, q]
        return world[name]

    for name in parents:
        visit(name)
    return world


def hand_parents():
    """Keep wrist parents for FK; publish only the 19 PN finger bones."""
    parents = dict(BODY_PARENTS)
    orders = {}
    for side in ("Left", "Right"):
        hand = f"{side}Hand"
        order = []
        previous = hand
        for i in (1, 2, 3):
            name = f"{side}HandThumb{i}"
            parents[name] = previous
            order.append(name)
            previous = name
        for finger in ("Index", "Middle", "Ring", "Pinky"):
            previous = hand
            for name in [f"{side}InHand{finger}", *(f"{side}Hand{finger}{i}" for i in (1, 2, 3))]:
                parents[name] = previous
                order.append(name)
                previous = name
        orders[side.lower()] = order
    return parents, orders


ALL_PARENTS, HAND_ORDERS = hand_parents()


class LatestFrameMailbox:
    """One pending frame. Replacement never blocks the PN poller."""

    def __init__(self):
        self.pending = None
        self.ready = asyncio.Event()
        self.replaced = 0
        self.offered = 0

    def offer(self, frame):
        self.offered += 1
        if self.pending is not None:
            self.replaced += 1
        self.pending = frame
        self.ready.set()

    async def take(self):
        await self.ready.wait()
        frame, self.pending = self.pending, None
        self.ready.clear()
        return frame


async def terminate_subprocess(process):
    """Bound shutdown even when an external runner retains a stdio pipe."""
    if process.stdin:
        process.stdin.close()
    if process.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
    try:
        await asyncio.wait_for(process.wait(), .5)
    except asyncio.TimeoutError:
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
        # asyncio Process has no public pipe-close method; its subprocess
        # transport owns both pipes. Closing them releases EOF waiters even
        # when the hosting runner retains a copied descriptor.
        transport = getattr(process, "_transport", None)
        if transport is not None:
            transport.close()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(process.wait(), .5)


def build_viser_command(args):
    """Use the browser runtime; the ROS bridge itself never imports Viser/FK."""
    return [str(args.viser_python), "-u", str(DEFAULT_VISER_SCRIPT),
            "--source", "stdin", "--config", str(args.calibration),
            "--host", args.viser_host, "--port", str(args.viser_port)]


class IntegratedViser:
    """Isolated display of the exact publication vector, with one pending pose.

    Sending is a background task: browser startup, rendering and pipe pressure
    cannot block the 50 Hz reference loop. The child drains stdin into its own
    latest-sample slot. It receives no PN connection and runs no retargeter.
    """
    def __init__(self, command):
        self.command = [str(value) for value in command]
        self.process = None
        self.task = None
        self.pending = None
        self.ready = asyncio.Event()
        self.startup_deadline_s = None
        self.status = {"enabled": True, "started": False, "offered": 0,
                       "replaced": 0, "sent": 0, "failure": None,
                       "closed": False}

    async def start(self):
        self.process = await asyncio.create_subprocess_exec(
            *self.command, stdin=asyncio.subprocess.PIPE,
            stdout=None, stderr=None,
        )
        self.status["started"] = True
        # Cold NumPy/model imports may precede the child's stdin reader.
        # Only this display task waits; the publication path remains direct.
        self.startup_deadline_s = time.monotonic()+10.
        self.task = asyncio.create_task(self._send(), name="pn-rp1-viser-pipe")

    def offer(self, vector, source=None, reference_source_time_s=None):
        if self.status["failure"] or self.status["closed"]:
            return
        self.status["offered"] += 1
        if self.pending is not None:
            self.status["replaced"] += 1
        packet = {"qpos": list(vector)}
        if reference_source_time_s is not None:
            packet["reference_source_time_s"] = reference_source_time_s
        if source is not None:
            try:
                positions = [[float(v) for v in source["poses"][name][0]] for name in BODY_PARENTS]
                if any(len(p) != 3 or not all(math.isfinite(v) for v in p) for p in positions):
                    raise ValueError("raw PN positions must be finite XYZ")
                packet["pn"] = {"joint_names": list(BODY_PARENTS), "positions_m": positions,
                                "seq": source["seq"], "source_time_s": source["source_time_s"],
                                "received_monotonic_s": source["received_monotonic_s"]}
            except (KeyError, IndexError, TypeError, ValueError, OverflowError):
                # A display-only source failure must not affect publication.
                pass
        self.pending = packet
        self.ready.set()

    async def _send(self):
        try:
            while True:
                await self.ready.wait()
                packet, self.pending = self.pending, None
                self.ready.clear()
                if self.process.returncode is not None:
                    raise RuntimeError(f"viewer exited ({self.process.returncode})")
                self.process.stdin.write((json.dumps(packet, allow_nan=False,
                                                    separators=(",", ":"))+"\n").encode())
                startup_remaining = max(0., self.startup_deadline_s-time.monotonic())
                await asyncio.wait_for(self.process.stdin.drain(), max(.25, startup_remaining))
                self.status["sent"] += 1
        except asyncio.CancelledError:
            raise
        except (OSError, RuntimeError, ValueError, asyncio.TimeoutError) as exc:
            self.status["failure"] = str(exc) or type(exc).__name__
            print(f"Viser display stopped: {self.status['failure']}; reference publication continues",
                  file=sys.stderr)

    async def close(self):
        self.status["closed"] = True
        if self.task is not None:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
            self.task = None
        if self.process is not None:
            process, self.process = self.process, None
            await terminate_subprocess(process)


class SolverWorker:
    def __init__(self, command, on_result: Callable, timeout_s=.5, startup_timeout_s=30., on_ready=None):
        self.command = [str(v) for v in command]
        self.on_result = on_result
        self.timeout_s = timeout_s
        self.startup_timeout_s = startup_timeout_s
        self.on_ready = on_ready
        self.mailbox = LatestFrameMailbox()
        self.process = None
        self.ready = asyncio.Event()
        self.startup_ms = []
        self.metadata = None
        self.solved = self.errors = self.restarts = 0
        self.last_error = None

    async def close(self):
        if self.process is None:
            return
        process, self.process = self.process, None
        await terminate_subprocess(process)

    async def run(self):
        try:
            while True:
                try:
                    self.ready.clear()
                    startup_started_s = time.monotonic()
                    self.process = await asyncio.create_subprocess_exec(
                        *self.command, stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE, stderr=None,
                    )
                    line = await asyncio.wait_for(self.process.stdout.readline(), self.startup_timeout_s)
                    if not line:
                        raise RuntimeError("GMR worker closed stdout before ready")
                    metadata = json.loads(line)
                    if metadata.get("type") != "ready":
                        raise ValueError("worker must send its ready metadata before accepting frames")
                    if self.on_ready:
                        self.on_ready(metadata)
                    self.metadata = metadata
                    self.startup_ms.append((time.monotonic()-startup_started_s)*1000.)
                    self.ready.set()
                    previous_dispatch_s = None
                    next_dispatch_s = 0.
                    while True:
                        # PN may arrive at 92+ Hz, but IK advances the reference
                        # at most once per 20 ms. Wait before taking so this
                        # time slot selects the newest pending PN sample.
                        delay = next_dispatch_s-time.monotonic()
                        if delay > 0:
                            await asyncio.sleep(delay)
                        frame = await self.mailbox.take()
                        start = time.monotonic()
                        payload = {k: frame[k] for k in ("seq", "source_time_s", "received_monotonic_s", "poses")}
                        payload["dt_s"] = (.02 if previous_dispatch_s is None else
                                           min(.1, max(.001, start-previous_dispatch_s)))
                        previous_dispatch_s, next_dispatch_s = start, start+.02
                        self.process.stdin.write((json.dumps(payload, allow_nan=False, separators=(",", ":"))+"\n").encode())
                        await asyncio.wait_for(self.process.stdin.drain(), self.timeout_s)
                        deadline = start + self.timeout_s
                        while True:
                            line = await asyncio.wait_for(self.process.stdout.readline(), max(.001, deadline-time.monotonic()))
                            if not line:
                                raise RuntimeError("GMR worker closed stdout")
                            result = json.loads(line)
                            if result.get("seq") != frame["seq"]:
                                raise ValueError("worker response sequence does not match its request")
                            if "error" in result:
                                raise RuntimeError(str(result["error"]))
                            result["qpos"] = validated_qpos(result["qpos"])
                            result["source_time_s"] = frame["source_time_s"]
                            result["received_monotonic_s"] = frame["received_monotonic_s"]
                            result["completed_monotonic_s"] = time.monotonic()
                            result["dispatch_monotonic_s"] = start
                            result["worker_roundtrip_ms"] = (result["completed_monotonic_s"]-start)*1000.
                            self.solved += 1
                            self.on_result(result)
                            break
                except asyncio.CancelledError:
                    raise
                except (OSError, RuntimeError, ValueError, KeyError, asyncio.TimeoutError) as exc:
                    self.errors += 1
                    self.ready.clear()
                    self.last_error = str(exc) or type(exc).__name__
                    print(f"GMR worker failed: {self.last_error}; returning to stand if stale", file=sys.stderr)
                    await self.close()
                    self.restarts += 1
                    await asyncio.sleep(.25)
        finally:
            await self.close()


class ReferenceSafetyWorker:
    """A serialized asynchronous IPC projector; no numerical imports here."""
    def __init__(self, command, timeout_s=.08, startup_timeout_s=10.):
        self.command = [str(v) for v in command]
        self.timeout_s, self.startup_timeout_s = timeout_s, startup_timeout_s
        self.process = None
        self.ready = asyncio.Event()
        self.roundtrips_ms, self.solve_ms = [], []
        self.startup_ms = None
        self.errors = self.fallbacks = 0
        self.timing_reprojections = 0
        self.last_error = None
        self.contact_provider = None

    async def close(self):
        if self.process is None:
            return
        process, self.process = self.process, None
        self.ready.clear()
        await terminate_subprocess(process)

    async def configure(self, metadata, initial):
        start = time.monotonic()
        self.process = await asyncio.create_subprocess_exec(*self.command,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=None)
        payload = {"type": "configure", "metadata": metadata, "initial_qpos": initial}
        self.process.stdin.write((json.dumps(payload, allow_nan=False)+"\n").encode())
        await asyncio.wait_for(self.process.stdin.drain(), self.startup_timeout_s)
        line = await asyncio.wait_for(self.process.stdout.readline(), self.startup_timeout_s)
        if not line or json.loads(line).get("type") != "ready":
            raise RuntimeError("final reference safety worker did not become ready")
        self.startup_ms = (time.monotonic()-start)*1000.
        self.ready.set()

    async def project(self, seq, published, previous_velocity, target, dt):
        if not self.ready.is_set():
            raise RuntimeError("final reference safety worker is not ready")
        started = time.monotonic()
        request = {"seq": seq, "published_qpos": published, "previous_world_velocity": previous_velocity,
                   "target_qpos": target, "dt_s": dt}
        request["contacts"] = self.contact_provider() if self.contact_provider is not None else None
        try:
            self.process.stdin.write((json.dumps(request, allow_nan=False, separators=(",", ":"))+"\n").encode())
            await asyncio.wait_for(self.process.stdin.drain(), self.timeout_s)
            line = await asyncio.wait_for(self.process.stdout.readline(), self.timeout_s)
            if not line:
                raise RuntimeError("final reference safety worker closed stdout")
            result = json.loads(line)
            if result.get("seq") != seq:
                raise ValueError("final reference safety response sequence mismatch")
            if "error" in result:
                raise RuntimeError(result["error"])
            result["qpos"] = validated_qpos(result["qpos"])
            self.roundtrips_ms.append((time.monotonic()-started)*1000.)
            self.solve_ms.append(result["stats"]["safety_ms"])
            self.fallbacks += bool(result["stats"].get("brake_fallback"))
            return result
        except Exception as exc:
            self.errors += 1
            self.last_error = str(exc) or type(exc).__name__
            self.ready.clear()
            raise


def _norm(values):
    return math.sqrt(sum(v*v for v in values))


def _clip_norm(values, limit):
    norm = _norm(values)
    return [v * min(1., limit/max(norm, 1e-12)) for v in values]


class OutputLimiter:
    """Limit the final published vector, including mode-transition blends.

    Scalar joints use conservative deployment speeds. Root linear and
    rotational velocities are norm-limited; acceleration applies to their
    world-frame velocity vectors. The integration period is the measured
    publisher interval, so scheduler jitter cannot rescale velocity history
    and bypass the final acceleration limits.
    """

    def __init__(self, initial=IDLE_REFERENCE, joint_speed=3., joint_acceleration=15.,
                 root_speed=1., root_acceleration=3., angular_speed=1.5, angular_acceleration=6.,
                 joint_bounds=None):
        self.output = validated_qpos(initial)
        self.linear_velocity = [0., 0., 0.]
        self.angular_velocity = [0., 0., 0.]
        self.joint_velocity = [0.]*28
        self.joint_speed, self.joint_acceleration = joint_speed, joint_acceleration
        self.root_speed, self.root_acceleration = root_speed, root_acceleration
        self.angular_speed, self.angular_acceleration = angular_speed, angular_acceleration
        self.joint_bounds = joint_bounds

    def configure_from_worker(self, ready, joint_names):
        if ready.get("joint_names") != list(joint_names):
            raise ValueError("worker joint order does not match the deployed 28-joint reference order")
        lower, upper = ready["joint_lower"], ready["joint_upper"]
        velocity, acceleration = ready["velocity_limits"], ready["acceleration_limits"]
        if any(len(v) != 28 for v in (lower, upper, velocity, acceleration)):
            raise ValueError("worker limit metadata must contain 28 values")
        for lo, hi, v, a in zip(lower, upper, velocity, acceleration):
            if not all(math.isfinite(x) for x in (lo, hi, v, a)) or lo >= hi or v <= 0 or a <= 0:
                raise ValueError("invalid worker limit metadata")
        self.joint_bounds = list(zip(lower, upper))
        self.joint_speed = list(velocity)
        self.joint_acceleration = list(acceleration)
        self.root_speed = float(ready["root_linear_velocity_limit_m_s"])
        self.root_acceleration = float(ready["root_linear_acceleration_limit_m_s2"])
        self.angular_speed = float(ready["root_angular_velocity_limit_rad_s"])
        self.angular_acceleration = float(ready["root_angular_acceleration_limit_rad_s2"])
        if any(not math.isfinite(x) or x <= 0 for x in
               (self.root_speed, self.root_acceleration, self.angular_speed, self.angular_acceleration)):
            raise ValueError("invalid worker root limits")

    def apply(self, target, dt):
        target = validated_qpos(target)
        dt = max(1e-5, float(dt))
        if not math.isfinite(dt):
            raise ValueError("publisher interval must be finite")
        old = self.output
        desired = _clip_norm([(b-a)/dt for a, b in zip(old[:3], target[:3])], self.root_speed)
        change = _clip_norm([b-a for a, b in zip(self.linear_velocity, desired)], self.root_acceleration*dt)
        self.linear_velocity = [a+b for a, b in zip(self.linear_velocity, change)]
        new = list(old)
        new[:3] = [a+b*dt for a, b in zip(old[:3], self.linear_velocity)]
        relative = quat_multiply(target[3:7], [old[3], -old[4], -old[5], -old[6]])
        if relative[0] < 0:
            relative = [-v for v in relative]
        sine = _norm(relative[1:])
        angle = 2*math.atan2(sine, max(0., relative[0]))
        desired = _clip_norm([v * angle / max(sine*dt, 1e-12) for v in relative[1:]], self.angular_speed)
        change = _clip_norm([b-a for a, b in zip(self.angular_velocity, desired)], self.angular_acceleration*dt)
        self.angular_velocity = [a+b for a, b in zip(self.angular_velocity, change)]
        step_angle = _norm(self.angular_velocity)*dt
        step_q = [math.cos(step_angle/2), *[v*dt*math.sin(step_angle/2)/max(step_angle, 1e-12) for v in self.angular_velocity]]
        new[3:7] = normalize_quat(quat_multiply(step_q, old[3:7]))
        for i in range(28):
            joint_speed = self.joint_speed[i] if isinstance(self.joint_speed, (list, tuple)) else self.joint_speed
            joint_acceleration = self.joint_acceleration[i] if isinstance(self.joint_acceleration, (list, tuple)) else self.joint_acceleration
            target_joint = target[7+i]
            if self.joint_bounds:
                lower, upper = self.joint_bounds[i]
                target_joint = max(lower, min(upper, target_joint))
            desired_v = max(-joint_speed, min(joint_speed, (target_joint-old[7+i])/dt))
            if self.joint_bounds:
                # Leave braking distance before either hard model boundary.
                lower, upper = self.joint_bounds[i]
                brake = lambda d: max(0., math.sqrt(max(0., 2*joint_acceleration*d)) - joint_acceleration*dt)
                desired_v = max(-brake(old[7+i]-lower), min(brake(upper-old[7+i]), desired_v))
            previous_v = self.joint_velocity[i]
            velocity = max(previous_v-joint_acceleration*dt,
                           min(previous_v+joint_acceleration*dt, desired_v))
            new[7+i] = old[7+i]+velocity*dt
            self.joint_velocity[i] = velocity
        self.output = validated_qpos(new)
        return list(self.output)


def validated_root_z_offset(value):
    """Return a finite numeric height offset; strings/bools are configuration errors."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("reference.root_z_offset_m must be a finite number in metres")
    try:
        value = float(value)
    except (OverflowError, ValueError):
        raise ValueError("reference.root_z_offset_m must be a finite number in metres") from None
    if not math.isfinite(value):
        raise ValueError("reference.root_z_offset_m must be a finite number in metres")
    return value


class ReferenceGate:
    def __init__(self, timeout_s=1., blend_s=2., limiter=None, *, root_z_offset_m=0.):
        self.timeout_s, self.blend_s = timeout_s, blend_s
        self.root_z_offset_m = validated_root_z_offset(root_z_offset_m)
        # The optional limiter is retained as legacy test state only. Neither
        # tick() nor the production bridge invokes it or creates one.
        self.limiter = limiter
        self._output = validated_qpos(limiter.output if limiter is not None else IDLE_REFERENCE)
        self.latest = None
        self.mode = "idle"
        self.origin = None
        self.hold = (0., 0., 0.)
        self.blend_from = None
        self.blend_started_s = 0.
        self.stale_count = 0

    @property
    def output(self):
        return self._output

    def accept(self, result):
        result["qpos"] = validated_qpos(result["qpos"])
        if self.latest is None or result["seq"] > self.latest["seq"]:
            self.latest = result

    def fresh(self, now):
        return (self.latest is not None and
                0 <= now-self.latest["received_monotonic_s"] <= self.timeout_s)

    def activate(self, now):
        if not self.fresh(now):
            return False
        source, robot = self.latest["qpos"], self.output
        self.origin = (source[0], source[1], yaw(source[3:7]), robot[0], robot[1], yaw(robot[3:7]))
        self._begin("live", now)
        return True

    def idle(self, now):
        self.hold = (self.output[0], self.output[1], yaw(self.output[3:7]))
        self.origin = None
        self._begin("idle", now)

    def _begin(self, mode, now):
        self.mode, self.blend_from, self.blend_started_s = mode, list(self.output), now

    def proposal(self, now):
        if self.mode == "live" and not self.fresh(now):
            self.stale_count += 1
            self.idle(now)
        if self.mode == "live":
            sx, sy, syaw, rx, ry, ryaw = self.origin
            target = list(self.latest["qpos"])
            # Always start from the raw IK pose. Apply this world-Z translation
            # once, before alignment/transition; never edit or accumulate in latest.
            target[2] += self.root_z_offset_m
            dx, dy, heading = target[0]-sx, target[1]-sy, ryaw-syaw
            c, s = math.cos(heading), math.sin(heading)
            target[:2] = [rx+c*dx-s*dy, ry+s*dx+c*dy]
            target[3:7] = quat_multiply(yaw_quat(heading), target[3:7])
        else:
            target = list(IDLE_REFERENCE)
            target[:2] = self.hold[:2]
            target[3:7] = yaw_quat(self.hold[2])
        if self.blend_from is not None:
            alpha = min(1., max(0., (now-self.blend_started_s)/self.blend_s))
            target_quaternion = target[3:7]
            target = [(1-alpha)*a+alpha*b for a, b in zip(self.blend_from, target)]
            target[3:7] = slerp(self.blend_from[3:7], target_quaternion, alpha)
            if alpha == 1.:
                self.blend_from = None
        return validated_qpos(target)

    def tick(self, now, dt=.02):
        vector = self.proposal(now)
        self.synchronize(vector)
        return vector

    def synchronize(self, qpos):
        self._output = validated_qpos(qpos)
        if self.limiter is not None:
            self.limiter.output = list(self._output)

    def support_contacts(self):
        """Legacy support accessor; the publication path does not use it."""
        if self.mode != "live" or self.latest is None:
            return None
        contacts = self.latest.get("stats", {}).get("contacts")
        if not isinstance(contacts, dict):
            return None
        return {side: bool(contacts.get(side, False)) for side in ("left", "right")}


def load_publish_config(path):
    """Read the repository's two-level YAML without importing ROS/PyYAML."""
    output, section = {}, None
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        key, value = line.strip().split(":", 1)
        if not line.startswith(" "):
            section = output.setdefault(key, {})
            continue
        value = value.strip()
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        else:
            try:
                if section is output.get("reference") and key == "root_z_offset_m":
                    # This scalar accepts scientific notation too. Quotes stay
                    # strings, and YAML .nan/.inf are rejected by validation.
                    value = float(value)
                else:
                    value = float(value) if "." in value else int(value)
            except ValueError:
                pass
        section[key] = value
    reference = output.get("reference", {})
    if float(reference.get("rate_hz", 0)) != 50.:
        raise ValueError("RP1 GMR bridge requires reference.rate_hz = 50")
    if not str(reference.get("topic", "")).startswith("/"):
        raise ValueError("reference topic must be absolute")
    if reference.get("reliability") not in ("reliable", "best_effort") or int(reference.get("qos_depth", 0)) < 1:
        raise ValueError("invalid reference QoS")
    if float(reference.get("source_timeout_ms", 0)) <= 0 or int(reference.get("transition_steps", 0)) < 1:
        raise ValueError("source timeout and transition_steps must be positive")
    reference["root_z_offset_m"] = validated_root_z_offset(reference.get("root_z_offset_m", 0.))
    return output


class ReplaySource:
    def __init__(self, path, speed=1.):
        import numpy as np
        with np.load(path, allow_pickle=False) as data:
            self.times = data["timestamps_s"].astype(float)
            self.names = [str(n) for n in data["joint_names"]]
            self.position = data["global_position_ros_m"].astype(float)
            self.rotation = data["global_rotation_ros_xyzw"].astype(float)
        if len(self.times) < 2 or not np.all(np.diff(self.times) >= 0):
            raise ValueError("replay requires at least two frames with nondecreasing timestamps")
        if not np.all(np.isfinite(self.times)) or not np.all(np.isfinite(self.position)):
            raise ValueError("replay contains nonfinite data")
        if self.position.shape != (len(self.times), len(self.names), 3) or self.rotation.shape != (len(self.times), len(self.names), 4):
            raise ValueError("replay pose array shapes do not match timestamps/joint_names")
        if not set(BODY_PARENTS).issubset(self.names):
            raise ValueError("replay does not contain the complete PN 21-joint body")
        self.times -= self.times[0]
        self.speed = speed
        self.emitted = self.skipped = 0

    def frame(self, index, received_s):
        poses = {}
        for j, name in enumerate(self.names):
            xyzw = self.rotation[index, j]
            poses[name] = [self.position[index, j].tolist(), normalize_quat([xyzw[3], *xyzw[:3]])]
        return {"seq": index, "source_time_s": float(self.times[index]),
                "received_monotonic_s": received_s, "poses": poses}

    async def run(self, submit, stop):
        start = time.monotonic()
        index = 0
        while index < len(self.times) and not stop.is_set():
            now = time.monotonic()
            due = start+float(self.times[index])/self.speed
            if due > now:
                await asyncio.sleep(min(due-now, .02))
                continue
            # An event-loop delay must not enqueue a burst of old frames.
            while index+1 < len(self.times) and start+float(self.times[index+1])/self.speed <= now:
                self.skipped += 1
                index += 1
            submit(self.frame(index, now))
            self.emitted += 1
            index += 1


def create_live_source(submit, on_key, network=None):
    # Lazy import: replay/dry-run cannot accidentally load or connect PN SDK.
    from pn_sdk import load_sdk
    sdk, MCPBase = load_sdk()
    MCPAvatar = sdk.MCPAvatar

    class PNLiveSource(MCPBase):
        def __init__(self):
            if network is None:
                super().__init__()
            else:
                super().__init__(local_ip=network.local_ip, local_port=network.local_port,
                                 server_ip=network.server_ip, server_port=network.server_port)
            self.sequence = 0
            self.origin_s = time.monotonic()

        def handleAvatar(self, avatar_handle):
            now = time.monotonic()
            try:
                locals_ = {}
                for joint in MCPAvatar(avatar_handle).get_joints():
                    name = joint.get_name()
                    if name not in ALL_PARENTS:
                        continue
                    position = joint.get_local_position()
                    if position is not None:
                        locals_[name] = (position, joint.get_local_rotation())
                world = world_poses_from_sdk_locals(locals_, ALL_PARENTS)
                if not set(BODY_PARENTS).issubset(world):
                    return
                submit({"seq": self.sequence, "source_time_s": now-self.origin_s,
                        "received_monotonic_s": now, "poses": world})
                self.sequence += 1
            except (ValueError, RuntimeError) as exc:
                print(f"PN frame rejected: {exc}", file=sys.stderr)

        def dispatch_key(self, key):
            if key in ("a", "x"):
                on_key(key)
            else:
                super().dispatch_key(key)

    return PNLiveSource()


class ROSPublisher:
    def __init__(self, publish_config, joint_names, reference_topic=None, source_viz=True):
        import rclpy
        from rclpy.node import Node
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float32MultiArray, Bool
        from geometry_msgs.msg import TransformStamped, PoseArray, Pose, Point
        from visualization_msgs.msg import Marker
        from tf2_ros import TransformBroadcaster
        self.rclpy, self.JointState, self.Reference = rclpy, JointState, Float32MultiArray
        self.TransformStamped, self.Marker, self.Point = TransformStamped, Marker, Point
        self.PoseArray, self.Pose, self.Bool = PoseArray, Pose, Bool
        rclpy.init(args=[])
        self.node = Node("mocap_rp1_gmr_reference")
        self.executor = SingleThreadedExecutor(context=self.node.context)
        self.executor.add_node(self.node)
        self.joint_names = joint_names
        ref, hands = publish_config["reference"], publish_config.get("hands", {})
        def qos(section):
            return QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=int(section["qos_depth"]),
                              reliability=(ReliabilityPolicy.RELIABLE if section["reliability"] == "reliable"
                                           else ReliabilityPolicy.BEST_EFFORT), durability=DurabilityPolicy.VOLATILE)
        self.reference_topic = reference_topic or ref["topic"]
        self.publisher = self.node.create_publisher(Float32MultiArray, self.reference_topic, qos(ref))
        self.joint_publisher = self.node.create_publisher(JointState, "/joint_states", 10)
        self.tf = TransformBroadcaster(self.node)
        self.raw_publisher = self.node.create_publisher(Marker, "/raw_skeleton", 10) if source_viz else None
        self.hands, self.hand_publishers = hands, {}
        if hands.get("enabled", False):
            for side in ("left", "right"):
                self.hand_publishers[side] = (
                    self.node.create_publisher(PoseArray, hands[f"{side}_poses"], qos(hands)),
                    self.node.create_publisher(Bool, hands[f"{side}_active"], qos(hands)),
                )

    def publish(self, vector, source=None):
        stamp = self.node.get_clock().now().to_msg()
        message = self.Reference()
        message.data = vector
        self.publisher.publish(message)
        joints = self.JointState()
        joints.header.stamp, joints.name, joints.position = stamp, self.joint_names, vector[7:]
        self.joint_publisher.publish(joints)
        transform = self.TransformStamped()
        transform.header.stamp, transform.header.frame_id, transform.child_frame_id = stamp, "world", "base_link"
        transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = vector[:3]
        transform.transform.rotation.w, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z = vector[3:7]
        transforms = [transform]
        poses = source["poses"] if source else {}
        if self.raw_publisher and poses:
            marker = self.Marker()
            marker.header.stamp, marker.header.frame_id = stamp, "world"
            marker.ns, marker.id, marker.type, marker.action = "raw_skeleton", 0, self.Marker.LINE_LIST, self.Marker.ADD
            marker.scale.x, marker.color.r, marker.color.g, marker.color.b = .02, .1, .9, .3
            marker.color.a, marker.pose.orientation.w = 1., 1.
            for name, parent in BODY_PARENTS.items():
                if name not in poses or (parent != "world" and parent not in poses):
                    continue
                p, q = poses[name]
                if parent != "world":
                    parent_p, parent_q = poses[parent]
                    parent_q = normalize_quat(parent_q)
                    inverse = [parent_q[0], -parent_q[1], -parent_q[2], -parent_q[3]]
                    p = rotate_vector(inverse, [a-b for a, b in zip(p, parent_p)])
                    q = quat_multiply(inverse, q)
                q = normalize_quat(q)
                raw = self.TransformStamped()
                raw.header.stamp, raw.header.frame_id = stamp, "world" if parent == "world" else f"raw_{parent}"
                raw.child_frame_id = f"raw_{name}"
                raw.transform.translation.x, raw.transform.translation.y, raw.transform.translation.z = p
                raw.transform.rotation.w, raw.transform.rotation.x, raw.transform.rotation.y, raw.transform.rotation.z = q
                transforms.append(raw)
                if parent in poses:
                    for p in (poses[parent][0], poses[name][0]):
                        marker.points.append(self.Point(x=p[0], y=p[1], z=p[2]))
            self.raw_publisher.publish(marker)
        self.tf.sendTransform(transforms)
        for side, (pose_pub, active_pub) in self.hand_publishers.items():
            message = self.PoseArray()
            message.header.stamp, message.header.frame_id = stamp, self.hands.get("frame_id", "world")
            for name in HAND_ORDERS[side]:
                if name not in poses:
                    continue
                p, q = poses[name]
                pose = self.Pose()
                pose.position.x, pose.position.y, pose.position.z = p
                pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z = q
                message.poses.append(pose)
            active = self.Bool()
            active.data = bool(message.poses)
            active_pub.publish(active)
            pose_pub.publish(message)
        self.executor.spin_once(timeout_sec=0.)

    def close(self, progress=None):
        started = time.monotonic()
        stages = {}
        def mark(name):
            if progress:
                progress({"status": "running", "stage": name, "elapsed_s": time.monotonic()-started,
                          "completed_stages_s": dict(stages)})
            return time.monotonic()
        before = mark("executor_shutdown")
        executor_drained = self.executor.shutdown(timeout_sec=.25)
        stages["executor_shutdown"] = time.monotonic()-before
        before = mark("node_destroy")
        self.executor.remove_node(self.node)
        self.node.destroy_node()
        stages["node_destroy"] = time.monotonic()-before
        before = mark("context_shutdown")
        if self.rclpy.ok():
            self.rclpy.try_shutdown()
        stages["context_shutdown"] = time.monotonic()-before
        return {"status": "complete", "elapsed_s": time.monotonic()-started,
                "executor_drained": executor_drained, "completed_stages_s": stages}

    def reference_contract(self):
        """Introspect the live publisher instead of relying on RMW discovery."""
        qos = self.publisher.qos_profile
        message_type = self.publisher.msg_type
        return {"topic": self.publisher.topic_name,
                "message_type": f"{message_type.__module__.split('.')[0]}/msg/{message_type.__name__}",
                "history": qos.history.name, "depth": qos.depth,
                "reliability": qos.reliability.name, "durability": qos.durability.name}


def write_metrics_atomic(path, metrics):
    if path:
        path = Path(path)
        temporary = path.with_suffix(path.suffix+".tmp")
        temporary.write_text(json.dumps(metrics, indent=2)+"\n")
        temporary.replace(path)


def start_cleanup_watchdog(metrics_path, timeout_s, field):
    # argv is passed directly to Python, never interpreted by a shell.
    watchdog_code = '''import json,os,signal,sys,time
from pathlib import Path
parent=int(sys.argv[1]);deadline=float(sys.argv[2]);path=sys.argv[3];field=sys.argv[4]
time.sleep(deadline)
if os.getppid()!=parent: sys.exit(0)
if path:
 p=Path(path)
 try:
  data=json.loads(p.read_text());last=data.get(field,{})
  data[field]={**last,"status":"timed_out","deadline_s":deadline}
  temp=p.with_suffix(p.suffix+".watchdog.tmp");temp.write_text(json.dumps(data,indent=2)+"\\n");temp.replace(p)
 except Exception as error:
  print("Could not persist cleanup timeout:",error,file=sys.stderr,flush=True)
print(field+" deadline exceeded; saved metrics, terminating bridge",file=sys.stderr,flush=True)
try: os.kill(parent,signal.SIGKILL)
except ProcessLookupError: pass
'''
    return subprocess.Popen([sys.executable, "-c", watchdog_code, str(os.getpid()),
                             str(timeout_s), str(metrics_path) if metrics_path else "", field],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)


def stop_cleanup_watchdog(watchdog):
    if watchdog.poll() is None:
        watchdog.terminate()
    try:
        watchdog.wait(timeout=.5)
    except subprocess.TimeoutExpired:
        watchdog.kill()
        watchdog.wait(timeout=.5)


def close_ros_bounded(ros, metrics, metrics_path, timeout_s=3.):
    """Bound native RMW teardown independently of Python's GIL."""
    metrics["ros_cleanup"] = {"status": "running", "stage": "start", "deadline_s": timeout_s}
    write_metrics_atomic(metrics_path, metrics)
    watchdog = start_cleanup_watchdog(metrics_path, timeout_s, "ros_cleanup")
    def progress(status):
        metrics["ros_cleanup"] = {**status, "deadline_s": timeout_s}
        write_metrics_atomic(metrics_path, metrics)
    try:
        metrics["ros_cleanup"] = ros.close(progress)
    except Exception as exc:
        metrics["ros_cleanup"] = {"status": "failed", "error": str(exc) or type(exc).__name__, "deadline_s": timeout_s}
        print(f"ROS cleanup failed: {exc}", file=sys.stderr)
    finally:
        stop_cleanup_watchdog(watchdog)
    write_metrics_atomic(metrics_path, metrics)


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered)-1)*fraction
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower]+(index-lower)*(ordered[upper]-ordered[lower])


async def run_bridge(args):
    global _FINAL_CLEANUP_WATCHDOG
    network = None
    if args.source == "live":
        network = preflight_pn_network(PNNetworkConfig(
            args.pn_local_ip, args.pn_local_port, args.pn_server_ip, args.pn_server_port))
        print(f"PN UDP: local={network.local_ip}:{network.local_port}, "
              f"PN-Link={network.server_ip}:{network.server_port}", file=sys.stderr)
    config = load_publish_config(args.publish_config)
    joint_config = json.loads(Path(args.joint_config).read_text())
    joint_names = joint_config["urdfJointNames"]
    if len(joint_names) != 28 or len(set(joint_names)) != 28:
        raise ValueError("joint config must contain 28 unique urdfJointNames")
    gate = ReferenceGate(float(config["reference"]["source_timeout_ms"])/1000.,
                         float(config["reference"]["transition_steps"])/50.,
                         root_z_offset_m=config["reference"]["root_z_offset_m"])
    latencies, roundtrips, intervals, solve_ms = [], [], [], []
    stop = asyncio.Event()
    latest_source = None
    autostart_pending = args.autostart
    def accept(result):
        nonlocal autostart_pending
        gate.accept(result)
        roundtrips.append(result["worker_roundtrip_ms"])
        value = result.get("stats", {}).get("solver_ms")
        if isinstance(value, (float, int)):
            solve_ms.append(value)
        if autostart_pending and gate.activate(time.monotonic()):
            autostart_pending = False
    command = [args.solver_python, "-u", args.worker_script, "worker", "--config", args.calibration]
    def on_ready(metadata):
        if metadata.get("joint_names") != list(joint_names):
            raise ValueError("worker joint order does not match the deployed 28-joint reference order")
        print(f"RP1 reference ready: fixed live root Z offset {gate.root_z_offset_m:+g} m, "
              "alignment/transition only; N capture, A follow, X stand", file=sys.stderr)
    worker = SolverWorker(command, accept, args.worker_timeout_s, args.worker_startup_timeout_s, on_ready)
    def submit(frame):
        nonlocal latest_source
        latest_source = frame
        worker.mailbox.offer(frame)
    def key(name):
        if name == "a":
            print("A: aligned/blending to live" if gate.activate(time.monotonic()) else "A ignored: no fresh solution; press N and wait", file=sys.stderr)
        elif name == "x":
            gate.idle(time.monotonic())
            print("X: blending to stand; x/y/yaw held", file=sys.stderr)
    for output_path in (args.output, args.metrics):
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    source, ros, ros_contract, output = None, None, None, None
    tasks = []
    terminal_state, fd = None, None
    started = time.monotonic()
    source_started_s = None
    published = 0
    previous_tick = None
    reference_fault = None
    deadline_lateness_ms = []
    process_cleanup = {"status": "pending"}
    process_watchdog = None
    ended = None
    viewer = IntegratedViser(build_viser_command(args)) if getattr(args, "viser", False) else None
    def distribution(values):
        return {"count": len(values), "p50": percentile(values, .5), "p95": percentile(values, .95),
                "p99": percentile(values, .99), "max": max(values) if values else None}
    def build_metrics():
        return {"source": args.source, "dry_run": args.dry_run, "duration_s": ended-started,
                "reference_pipeline": REFERENCE_PIPELINE, "final_constraints_enabled": False,
                "root_z_offset_m": gate.root_z_offset_m,
                "reference_stages": ["fixed_root_z_offset", "initial_planar_alignment", "activation_stand_transition"],
                "transition_duration_s": gate.blend_s,
                "source_duration_s": ended-source_started_s if source_started_s is not None else None,
                "worker_startup_ms": distribution(worker.startup_ms),
                "worker_metadata": {k: v for k, v in (worker.metadata or {}).items() if k != "kinematic_model_xml"},
                "publish_deadline_lateness_ms": distribution(deadline_lateness_ms),
                "late_publish_intervals_over_21ms": sum(v > 21. for v in intervals),
                "reference_fault": reference_fault, "process_cleanup": process_cleanup,
                "ros_reference_publisher": ros_contract,
                "viser": dict(viewer.status) if viewer else {"enabled": False},
                "shutdown_duration_s": time.monotonic()-ended, "published": published,
                "publish_hz": (published-1)/(sum(intervals)/1000.) if intervals else 0.,
                "source_frames": worker.mailbox.offered, "mailbox_replaced": worker.mailbox.replaced,
                "solved": worker.solved, "worker_errors": worker.errors, "worker_restarts": worker.restarts,
                "last_worker_error": worker.last_error, "stale_fallbacks": gate.stale_count,
                "publish_interval_ms": distribution(intervals), "source_to_publish_ms": distribution(latencies),
                "worker_roundtrip_ms": distribution(roundtrips), "solver_ms": distribution(solve_ms),
                "source_frames_skipped": getattr(source, "skipped", None) if args.source == "replay" else None}
    try:
        output = Path(args.output).open("w") if args.output else None
        source = ReplaySource(args.npz, args.replay_speed) if args.source == "replay" else create_live_source(submit, key, network)
        ros = None if args.dry_run else ROSPublisher(config, joint_names, args.reference_topic, args.source_viz)
        ros_contract = ros.reference_contract() if ros else None
        loop = asyncio.get_running_loop()
        if viewer is not None:
            try:
                await viewer.start()
                print(f"Viser reference display: http://{args.viser_host}:{args.viser_port}/ "
                      "(published reference; read-only)", file=sys.stderr)
            except (OSError, RuntimeError) as exc:
                viewer.status["failure"] = str(exc) or type(exc).__name__
                print(f"Viser could not start: {viewer.status['failure']}; "
                      "reference publication continues", file=sys.stderr)
        if sys.stdin.isatty():
            fd = sys.stdin.fileno()
            terminal_state = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            def read_keys():
                data = os.read(fd, 16)
                if not data or data == b"\x1b":
                    stop.set()
                    return
                if data.startswith(b"\x1b"):
                    return
                for c in data:
                    name = chr(c).lower()
                    source.dispatch_key(name) if args.source == "live" else key(name)
            loop.add_reader(fd, read_keys)
        print(f"PN→RP1 GMR bridge: source={args.source}, 50 Hz, "
              f"reference={REFERENCE_PIPELINE}, "
              f"live_root_z_offset={gate.root_z_offset_m:+g} m, "
              f"topic={'dry-run' if args.dry_run else ros.reference_topic}; N capture, A follow, X stand, ESC quit", file=sys.stderr)
        tasks.append(asyncio.create_task(worker.run()))
        async def run_source():
            nonlocal source_started_s
            if args.source == "replay":
                # Preserve the clip's beginning while IK loads the model;
                # stand still publishes at 50 Hz during the warmup.
                await asyncio.wait_for(worker.ready.wait(), args.worker_startup_timeout_s)
            source_started_s = time.monotonic()
            if args.source == "replay":
                await source.run(submit, stop)
            else:
                await source.update()
        source_task = asyncio.create_task(run_source())
        tasks.append(source_task)
        next_tick = started
        while not stop.is_set():
            now = time.monotonic()
            if args.max_duration_s and source_started_s is not None and now-source_started_s >= args.max_duration_s:
                break
            if args.source == "replay" and source_task.done():
                try:
                    source_task.result()
                except (RuntimeError, ValueError, OSError, asyncio.TimeoutError) as exc:
                    reference_fault = f"source/worker initialization failed: {str(exc) or type(exc).__name__}"
                    print(f"Reference stopped: {reference_fault}", file=sys.stderr)
                    break
                if now > source_started_s+float(source.times[-1])/source.speed+args.replay_tail_s:
                    break
            if now < next_tick:
                await asyncio.sleep(min(next_tick-now, .02))
                continue
            now = time.monotonic()
            # This is the entire post-retarget path: evaluate the current
            # fixed live root-Z offset, SE(2) alignment and mode-transition blend
            # at publication time. No final optimizer, limiter or contact handling.
            vector = gate.proposal(now)
            output_source = gate.latest
            if previous_tick is not None:
                intervals.append((now-previous_tick)*1000.)
            deadline_lateness_ms.append(max(0., now-next_tick)*1000.)
            previous_tick = now
            gate.synchronize(vector)
            fresh_source = latest_source if latest_source and now-latest_source["received_monotonic_s"] <= gate.timeout_s else None
            if ros:
                ros.publish(vector, fresh_source)
            if viewer is not None and viewer.status["started"]:
                viewer.offer(vector, source=fresh_source,
                             reference_source_time_s=output_source["source_time_s"] if output_source else None)
            age_ms = (now-output_source["received_monotonic_s"])*1000. if output_source else None
            if gate.mode == "live" and age_ms is not None:
                latencies.append(age_ms)
            if output:
                output.write(json.dumps({"time_s": now-started, "seq": output_source["seq"] if output_source else None,
                                         "source_time_s": output_source["source_time_s"] if output_source else None,
                                         "mode": gate.mode, "source_age_ms": age_ms,
                                         "alignment_se2": list(gate.origin) if gate.origin is not None else None,
                                         "blend_active": gate.blend_from is not None,
                                         "worker_roundtrip_ms": output_source["worker_roundtrip_ms"] if output_source else None,
                                         "solver_stats": output_source.get("stats", {}) if output_source else {},
                                         "reference_pipeline": REFERENCE_PIPELINE,
                                         "final_constraints_enabled": False,
                                         "root_z_offset_m": gate.root_z_offset_m,
                                         "reference_stages": ["fixed_root_z_offset", "initial_planar_alignment", "activation_stand_transition"],
                                         "solver_qpos": output_source["qpos"] if output_source else None,
                                         "desired_qpos": vector,
                                         "qpos": vector}, allow_nan=False)+"\n")
            published += 1
            next_tick += .02
            if next_tick <= now:
                next_tick = now+.02
    finally:
        ended = time.monotonic()
        stop.set()
        if output:
            output.flush()
            output.close()
        process_cleanup = {"status": "running", "stage": "cancel_background_tasks"}
        checkpoint = build_metrics()
        write_metrics_atomic(args.metrics, checkpoint)
        process_watchdog = start_cleanup_watchdog(args.metrics, 3., "process_cleanup")
        _FINAL_CLEANUP_WATCHDOG = process_watchdog
        if terminal_state is not None:
            asyncio.get_running_loop().remove_reader(fd)
            termios.tcsetattr(fd, termios.TCSADRAIN, terminal_state)
        captured_processes = [worker.process] if worker.process is not None else []
        for task in tasks:
            task.cancel()
        _, pending = await asyncio.wait(tasks, timeout=1.) if tasks else (set(), set())
        if pending:
            for process in captured_processes:
                if process.returncode is None:
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
                transport = getattr(process, "_transport", None)
                if transport is not None:
                    transport.close()
            for task in pending:
                task.cancel()
            _, pending = await asyncio.wait(pending, timeout=.5)
        if viewer is not None:
            await viewer.close()
        if args.source == "live" and source is not None:
            source.app.close()
        process_cleanup = {"status": "timed_out" if pending else "complete",
                           "pending_tasks": len(pending), "elapsed_s": time.monotonic()-ended}
        if sys.exc_info()[0] is not None:
            if ros is not None:
                close_ros_bounded(ros, build_metrics(), args.metrics)
            stop_cleanup_watchdog(process_watchdog)
            _FINAL_CLEANUP_WATCHDOG = None
    metrics = build_metrics()
    metrics["ros_cleanup"] = {"status": "not_applicable"} if ros is None else {"status": "pending"}
    write_metrics_atomic(args.metrics, metrics)
    if ros:
        close_ros_bounded(ros, metrics, args.metrics)
    print(json.dumps(metrics, indent=2))
    return metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("live", "replay"), default="live")
    parser.add_argument("--npz", type=Path)
    parser.add_argument("--pn-local-ip", help="override pn_network.local_ip in publish.yaml (PC address)")
    parser.add_argument("--pn-local-port", type=int, help="override pn_network.local_port (UDP receive)")
    parser.add_argument("--pn-server-ip", help="override pn_network.server_ip (PN-Link device address)")
    parser.add_argument("--pn-server-port", type=int, help="override pn_network.server_port (UDP command)")
    parser.add_argument("--dry-run", action="store_true", help="no ROS publisher; replay does not import/open PN")
    parser.add_argument("--reference-topic", help="override publish.yaml reference topic")
    source_viz = parser.add_mutually_exclusive_group()
    source_viz.add_argument("--source-viz", dest="source_viz", action="store_true",
                            help="publish raw 21-joint skeleton and raw_* TF for RViz (default)")
    source_viz.add_argument("--no-source-viz", dest="source_viz", action="store_false",
                            help="disable ROS raw skeleton/TF; Viser display remains independent")
    parser.set_defaults(source_viz=True)
    viser = parser.add_mutually_exclusive_group()
    viser.add_argument("--viser", dest="viser", action="store_true",
                       help="display the published reference in Viser (default for live publishing)")
    viser.add_argument("--no-viser", dest="viser", action="store_false",
                       help="disable the integrated browser model")
    parser.set_defaults(viser=None)
    parser.add_argument("--viser-host", default="127.0.0.1", help="Viser bind address")
    parser.add_argument("--viser-port", type=int, default=8891, help="Viser HTTP port (default: 8891)")
    parser.add_argument("--viser-python", type=Path, default=DEFAULT_VISER_PYTHON,
                        help="Python runtime containing Viser, NumPy and MuJoCo")
    parser.add_argument("--autostart", action="store_true", help="activate first fresh result; replay validation convenience")
    parser.add_argument("--publish-config", type=Path, default=DEFAULT_PUBLISH_CONFIG)
    parser.add_argument("--joint-config", type=Path, default=DEFAULT_JOINT_CONFIG)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--solver-python", type=Path, default=DEFAULT_SOLVER_PYTHON)
    parser.add_argument("--worker-script", type=Path, default=DEFAULT_WORKER_SCRIPT)
    parser.add_argument("--worker-timeout-s", type=float, default=.5)
    parser.add_argument("--worker-startup-timeout-s", type=float, default=30.)
    parser.add_argument("--replay-speed", type=float, default=1.)
    parser.add_argument("--replay-tail-s", type=float, default=3.5,
                        help="continue through source timeout and the stand blend after the final frame")
    parser.add_argument("--max-duration-s", type=float, default=0.,
                        help="stop after this source duration, excluding replay model startup; 0 = full take")
    parser.add_argument("--output", type=Path, help="optional published vectors JSONL")
    parser.add_argument("--metrics", type=Path, help="optional timing summary JSON")
    args = parser.parse_args(argv)
    try:
        overrides = {key: getattr(args, 'pn_' + key) for key in
                     ('local_ip', 'local_port', 'server_ip', 'server_port')}
        if args.source == 'live':
            network = load_pn_network(args.publish_config, overrides=overrides)
        else:
            # Offline playback does not depend on a configured PN interface.
            defaults = PNNetworkConfig()
            network = validate_pn_network(PNNetworkConfig(**{
                key: value if value is not None else getattr(defaults, key)
                for key, value in overrides.items()}))
        for key in overrides:
            setattr(args, 'pn_' + key, getattr(network, key))
    except (PNNetworkError, OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.viser is None:
        args.viser = args.source == "live" and not args.dry_run
    if args.source == "replay" and not args.npz:
        parser.error("--source replay requires --npz")
    if args.reference_topic and not args.reference_topic.startswith("/"):
        parser.error("--reference-topic must be absolute")
    if not 1 <= args.viser_port <= 65535:
        parser.error("--viser-port must be between 1 and 65535")
    if not args.viser_host.strip():
        parser.error("--viser-host must not be empty")
    for name in ("worker_timeout_s", "worker_startup_timeout_s", "replay_speed"):
        if getattr(args, name) is not None and (not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0):
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    for name in ("max_duration_s", "replay_tail_s"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative and finite")
    return args


if __name__ == "__main__":
    try:
        result = asyncio.run(run_bridge(parse_args()))
        if _FINAL_CLEANUP_WATCHDOG is not None:
            stop_cleanup_watchdog(_FINAL_CLEANUP_WATCHDOG)
            _FINAL_CLEANUP_WATCHDOG = None
        if (result.get("reference_fault") or result.get("ros_cleanup", {}).get("status") == "failed"
                or result.get("process_cleanup", {}).get("status") != "complete"):
            raise SystemExit(2)
    except KeyboardInterrupt:
        pass
    except PNNetworkError as exc:
        print(f"PN network configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2)
