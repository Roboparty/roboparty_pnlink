#!/usr/bin/env python3
"""Generate a synthetic standard PN skeleton with rigid XY/yaw motion.

This is a format/installation example, not a captured person or walking gait.
No SDK, ROS, robot model or network access is needed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/demo_pn.npz')
    parser.add_argument('--duration-s', type=float, default=4.0)
    parser.add_argument('--rate-hz', type=float, default=50.0)
    args = parser.parse_args(argv)
    import math
    for field in ('duration_s', 'rate_hz'):
        value = getattr(args, field)
        if not math.isfinite(value) or value <= 0:
            parser.error(f'--{field.replace("_", "-")} must be finite and positive')
    frames = int(round(args.duration_s * args.rate_hz)) + 1
    if not 2 <= frames <= 1_000_000:
        parser.error('choose a duration/rate producing 2 through 1000000 frames')
    return args, frames


def main(argv=None):
    args, frames = parse_args(argv)
    import numpy as np

    sys.path.insert(0, str(ROOT / 'mocapapi_python'))
    from pn_rp1_calibration import PN_JOINT_NAMES, standard_pn_tpose

    pose = standard_pn_tpose()
    names = list(PN_JOINT_NAMES)
    standard = np.asarray([pose[name][0] for name in names], dtype=float)
    hips = standard[names.index('Hips')]
    local = standard - hips
    times = np.arange(frames, dtype=float) / args.rate_hz
    phase = 2.0 * np.pi * times / args.duration_s
    x = 0.12 * np.sin(phase)
    y = 0.06 * (1.0 - np.cos(phase))
    yaw = 0.25 * np.sin(phase)
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    positions = np.empty((frames, len(names), 3), dtype=float)
    positions[:, :, 0] = hips[0] + x[:, None] + c * local[:, 0] - s * local[:, 1]
    positions[:, :, 1] = hips[1] + y[:, None] + s * local[:, 0] + c * local[:, 1]
    positions[:, :, 2] = standard[:, 2]
    rotations = np.zeros((frames, len(names), 4), dtype=float)
    rotations[:, :, 2] = np.sin(yaw / 2.0)[:, None]
    rotations[:, :, 3] = np.cos(yaw / 2.0)[:, None]
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # A stream preserves the exact requested filename rather than silently
    # appending an extension; all arrays load with allow_pickle=False.
    with output.open('wb') as stream:
        np.savez_compressed(stream, timestamps_s=times,
                            joint_names=np.asarray(names),
                            global_position_ros_m=positions,
                            global_rotation_ros_xyzw=rotations)
    print(f'{output}: {frames} synthetic frames, {times[-1]:g} s, {args.rate_hz:g} Hz; no captured human data')


if __name__ == '__main__':
    main()
