#!/usr/bin/env python3
"""Check PN/RP1 runtimes without opening PN or creating a ROS node.

The solver probe loads the supplied model/calibration and solves one analytical
standard-T frame. SDK/ROS probes only load libraries and interfaces. No device
application, socket, publisher, viewer server or physics simulation is started.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'mocapapi_python'
MARKER = 'PN_RP1_ENV_CHECK='


def probe(python, code, timeout=45):
    """Isolate native/import failures; never construct a device application."""
    prefix = ('import sys, json\nsys.dont_write_bytecode = True\n'
              f'sys.path.insert(0, {str(SOURCE)!r})\n')
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    try:
        result = subprocess.run([str(Path(python).expanduser()), '-c', prefix + code], env=environment,
                                cwd=ROOT, text=True, capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if result.returncode:
        details = (result.stderr.strip() or result.stdout.strip() or
                   f'process returned {result.returncode}')
        return False, details[-6000:]
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(MARKER):
            try:
                return True, json.loads(line[len(MARKER):])
            except ValueError:
                break
    return False, 'probe did not return its structured result'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('offline', 'live'), default='live')
    parser.add_argument('--model', type=Path, help='RP1.3.2 MJCF; otherwise PN_RP1_MODEL/config path')
    parser.add_argument('--calibration', type=Path,
                        default=SOURCE / 'rp1/pn_axis_to_rp1_28dof_current.json')
    parser.add_argument('--solver-python', type=Path, help='otherwise PN_RP1_SOLVER_PYTHON/current runtime')
    parser.add_argument('--viser-python', type=Path, help='otherwise PN_RP1_VISER_PYTHON/current runtime')
    parser.add_argument('--no-viser', action='store_true', help='skip optional browser-viewer runtime check')
    parser.add_argument('--skip-sdk-load', action='store_true',
                        help='check SDK files only, without loading the native library')
    parser.add_argument('--json-output', type=Path, help='optional local report, e.g. outputs/environment.json')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, str(SOURCE))
    from pn_rp1_paths import resolve_model_path, resolve_python_runtime
    from mocap_rp1_gmr_reference import load_publish_config

    checks = []

    def add(name, ok, details):
        checks.append({'name': name, 'ok': bool(ok), 'details': details})
        print(f'{"PASS" if ok else "FAIL"} {name}: '
              f'{json.dumps(details, ensure_ascii=False) if isinstance(details, dict) else details}', flush=True)

    machine = platform.machine()
    supported = platform.system() == 'Linux' and machine in ('x86_64', 'aarch64', 'arm64')
    add('platform', supported, {'os': platform.system(), 'architecture': machine,
                               'python': platform.python_version(),
                               'arm_runtime_tested': False if machine in ('aarch64', 'arm64') else None})
    add('python', sys.version_info >= (3, 10), 'Python 3.10+ required; ROS ABI must match the installed ROS distribution.')
    try:
        publish = load_publish_config(SOURCE / 'rp1/publish.yaml')['reference']
        add('reference_config', True, {key: publish[key] for key in
            ('topic', 'rate_hz', 'transition_steps', 'root_z_offset_m')})
    except (OSError, ValueError, KeyError) as exc:
        add('reference_config', False, str(exc))

    calibration_path = args.calibration.expanduser().resolve()
    calibration = None
    try:
        calibration = json.loads(calibration_path.read_text())
        calibration['source_calibration']['robot_model_path']
        add('calibration', True, {'path': str(calibration_path),
            'scale': calibration['source_calibration']['root_position_scale'],
            'joint_acceleration_enabled': calibration.get('output_limits', {}).get('joint_acceleration_enabled', True),
            'contact_enabled': calibration.get('contact', {}).get('enabled', True),
            'pitch_compensation_enabled': calibration.get('torso_compensation', {}).get('enabled', True)})
    except (OSError, ValueError, KeyError, TypeError) as exc:
        add('calibration', False, f'{exc}; generate with tools/prepare_calibration.py after configuring PN_RP1_MODEL.')
        calibration = None
    model_path = None
    try:
        configured = calibration['source_calibration']['robot_model_path'] if calibration else None
        model_path = resolve_model_path(args.model, configured_path=configured,
                                        config_path=calibration_path)
        if calibration:
            actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
            if actual != calibration['source_calibration']['robot_model_sha256']:
                raise ValueError('model SHA differs from calibration; regenerate the local calibration')
        add('model_path', True, str(model_path))
    except (OSError, ValueError, KeyError) as exc:
        add('model_path', False, str(exc))

    solver_python = args.solver_python or resolve_python_runtime('solver')
    run_ik = calibration is not None and model_path is not None
    solver_code = '''
import inspect
import numpy as np
import scipy, mujoco, mink, qpsolvers, daqp
from pn_rp1_paths import resolve_gmr_root
gmr_root = resolve_gmr_root()
if gmr_root is not None:
    sys.path.insert(0, str(gmr_root))
import general_motion_retargeting.motion_retarget as gm
required = ('ROBOT_XML_DICT', 'IK_CONFIG_DICT', 'GeneralMotionRetargeting')
assert all(hasattr(gm, name) for name in required), 'incompatible GMR registry API'
assert 'use_velocity_limit' in inspect.signature(gm.GeneralMotionRetargeting).parameters, 'incompatible GMR constructor'
result = {'python': sys.executable, 'numpy': np.__version__, 'mujoco': mujoco.__version__,
          'gmr_module': gm.__file__, 'standard_tpose_ik_checked': False}
'''
    if run_ik:
        solver_code += f'''
from pn_rp1_retarget import PNToRP1Retargeter
solver = PNToRP1Retargeter({str(calibration_path)!r}, model_path={str(model_path)!r})
qpos = solver.retarget(solver.adapter.source_tpose, dt=.02, source_time_s=0.)
assert qpos.shape == (35,) and np.isfinite(qpos).all(), 'invalid reference vector'
assert solver.last_stats['qp_solves'] == 1, 'expected one QP per frame'
result.update(standard_tpose_ik_checked=True, reference_dimension=len(qpos),
              qp_solves=solver.last_stats['qp_solves'], floor_enabled=solver.config['solver']['floor_enabled'])
'''
    solver_code += f'print({MARKER!r} + json.dumps(result))\n'
    ok, details = probe(solver_python, solver_code)
    add('solver_runtime_and_one_frame_ik' if run_ik else 'solver_dependencies', ok, details)

    if args.mode == 'live':
        try:
            import pn_sdk
            sdk_dir = pn_sdk._sdk_directory()
            library = ('librobotapi_arm64.so' if machine in ('aarch64', 'arm64') else
                       'librobotapi_x86-64.so')
            sdk_files = [sdk_dir / name for name in ('mocap_api.py', 'mocap_main_base.py')]
            sdk_files.append(sdk_dir / 'lib' / library)
            missing = [str(path) for path in sdk_files if not path.is_file()]
            if missing:
                raise FileNotFoundError('missing SDK files: ' + ', '.join(missing))
            add('sdk_files', supported, {'directory': str(sdk_dir), 'library': library})
            if not args.skip_sdk_load and supported:
                sdk_code = '''
from pn_sdk import load_sdk
api, base = load_sdk()
assert hasattr(api.MocapApi, 'MCPGetGenericInterface'), 'SDK native entry point missing'
result = {'wrapper': api.__file__, 'base_class': base.__name__,
          'native_library': str(api.MocapApi._name), 'application_created': False}
'''
                sdk_code += f'print({MARKER!r} + json.dumps(result))\n'
                ok, details = probe(sys.executable, sdk_code)
                add('sdk_load_only', ok, details)
        except (OSError, RuntimeError) as exc:
            add('sdk_files', False, str(exc))
        ros_code = '''
import rclpy
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Float32MultiArray, Bool
from geometry_msgs.msg import TransformStamped, PoseArray, Pose, Point
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker
from tf2_ros import TransformBroadcaster
result = {'python': sys.executable, 'rclpy': rclpy.__file__, 'node_created': False,
          'reference_message': 'std_msgs/msg/Float32MultiArray'}
'''
        ros_code += f'print({MARKER!r} + json.dumps(result))\n'
        ok, details = probe(sys.executable, ros_code)
        add('ros_and_messages_import_only', ok, details)

    if not args.no_viser:
        viser_python = args.viser_python or resolve_python_runtime('viser')
        viewer_code = '''
import numpy, mujoco, viser, trimesh, open3d
result = {'python': sys.executable, 'mujoco': mujoco.__version__, 'server_created': False}
'''
        viewer_code += f'print({MARKER!r} + json.dumps(result))\n'
        ok, details = probe(viser_python, viewer_code)
        add('viser_runtime_import_only', ok, details)

    success = all(check['ok'] for check in checks)
    report = {'ok': success, 'mode': args.mode, 'checks': checks,
              'sdk_load_skipped': args.skip_sdk_load, 'pn_connection_opened': False,
              'ros_node_created': False, 'viewer_server_created': False,
              'scope': 'Dependency/model loading and one-frame kinematic IK only; no transport, hardware, or balance validation.'}
    if args.json_output:
        path = args.json_output.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('PASS: selected runtime checks completed.' if success else
          'FAIL: fix the failed checks before starting the live bridge.', flush=True)
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
