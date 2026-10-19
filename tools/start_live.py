#!/usr/bin/env python3
"""Start the bundled PN→RP1 bridge after dependencies and PN IPs are configured.

Source ROS and load .env explicitly before running this helper. PN endpoints
come from explicit CLI options, then publish.yaml, compatible environment
variables, and the SDK defaults. It never changes network interfaces or
automatically enters follow.
"""
from __future__ import annotations

import argparse
import ipaddress
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'mocapapi_python'


def absolute_runtime(value):
    # Keep a venv python symlink intact rather than resolving it to its base.
    return Path(os.path.abspath(Path(value).expanduser()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pn-local-ip', help='override the PC PN interface address in publish.yaml')
    parser.add_argument('--pn-server-ip', help='override the PN-Link device address in publish.yaml')
    parser.add_argument('--pn-local-port')
    parser.add_argument('--pn-server-port')
    parser.add_argument('--publish-config', type=Path,
                        default=SOURCE / 'rp1/publish.yaml', help='publish and PN network YAML')
    parser.add_argument('--ros-python', type=Path, default=os.environ.get('PN_RP1_ROS_PYTHON'))
    parser.add_argument('--solver-python', type=Path)
    parser.add_argument('--viser-python', type=Path)
    parser.add_argument('--calibration', type=Path,
                        default=SOURCE / 'rp1/pn_axis_to_rp1_28dof_current.json')
    parser.add_argument('--print-command', action='store_true',
                        help='print the selected command without importing SDK/ROS or opening PN')
    args, bridge_arguments = parser.parse_known_args(argv)
    if '--source' in bridge_arguments or any(value.startswith('--source=') for value in bridge_arguments):
        parser.error('this helper starts live input; use mocap_rp1_gmr_reference.py directly for replay')
    sys.path.insert(0, str(SOURCE))
    from pn_network import PNNetworkError, load_pn_network
    from pn_rp1_paths import resolve_python_runtime
    publish_config = args.publish_config.expanduser().absolute()
    overrides = {name: getattr(args, 'pn_' + name)
                 for name in ('local_ip', 'local_port', 'server_ip', 'server_port')
                 if getattr(args, 'pn_' + name) is not None}
    try:
        network = load_pn_network(publish_config, overrides=overrides, environ=os.environ)
    except (OSError, ValueError, PNNetworkError) as exc:
        parser.error(str(exc))
    examples = [ipaddress.ip_network(value) for value in
                ('192.0.2.0/24', '198.51.100.0/24', '203.0.113.0/24')]
    if any(ipaddress.ip_address(value) in block for value in
           (network.local_ip, network.server_ip) for block in examples):
        parser.error('replace the documentation-only PN IP addresses with the actual PC and PN-Link addresses')
    calibration = args.calibration.expanduser().absolute()
    if not calibration.is_file():
        parser.error(f'calibration not found: {calibration}; restore the bundled file or regenerate with tools/prepare_calibration.py')
    ros_default = ROOT / '.venv-ros/bin/python'
    ros_python = absolute_runtime(args.ros_python or (ros_default if ros_default.is_file() else sys.executable))
    solver_python = absolute_runtime(args.solver_python or resolve_python_runtime('solver'))
    viser_python = absolute_runtime(args.viser_python or resolve_python_runtime('viser'))
    viser_enabled = True
    for argument in bridge_arguments:
        if argument in ('--viser', '--no-viser'):
            viser_enabled = argument == '--viser'
    for name, path in (('ROS', ros_python), ('solver', solver_python), ('Viser', viser_python)):
        if name == 'Viser' and not viser_enabled:
            continue
        if not path.is_file():
            parser.error(f'{name} Python not found: {path}; follow docs/INSTALLATION.md')
    command = [str(ros_python), str(SOURCE / 'mocap_rp1_gmr_reference.py'),
               '--source', 'live', '--calibration', str(calibration),
               '--publish-config', str(publish_config),
               '--pn-local-ip', network.local_ip, '--pn-local-port', str(network.local_port),
               '--pn-server-ip', network.server_ip, '--pn-server-port', str(network.server_port),
               '--solver-python', str(solver_python), '--viser-python', str(viser_python),
               *bridge_arguments]
    print(shlex.join(command), flush=True)
    if args.print_command:
        return 0
    try:
        os.execve(str(ros_python), command, os.environ.copy())
    except OSError as exc:
        parser.exit(2, f'Cannot start the bridge: {exc}\n')


if __name__ == '__main__':
    sys.exit(main())
