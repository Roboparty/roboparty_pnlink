#!/usr/bin/env python3
"""Install, configure, check and run the bundled PN-to-RP1 project.

The entry point itself uses only the standard library. Only ``start`` opens PN;
``demo`` uses synthetic data and never creates a ROS publisher.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ROS_SETUP = Path('/opt/ros/humble/setup.bash')
SYSTEM_PYTHON = Path('/usr/bin/python3')
NETWORK_KEYS = ('PN_LOCAL_IP', 'PN_SERVER_IP', 'PN_LOCAL_PORT', 'PN_SERVER_PORT', 'ROS_DOMAIN_ID')
SYSTEM_PACKAGES = (
    'python3-venv', 'python3-pip', 'python3-dev', 'build-essential',
    'libstdc++6', 'libgl1', 'libegl1', 'libglib2.0-0',
    'ros-humble-ros-base', 'ros-humble-rclpy', 'ros-humble-std-msgs',
    'ros-humble-sensor-msgs', 'ros-humble-geometry-msgs',
    'ros-humble-visualization-msgs', 'ros-humble-tf2-ros',
)
ASSIGNMENT = re.compile(r'^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$')
VARIABLE = re.compile(r'\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))')


def read_env(path, base=None):
    """Read literal KEY=value settings; never execute shell substitutions."""
    environment = dict(os.environ if base is None else base)
    environment['PWD'] = str(ROOT)
    path = Path(path)
    if not path.exists():
        return environment
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        match = ASSIGNMENT.fullmatch(line)
        if not match:
            raise ValueError(f'{path.name}:{number}: expected a literal KEY=value assignment')
        key, value = match.groups()
        parts = shlex.split(value, comments=True, posix=True)
        if len(parts) > 1:
            raise ValueError(f'{path.name}:{number}: quote values containing spaces')
        value = parts[0] if parts else ''
        environment[key] = VARIABLE.sub(
            lambda m: environment.get(m.group(1) or m.group(2), m.group(0)), value)
    return environment


def write_env(path, updates):
    """Update selected assignments atomically, preserving unrelated lines."""
    path = Path(path)
    lines = path.read_text().splitlines() if path.exists() else []
    remaining = dict(updates)
    result = []
    seen = set()
    for line in lines:
        match = ASSIGNMENT.fullmatch(line)
        key = match.group(1) if match else None
        if key in updates:
            if key not in seen:
                result.append(f'{key}={shlex.quote(str(updates[key]))}')
                seen.add(key)
                remaining.pop(key, None)
        else:
            result.append(line)
    result.extend(f'{key}={shlex.quote(str(value))}' for key, value in remaining.items())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text('\n'.join(result) + '\n')
    temporary.chmod(path.stat().st_mode & 0o777 if path.exists() else 0o600)
    temporary.replace(path)


def validate_settings(updates):
    for key, value in updates.items():
        if key.endswith('_IP'):
            address = ipaddress.IPv4Address(value)
            if address.is_unspecified or address.is_multicast or address.is_loopback or int(address) == 0xffffffff:
                raise ValueError(f'{key}: enter an actual unicast PN interface/device IPv4 address')
        elif key.endswith('_PORT'):
            if not 1 <= int(value) <= 65535:
                raise ValueError(f'{key}: port must be within 1..65535')
        elif key == 'ROS_DOMAIN_ID' and not 0 <= int(value) <= 232:
            raise ValueError('ROS_DOMAIN_ID must be within 0..232')


def explicit_settings(args):
    updates = {key: str(getattr(args, key.lower())) for key in NETWORK_KEYS
               if getattr(args, key.lower(), None) is not None}
    validate_settings(updates)
    return updates


def pn_addresses_ready(env):
    examples = [ipaddress.ip_network(value) for value in
                ('192.0.2.0/24', '198.51.100.0/24', '203.0.113.0/24')]
    try:
        values = {key: env[key] for key in NETWORK_KEYS if key.startswith('PN_')}
        validate_settings(values)
        return all(not any(ipaddress.ip_address(env[key]) in block for block in examples)
                   for key in ('PN_LOCAL_IP', 'PN_SERVER_IP'))
    except (KeyError, ValueError):
        return False


def show_interfaces():
    if not shutil.which('ip'):
        return
    result = subprocess.run(['ip', '-j', '-4', 'addr', 'show', 'up'],
                            text=True, capture_output=True, check=False)
    try:
        interfaces = json.loads(result.stdout)
        print('电脑当前网口地址（只列出，不修改）：', flush=True)
        for interface in interfaces:
            for address in interface.get('addr_info', []):
                if address.get('scope') == 'global':
                    print(f"  {interface['ifname']}: {address['local']}", flush=True)
    except (ValueError, KeyError, TypeError):
        pass


def configure(args, env):
    updates = explicit_settings(args)
    sys.path.insert(0, str(ROOT / 'mocapapi_python'))
    from pn_network import PNNetworkError, load_pn_network, write_pn_network_yaml
    path = (args.publish_config or ROOT / 'mocapapi_python/rp1/publish.yaml').expanduser().absolute()
    network_updates = {key.removeprefix('PN_').lower(): value for key, value in updates.items()
                       if key.startswith('PN_')}
    try:
        network = load_pn_network(path, overrides=network_updates, environ=env)
    except PNNetworkError as exc:
        raise ValueError(str(exc)) from exc
    defaults = dict(env, PN_LOCAL_IP=network.local_ip, PN_SERVER_IP=network.server_ip)
    if not args.non_interactive and sys.stdin.isatty():
        show_interfaces()
        for key, label in (('PN_LOCAL_IP', '电脑连接 PN 的网口 IP'),
                           ('PN_SERVER_IP', 'PN-Link 设备 IP'),
                           ('ROS_DOMAIN_ID', '跟踪端 ROS_DOMAIN_ID')):
            if key not in updates:
                value = input(f'{label} [{defaults.get(key, "")}，回车保留/稍后配置]: ').strip()
                if value:
                    updates[key] = value
    validate_settings(updates)
    network_updates = {key.removeprefix('PN_').lower(): value for key, value in updates.items()
                       if key.startswith('PN_')}
    try:
        network = load_pn_network(path, overrides=network_updates, environ=env)
        if network_updates:
            write_pn_network_yaml(path, network_updates)
    except PNNetworkError as exc:
        raise ValueError(str(exc)) from exc
    env_path = ROOT / '.env'
    if not env_path.exists():
        shutil.copyfile(ROOT / '.env.example', env_path)
        env_path.chmod(0o600)
    if 'ROS_DOMAIN_ID' in updates:
        write_env(env_path, {'ROS_DOMAIN_ID': updates['ROS_DOMAIN_ID']})
    result = read_env(env_path)
    result.update(PN_LOCAL_IP=network.local_ip, PN_SERVER_IP=network.server_ip,
                  PN_LOCAL_PORT=str(network.local_port), PN_SERVER_PORT=str(network.server_port))
    print(f'PN 地址和端口配置：{path} 的 pn_network；ROS_DOMAIN_ID 和运行环境路径：.env。', flush=True)
    if not pn_addresses_ready(result):
        print('PN 地址尚未填完：可以先运行 demo，连接设备前运行 ./run.sh configure。', flush=True)
    return result


def runtime(env, kind):
    key = 'PN_RP1_ROS_PYTHON' if kind == 'ros' else 'PN_RP1_SOLVER_PYTHON'
    default = ROOT / ('.venv-ros' if kind == 'ros' else '.venv-ik') / 'bin/python'
    path = Path(env.get(key) or default).expanduser()
    # Preserve the venv interpreter symlink rather than resolving its target.
    return Path(os.path.abspath(ROOT / path if not path.is_absolute() else path))


def viewer_runtime(env):
    path = Path(env.get('PN_RP1_VISER_PYTHON') or runtime(env, 'ik')).expanduser()
    return Path(os.path.abspath(ROOT / path if not path.is_absolute() else path))


def state_no_viser():
    try:
        return bool(json.loads((ROOT / '.runtime/setup.json').read_text())['no_viser'])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def use_viser(args):
    return not state_no_viser() if args.viser is None else args.viser


def command_line(command):
    print('+ ' + shlex.join(map(str, command)), flush=True)


def execute(command, env=None, print_plan=False):
    command = list(map(str, command))
    command_line(command)
    if not print_plan:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def load_ros_environment(env, required=False):
    if not ROS_SETUP.is_file():
        if required:
            raise ValueError('ROS 2 Humble 未安装。先按 docs/INSTALLATION.md 配置 ROS apt 源，再运行 ./run.sh setup --system-deps。')
        return env
    result = subprocess.run(
        ['bash', '-c', 'set +u; source "$1" >/dev/null && env -0', 'pn-rp1-ros', str(ROS_SETUP)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    loaded = dict(item.split('=', 1) for item in result.stdout.decode().split('\0') if '=' in item)
    loaded['PYTHONNOUSERSITE'] = '1'
    return loaded


def ensure_venv(path, python, system_site_packages=False):
    path = Path(path)
    if (path / 'bin/python').is_file():
        print(f'复用已有环境：{path}', flush=True)
        return
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'{path} 不是完整的 venv；请检查该目录。不会自动删除它。')
    command = [str(python), '-m', 'venv']
    if system_site_packages:
        # Bootstrap this venv with the IK environment's pip --python; this also
        # works on system Python installations without ensurepip.
        command += ['--system-site-packages', '--without-pip']
    execute([*command, str(path)])


def install_system_packages(env):
    if not shutil.which('apt-get'):
        raise ValueError('--system-deps requires Ubuntu apt-get')
    if not shutil.which('sudo') and os.geteuid() != 0:
        raise ValueError('sudo is unavailable; install the documented system packages as administrator')
    # Do not add repositories, download shell scripts, or alter network devices.
    prefix = [] if os.geteuid() == 0 else ['sudo']
    execute([*prefix, 'apt-get', 'update'], env)
    candidate = subprocess.run(['apt-cache', 'show', 'ros-humble-ros-base'],
                               text=True, capture_output=True, check=False)
    if candidate.returncode:
        raise ValueError('ROS apt 源尚未就绪；请先完成 https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html 的源配置，再重试 --system-deps。')
    execute([*prefix, 'apt-get', 'install', '-y', *SYSTEM_PACKAGES], env)


def check_command(args, env):
    offline = getattr(args, 'offline', False)
    python = runtime(env, 'ik' if offline else 'ros')
    command = [python, ROOT / 'tools/check_environment.py', '--mode', 'offline' if offline else 'live',
               '--solver-python', runtime(env, 'ik'), '--viser-python', viewer_runtime(env)]
    if not use_viser(args):
        command.append('--no-viser')
    return command


def setup(args, env):
    if args.print_plan:
        if args.system_deps:
            command_line(['sudo', 'apt-get', 'update'])
            command_line(['sudo', 'apt-get', 'install', '-y', *SYSTEM_PACKAGES])
        command_line([args.python, '-m', 'venv', ROOT / '.venv-ik'])
        command_line([runtime(env, 'ik'), '-m', 'pip', 'install', '-r',
                      ROOT / ('requirements.txt' if use_viser(args) else 'requirements-ik.txt')])
        command_line([SYSTEM_PYTHON, '-m', 'venv', '--system-site-packages', '--without-pip', ROOT / '.venv-ros'])
        command_line([runtime(env, 'ik'), '-m', 'pip', '--python', runtime(env, 'ros'),
                      'install', '-r', ROOT / 'requirements-live.txt'])
        command_line(check_command(args, env))
        print('仅安装运行环境并检查；此处仅预览，不安装、不写文件、不连接设备。')
        return
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'aarch64', 'arm64'):
        raise ValueError('The bundled PN SDK requires Linux x86_64 or ARM64.')
    os_release = Path('/etc/os-release').read_text()
    if 'VERSION_ID="22.04"' not in os_release or 'ID=ubuntu' not in os_release:
        raise ValueError('一键安装面向 Ubuntu 22.04 / ROS 2 Humble；其它平台请按文档手动配置。')
    if args.system_deps:
        install_system_packages(env)
    base = subprocess.check_output([str(SYSTEM_PYTHON), '-c', 'import sys; print("%d.%d" % sys.version_info[:2])'], text=True).strip()
    if base != '3.10':
        raise ValueError(f'ROS Humble 需要系统 Python 3.10，当前为 {base}')
    # Confirm ROS before installing large Python packages.
    load_ros_environment(env, required=True)
    for kind in ('ik', 'ros'):
        selected = runtime(env, kind)
        default = ROOT / ('.venv-ik' if kind == 'ik' else '.venv-ros') / 'bin/python'
        if selected != default and not selected.is_file():
            raise ValueError(f'.env 指定的 {kind} Python 不存在：{selected}；修正路径或移除该覆盖值。')
        if kind == 'ros' and selected.is_file():
            version = subprocess.check_output([str(selected), '-c', 'import sys; print("%d.%d" % sys.version_info[:2])'], text=True).strip()
            if version != '3.10':
                raise ValueError(f'ROS Python 必须为 3.10：{selected}')
    # Keep pip isolated from ROS PYTHONPATH and ~/.local packages.
    pip_env = dict(env, PYTHONNOUSERSITE='1', PYTHONPATH='')
    if not runtime(env, 'ik').is_file():
        seed_version = subprocess.check_output([str(args.python), '-c', 'import sys; print("%d.%d" % sys.version_info[:2])'], text=True).strip()
        if seed_version != '3.10':
            raise ValueError('--python 需要指向 Python 3.10。')
        try:
            ensure_venv(ROOT / '.venv-ik', args.python)
        except subprocess.CalledProcessError as exc:
            raise ValueError('创建 IK venv 失败。若缺少 ensurepip/python3-venv，运行 ./run.sh setup --system-deps 后重试；已保留现有文件。') from exc
    ik = runtime(env, 'ik')
    version = subprocess.check_output([str(ik), '-c', 'import sys; print("%d.%d" % sys.version_info[:2])'], text=True).strip()
    if version != '3.10':
        raise ValueError('固定依赖的一键安装需要 Python 3.10；可通过 --python /path/to/python3.10 指定 IK 环境来源。')
    pip_probe = subprocess.run([str(ik), '-m', 'pip', '--version'], env=pip_env,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    if pip_probe.returncode:
        # Repair a venv left by an earlier missing-ensurepip failure after the
        # user has installed python3-venv; never clear or recreate it.
        try:
            execute([ik, '-m', 'ensurepip', '--upgrade'], pip_env)
        except subprocess.CalledProcessError as exc:
            raise ValueError('IK 环境缺少 pip/ensurepip。运行 ./run.sh setup --system-deps 后重试。') from exc
    execute([ik, '-m', 'pip', 'install', '--upgrade', 'pip'], pip_env)
    requirements = 'requirements.txt' if use_viser(args) else 'requirements-ik.txt'
    execute([ik, '-m', 'pip', 'install', '-r', ROOT / requirements], pip_env)
    if not runtime(env, 'ros').is_file():
        ensure_venv(ROOT / '.venv-ros', SYSTEM_PYTHON, system_site_packages=True)
    execute([ik, '-m', 'pip', '--python', runtime(env, 'ros'), 'install', '-r', ROOT / 'requirements-live.txt'], pip_env)
    state = ROOT / '.runtime/setup.json'
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({'no_viser': not use_viser(args)}, indent=2) + '\n')
    env['PYTHONNOUSERSITE'] = '1'
    execute(check_command(args, env), load_ros_environment(env, required=True))
    print('运行环境安装检查完成；PN 地址、retarget 参数和标定文件未改动。', flush=True)


def demo_commands(args, env):
    python = runtime(env, 'ik')
    directory = ROOT / 'outputs/quick_demo'
    config = ROOT / 'mocapapi_python/rp1/pn_axis_to_rp1_28dof_current.json'
    npz = directory / 'synthetic_pn.npz'
    core = directory / 'ik/replay.npz'
    reference = directory / 'reference.jsonl'
    commands = [
        [python, ROOT / 'tools/make_demo_recording.py', '--output', npz],
        [python, ROOT / 'tools/evaluate_recording.py', '--source', npz, '--config', config,
         '--output-dir', directory / 'ik'],
        [python, ROOT / 'mocapapi_python/mocap_rp1_gmr_reference.py', '--source', 'replay',
         '--npz', npz, '--calibration', config, '--dry-run', '--autostart', '--no-viser',
         '--solver-python', python, '--replay-tail-s', '.2',
         '--output', reference, '--metrics', directory / 'metrics.json'],
    ]
    if use_viser(args):
        commands.append([viewer_runtime(env), ROOT / 'mocapapi_python/view_rp1_gmr_3d.py',
                         '--source', 'final', '--config', config, '--core', core,
                         '--reference', reference, '--port', args.viser_port])
    return commands


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog='Example: ./run.sh setup && ./run.sh demo; ./run.sh start')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('setup', 'configure', 'check', 'start', 'demo'):
        sub = commands.add_parser(name)
        sub.add_argument('--print-plan', action='store_true', help='print commands only; do not install, write, open PN or publish')
        if name in ('configure', 'start'):
            for key in NETWORK_KEYS:
                sub.add_argument('--' + key.lower().replace('_', '-'))
            sub.add_argument('--non-interactive', action='store_true')
            sub.add_argument('--publish-config', type=Path, help='publish and PN network YAML')
        if name == 'setup':
            sub.add_argument('--non-interactive', action='store_true', help=argparse.SUPPRESS)
        if name != 'configure':
            group = sub.add_mutually_exclusive_group()
            group.add_argument('--no-viser', dest='viser', action='store_false')
            group.add_argument('--viser', dest='viser', action='store_true')
            sub.set_defaults(viser=None)
        if name == 'setup':
            sub.add_argument('--system-deps', action='store_true', help='explicitly install Ubuntu/ROS system packages with apt/sudo')
            sub.add_argument('--python', type=Path, default=SYSTEM_PYTHON, help='Python 3.10 used to create the IK venv')
        if name == 'check':
            sub.add_argument('--offline', action='store_true', help='only IK/model and optional viewer; skip SDK/ROS')
        if name == 'demo':
            sub.add_argument('--viser-port', type=int, default=8890)
    args, extra = parser.parse_known_args(argv)
    if extra and args.command != 'start':
        parser.error('unrecognized arguments: ' + ' '.join(extra))
    try:
        env = read_env(ROOT / '.env' if (ROOT / '.env').is_file() else ROOT / '.env.example')
        settings = explicit_settings(args)
        if 'ROS_DOMAIN_ID' in settings:
            env['ROS_DOMAIN_ID'] = settings['ROS_DOMAIN_ID']
        env['PYTHONNOUSERSITE'] = '1'
        if args.command == 'setup':
            setup(args, env)
        elif args.command == 'configure':
            if args.print_plan:
                path = args.publish_config or ROOT / 'mocapapi_python/rp1/publish.yaml'
                print(f'Update {path} pn_network IP/port fields; only explicit ROS_DOMAIN_ID updates .env. Preserve other settings.')
            else:
                configure(args, env)
        elif args.command == 'check':
            execute(check_command(args, env), env if args.print_plan else load_ros_environment(env, required=not args.offline), args.print_plan)
        elif args.command == 'demo':
            if use_viser(args):
                print(f'离线 3D: http://127.0.0.1:{args.viser_port}/ （保持终端运行，Ctrl-C 退出）', flush=True)
            for command in demo_commands(args, env):
                execute(command, env, args.print_plan)
        else:
            pn_arguments = [value for key in NETWORK_KEYS if key.startswith('PN_') and key in settings
                            for value in ('--' + key.lower().replace('_', '-'), settings[key])]
            if args.publish_config is not None:
                pn_arguments.extend(('--publish-config', str(args.publish_config)))
            command = [SYSTEM_PYTHON, ROOT / 'tools/start_live.py',
                       '--ros-python', runtime(env, 'ros'), '--solver-python', runtime(env, 'ik'),
                       '--viser-python', viewer_runtime(env), '--viser' if use_viser(args) else '--no-viser',
                       *pn_arguments, *extra]
            if args.print_plan:
                execute(command, env, True)
            else:
                env = load_ros_environment(env, required='--dry-run' not in extra)
                command_line(command)
                os.chdir(ROOT)
                os.execve(str(SYSTEM_PYTHON), list(map(str, command)), env)
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('\n已停止。', file=sys.stderr)
        return 130


if __name__ == '__main__':
    sys.exit(main())
