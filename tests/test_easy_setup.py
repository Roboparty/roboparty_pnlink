"""Exercise first-run setup without installing packages or opening a device."""

import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = next(path for path in (ROOT / "mocapapi_python", ROOT.parents[1] / "mocapapi_python")
              if (path / "pn_network.py").is_file())


@pytest.fixture
def setup_module(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "pn_rp1_easy_setup_test", ROOT / "tools/setup_environment.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    repository = tmp_path / "repository with spaces"
    repository.mkdir()
    monkeypatch.setattr(module, "ROOT", repository)
    monkeypatch.syspath_prepend(str(SOURCE))
    publish = repository / "mocapapi_python/rp1/publish.yaml"
    publish.parent.mkdir(parents=True)
    publish.write_text(
        "# Preserve publication settings\n"
        "reference:\n  rate_hz: 50\n  root_z_offset_m: -0.04\n"
        "hands:\n  enabled: true\n"
        "pn_network:\n  local_ip: 10.42.0.101\n  local_port: 8002\n"
        "  server_ip: 10.42.0.202\n  server_port: 8080\n")
    return module


def test_env_updates_preserve_custom_settings_and_comments(setup_module):
    path = setup_module.ROOT / ".env"
    path.write_text(
        "# Operator settings\n"
        "CUSTOM_OPTION='keep me'\n"
        "PN_LOCAL_IP=10.42.0.101\n"
        "PN_SERVER_IP=10.42.0.202\n"
        "PN_RP1_SOLVER_PYTHON='/custom/runtime with spaces/bin/python'\n")
    setup_module.write_env(path, {"PN_LOCAL_IP": "192.168.10.1"})
    values = setup_module.read_env(path)
    assert values["PN_LOCAL_IP"] == "192.168.10.1"
    assert values["PN_SERVER_IP"] == "10.42.0.202"
    assert values["PN_RP1_SOLVER_PYTHON"] == "/custom/runtime with spaces/bin/python"
    assert values["CUSTOM_OPTION"] == "keep me"
    assert "# Operator settings" in path.read_text()


def test_read_env_expands_repo_path_without_shell_execution(setup_module):
    marker = setup_module.ROOT / "must-not-be-created"
    path = setup_module.ROOT / ".env"
    path.write_text(
        'PN_RP1_MODEL="$PWD/assets/robots/rp1.xml"\n'
        'PN_RP1_GMR_ROOT="${PWD}/third_party/GMR"\n'
        f'CUSTOM_OPTION="$(touch {marker})"\n')
    values = setup_module.read_env(path)
    assert values["PN_RP1_MODEL"] == str(setup_module.ROOT / "assets/robots/rp1.xml")
    assert values["PN_RP1_GMR_ROOT"] == str(setup_module.ROOT / "third_party/GMR")
    assert "$(touch " in values["CUSTOM_OPTION"]
    assert not marker.exists()


def test_write_env_does_not_duplicate_existing_keys(setup_module):
    path = setup_module.ROOT / ".env"
    path.write_text("PN_LOCAL_IP=10.42.0.101\n# keep this comment\n")
    setup_module.write_env(path, {"PN_LOCAL_IP": "192.168.10.1"})
    setup_module.write_env(path, {"PN_LOCAL_IP": "192.168.10.2"})
    lines = path.read_text().splitlines()
    assert sum(line.startswith("PN_LOCAL_IP=") for line in lines) == 1
    assert setup_module.read_env(path)["PN_LOCAL_IP"] == "192.168.10.2"
    assert "# keep this comment" in lines


def test_existing_venv_is_reused_without_running_any_install(setup_module):
    environment = setup_module.ROOT / ".venv-ik"
    python = environment / "bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("existing interpreter")
    with mock.patch.object(setup_module, "execute") as execute:
        setup_module.ensure_venv(environment, "/missing/bootstrap/python")
    execute.assert_not_called()
    assert python.read_text() == "existing interpreter"


def test_incomplete_venv_is_preserved_and_reported(setup_module):
    environment = setup_module.ROOT / ".venv-ik"
    environment.mkdir()
    marker = environment / "existing-local-file"
    marker.write_text("keep")
    with mock.patch.object(setup_module, "execute") as execute:
        with pytest.raises(ValueError, match="venv"):
            setup_module.ensure_venv(environment, sys.executable)
    execute.assert_not_called()
    assert marker.read_text() == "keep"


@pytest.mark.parametrize("arguments", [
    ["setup", "--print-plan", "--system-deps", "--no-viser"],
    ["configure", "--print-plan", "--pn-local-ip", "10.42.0.101"],
    ["check", "--print-plan", "--offline", "--no-viser"],
    ["start", "--print-plan", "--no-viser"],
    ["demo", "--print-plan", "--no-viser"],
])
def test_plan_modes_do_not_execute_or_write(setup_module, arguments, monkeypatch):
    (setup_module.ROOT / ".env.example").write_text(
        'PN_RP1_MODEL="$PWD/assets/model.xml"\n'
        "PN_LOCAL_IP=10.42.0.101\nPN_SERVER_IP=10.42.0.202\n")
    before = {path.relative_to(setup_module.ROOT): path.read_bytes()
              for path in setup_module.ROOT.rglob("*") if path.is_file()}
    # Importing native or ROS modules here would violate the setup-plan contract.
    for name in ("mocap_api", "pn_sdk", "rclpy", "mujoco", "mink"):
        monkeypatch.setitem(sys.modules, name, None)
    with mock.patch.object(setup_module.subprocess, "run") as run, \
            mock.patch.object(setup_module.subprocess, "check_output") as output, \
            mock.patch.object(setup_module.os, "execve") as execve:
        assert setup_module.main(arguments) == 0
    run.assert_not_called()
    output.assert_not_called()
    execve.assert_not_called()
    after = {path.relative_to(setup_module.ROOT): path.read_bytes()
             for path in setup_module.ROOT.rglob("*") if path.is_file()}
    assert after == before


def test_configure_only_changes_explicit_network_fields(setup_module):
    path = setup_module.ROOT / ".env"
    original = (b"# local choices\r\nPN_LOCAL_IP=10.42.0.101\r\n"
                b"PN_SERVER_IP=10.42.0.202\r\nPN_LOCAL_PORT=8002\r\n"
                b"PN_SERVER_PORT=8080\r\nROS_DOMAIN_ID=217\r\n"
                b"PN_RP1_SOLVER_PYTHON='/custom ik/bin/python'\r\n")
    path.write_bytes(original)
    assert setup_module.main(["configure", "--non-interactive",
                              "--pn-local-ip", "192.168.10.1"]) == 0
    from pn_network import read_pn_network_yaml
    publish = setup_module.ROOT / "mocapapi_python/rp1/publish.yaml"
    network = read_pn_network_yaml(publish)
    assert network["local_ip"] == "192.168.10.1"
    assert network["server_ip"] == "10.42.0.202"
    assert network["local_port"] == 8002
    assert network["server_port"] == 8080
    assert path.read_bytes() == original
    result = setup_module.read_env(path)
    assert result["PN_LOCAL_IP"] == "10.42.0.101"
    assert result["PN_SERVER_IP"] == "10.42.0.202"
    assert result["ROS_DOMAIN_ID"] == "217"
    assert result["PN_RP1_SOLVER_PYTHON"] == "/custom ik/bin/python"
    assert "# local choices" in path.read_text()
    assert "# Preserve publication settings" in publish.read_text()
    assert "  root_z_offset_m: -0.04\n" in publish.read_text()
    assert "hands:\n  enabled: true\n" in publish.read_text()


def test_configure_selected_yaml_and_domain_preserves_default_yaml(setup_module):
    path = setup_module.ROOT / ".env"
    path.write_text("# keep runtime\nROS_DOMAIN_ID=217\nPN_LOCAL_IP=10.42.0.101\n")
    default = setup_module.ROOT / "mocapapi_python/rp1/publish.yaml"
    original = default.read_bytes()
    selected = setup_module.ROOT / "another publish.yaml"
    selected.write_bytes(original)
    assert setup_module.main([
        "configure", "--non-interactive", "--publish-config", str(selected),
        "--pn-local-ip", "192.168.10.1", "--pn-server-ip", "192.168.10.2",
        "--pn-local-port", "8003", "--pn-server-port", "8081",
        "--ros-domain-id", "218",
    ]) == 0
    from pn_network import read_pn_network_yaml
    assert read_pn_network_yaml(selected) == {
        "local_ip": "192.168.10.1", "local_port": 8003,
        "server_ip": "192.168.10.2", "server_port": 8081,
    }
    assert default.read_bytes() == original
    assert setup_module.read_env(path)["ROS_DOMAIN_ID"] == "218"
    assert "PN_LOCAL_IP=10.42.0.101\n" in path.read_text()
    assert "# keep runtime" in path.read_text()


def test_start_forwards_explicit_pn_fields_and_yaml_without_replacing_env(setup_module):
    path = setup_module.ROOT / ".env"
    path.write_text("PN_LOCAL_IP=10.42.0.101\nPN_SERVER_IP=10.42.0.202\nROS_DOMAIN_ID=217\n")
    original = path.read_bytes()
    selected = setup_module.ROOT / "selected publish.yaml"
    with mock.patch.object(setup_module, "execute") as execute:
        assert setup_module.main([
            "start", "--print-plan", "--no-viser", "--pn-local-ip", "192.168.10.1",
            "--pn-server-port", "8081", "--publish-config", str(selected),
            "--ros-domain-id", "218",
        ]) == 0
    command, environment, preview = execute.call_args.args
    command = list(map(str, command))
    assert command[command.index("--pn-local-ip") + 1] == "192.168.10.1"
    assert command[command.index("--pn-server-port") + 1] == "8081"
    assert command[command.index("--publish-config") + 1] == str(selected)
    assert "--pn-server-ip" not in command
    assert "--pn-local-port" not in command
    assert environment["PN_LOCAL_IP"] == "10.42.0.101"
    assert environment["PN_SERVER_IP"] == "10.42.0.202"
    assert environment["ROS_DOMAIN_ID"] == "218"
    assert preview is True
    assert path.read_bytes() == original


@pytest.fixture
def start_module(setup_module, monkeypatch):
    spec = importlib.util.spec_from_file_location("pn_rp1_start_test", ROOT / "tools/start_live.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", setup_module.ROOT)
    monkeypatch.setattr(module, "SOURCE", setup_module.ROOT / "mocapapi_python")
    calibration = module.SOURCE / "rp1/pn_axis_to_rp1_28dof_current.json"
    calibration.write_text("{}\n")
    return module


def start_preview(module, *arguments):
    return module.main([
        "--print-command", "--no-viser", "--ros-python", sys.executable,
        "--solver-python", sys.executable, "--viser-python", sys.executable,
        *arguments,
    ])


def test_start_live_uses_yaml_ahead_of_legacy_env_without_opening_device(
        start_module, monkeypatch, capsys):
    monkeypatch.setenv("PN_LOCAL_IP", "192.0.2.10")
    monkeypatch.setenv("PN_SERVER_IP", "192.0.2.20")
    monkeypatch.setenv("PN_LOCAL_PORT", "9000")
    for name in ("mocap_api", "pn_sdk", "rclpy", "mujoco", "mink"):
        monkeypatch.setitem(sys.modules, name, None)
    import pn_network
    with mock.patch.object(pn_network, "preflight_pn_network", side_effect=AssertionError("must not bind")), \
            mock.patch.object(start_module.os, "execve") as execve:
        assert start_preview(start_module) == 0
    execve.assert_not_called()
    command = shlex.split(capsys.readouterr().out.strip())
    assert command[command.index("--pn-local-ip") + 1] == "10.42.0.101"
    assert command[command.index("--pn-server-ip") + 1] == "10.42.0.202"
    assert command[command.index("--pn-local-port") + 1] == "8002"
    assert command[command.index("--publish-config") + 1] == str(start_module.SOURCE / "rp1/publish.yaml")


def test_start_live_cli_overrides_only_given_fields_in_selected_yaml(
        start_module, capsys):
    selected = start_module.ROOT / "selected publish.yaml"
    selected.write_text("pn_network:\n  local_ip: 192.168.10.1\n  local_port: 8003\n"
                        "  server_ip: 192.168.10.2\n  server_port: 8081\n")
    assert start_preview(start_module, "--publish-config", str(selected),
                         "--pn-local-ip", "192.168.10.3", "--pn-server-port", "8082") == 0
    command = shlex.split(capsys.readouterr().out.strip())
    assert command[command.index("--pn-local-ip") + 1] == "192.168.10.3"
    assert command[command.index("--pn-local-port") + 1] == "8003"
    assert command[command.index("--pn-server-ip") + 1] == "192.168.10.2"
    assert command[command.index("--pn-server-port") + 1] == "8082"
    assert command[command.index("--publish-config") + 1] == str(selected)


def test_start_live_still_rejects_documentation_only_addresses(start_module):
    with pytest.raises(SystemExit) as raised:
        start_preview(start_module, "--pn-local-ip", "192.0.2.10")
    assert raised.value.code == 2


@pytest.mark.parametrize("option,value", [
    ("--pn-local-ip", "999.1.2.3"),
    ("--pn-server-ip", "::1"),
    ("--pn-local-ip", "0.0.0.0"),
    ("--pn-server-ip", "224.0.0.1"),
    ("--pn-local-port", "65536"),
    ("--ros-domain-id", "233"),
])
def test_invalid_network_settings_are_rejected_before_any_write(
        setup_module, option, value):
    example = setup_module.ROOT / ".env.example"
    example.write_text("PN_LOCAL_IP=10.42.0.101\n")
    with mock.patch.object(setup_module.subprocess, "run") as run:
        assert setup_module.main(["configure", "--non-interactive", option, value]) == 1
    run.assert_not_called()
    assert not (setup_module.ROOT / ".env").exists()


def test_saved_no_viser_choice_is_used_and_can_be_overridden(setup_module):
    state = setup_module.ROOT / ".runtime/setup.json"
    state.parent.mkdir()
    state.write_text(json.dumps({"no_viser": True}))
    assert not setup_module.use_viser(SimpleNamespace(viser=None))
    assert setup_module.use_viser(SimpleNamespace(viser=True))
    args = SimpleNamespace(viser=None, viser_port=8890)
    commands = setup_module.demo_commands(args, {})
    assert all("view_rp1_gmr_3d.py" not in str(command) for command in commands)
    args.viser = True
    assert any("view_rp1_gmr_3d.py" in str(command)
               for command in setup_module.demo_commands(args, {}))


def test_demo_reference_command_has_no_live_input_or_ros_publishing(setup_module):
    commands = setup_module.demo_commands(SimpleNamespace(viser=False), {})
    bridge = next(command for command in commands
                  if str(command[1]).endswith("mocap_rp1_gmr_reference.py"))
    assert bridge[bridge.index("--source") + 1] == "replay"
    assert "--dry-run" in bridge
    assert "--no-viser" in bridge
    assert all("start_live.py" not in str(command) for command in commands)


@pytest.mark.parametrize("arguments", [
    ["check", "--offline", "--no-viser"],
    ["demo", "--no-viser"],
])
def test_check_and_demo_execute_only_runtime_commands_without_installing(
        setup_module, arguments):
    (setup_module.ROOT / ".env.example").write_text("ROS_DOMAIN_ID=217\n")
    with mock.patch.object(setup_module, "execute") as execute, \
            mock.patch.object(setup_module, "ensure_venv") as venv, \
            mock.patch.object(setup_module, "install_system_packages") as apt, \
            mock.patch.object(setup_module, "configure") as configure:
        assert setup_module.main(arguments) == 0
    venv.assert_not_called()
    apt.assert_not_called()
    configure.assert_not_called()
    assert execute.call_count >= 1
    for call in execute.call_args_list:
        command = list(map(str, call.args[0]))
        assert "pip" not in command
        assert "apt-get" not in command
        assert "start_live.py" not in command


def test_ros_setup_is_passed_as_positional_path_without_shell_interpolation(
        setup_module, monkeypatch):
    ros_setup = setup_module.ROOT / "ros directory with spaces/setup.bash"
    ros_setup.parent.mkdir()
    ros_setup.write_text("# local test only")
    monkeypatch.setattr(setup_module, "ROS_SETUP", ros_setup)
    result = SimpleNamespace(stdout=b"ROS_DISTRO=humble\0EXTRA=kept\0", stderr=b"")
    with mock.patch.object(setup_module.subprocess, "run", return_value=result) as run:
        environment = setup_module.load_ros_environment({"CUSTOM_OPTION": "$(touch ignored)"})
    argv = run.call_args.args[0]
    assert argv[0:2] == ["bash", "-c"]
    assert 'source "$1"' in argv[2]
    assert str(ros_setup) not in argv[2]
    assert argv[-1] == str(ros_setup)
    assert environment["ROS_DISTRO"] == "humble"
    assert environment["PYTHONNOUSERSITE"] == "1"


def test_root_shell_entry_works_from_another_directory_with_spaces(tmp_path):
    repository = tmp_path / "a repository with spaces"
    tools = repository / "tools"
    tools.mkdir(parents=True)
    shutil.copyfile(ROOT / "run.sh", repository / "run.sh")
    shutil.copyfile(ROOT / "tools/setup_environment.py", tools / "setup_environment.py")
    (repository / ".env.example").write_text("PN_LOCAL_IP=10.42.0.101\n")
    result = subprocess.run(["bash", str(repository / "run.sh"), "setup",
                             "--print-plan", "--no-viser"],
                            cwd=tmp_path, env=dict(os.environ, PYTHONNOUSERSITE="1"),
                            text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert str(repository / ".venv-ik") in result.stdout
    assert not (repository / ".env").exists()
    assert not (repository / ".venv-ik").exists()


@pytest.mark.parametrize("existing_env", [False, True])
def test_full_setup_only_installs_environment_and_preserves_env(
        setup_module, existing_env, monkeypatch):
    """Mock system/package operations, but exercise the real setup control flow."""
    path = setup_module.ROOT / ".env"
    publish = setup_module.ROOT / "mocapapi_python/rp1/publish.yaml"
    original_publish = publish.read_bytes()
    original = (b"# existing user configuration\r\n"
                b"PN_LOCAL_IP=10.42.0.101\r\n"
                b"PN_SERVER_IP=10.42.0.202\r\n"
                b"CUSTOM_OPTION='leave this alone'\r\n")
    if existing_env:
        path.write_bytes(original)
    (setup_module.ROOT / ".env.example").write_text("ROS_DOMAIN_ID=217\n")
    for key in ("PN_RP1_SOLVER_PYTHON", "PN_RP1_ROS_PYTHON", "PN_RP1_VISER_PYTHON"):
        monkeypatch.delenv(key, raising=False)
    read_text = Path.read_text

    def read_with_supported_os(file, *args, **kwargs):
        if str(file) == "/etc/os-release":
            return 'ID=ubuntu\nVERSION_ID="22.04"\n'
        return read_text(file, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_with_supported_os)
    monkeypatch.setattr(setup_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(setup_module.platform, "machine", lambda: "x86_64")
    with mock.patch.object(setup_module.subprocess, "check_output", return_value="3.10\n"), \
            mock.patch.object(setup_module.subprocess, "run", return_value=SimpleNamespace(returncode=0)), \
            mock.patch.object(setup_module, "load_ros_environment", side_effect=lambda env, **kw: dict(env)), \
            mock.patch.object(setup_module, "ensure_venv") as venv, \
            mock.patch.object(setup_module, "execute") as execute, \
            mock.patch.object(setup_module, "configure", side_effect=AssertionError("setup must not configure PN")) as configure, \
            mock.patch.object(setup_module, "demo_commands", side_effect=AssertionError("setup must not launch demo")), \
            mock.patch("builtins.input", side_effect=AssertionError("setup must not prompt for IPs")):
        assert setup_module.main(["setup", "--no-viser"]) == 0
    configure.assert_not_called()
    assert venv.call_count == 2
    assert any("pip" in list(map(str, call.args[0])) for call in execute.call_args_list)
    for call in execute.call_args_list:
        command = list(map(str, call.args[0]))
        assert not any("start_live.py" in value or "mocap_rp1_gmr_reference.py" in value
                       or "make_demo_recording.py" in value for value in command)
    if existing_env:
        assert path.read_bytes() == original
    else:
        assert not path.exists()
    assert publish.read_bytes() == original_publish


@pytest.mark.parametrize("option,value", [
    ("--pn-local-ip", "10.42.0.101"),
    ("--pn-server-ip", "10.42.0.202"),
    ("--ros-domain-id", "217"),
])
def test_setup_does_not_accept_network_configuration_arguments(setup_module, option, value):
    with pytest.raises(SystemExit) as raised:
        setup_module.main(["setup", "--print-plan", option, value])
    assert raised.value.code == 2
    assert not (setup_module.ROOT / ".env").exists()


def test_install_shell_entry_works_from_another_directory_with_spaces(tmp_path):
    repository = tmp_path / "install repository with spaces"
    tools = repository / "tools"
    tools.mkdir(parents=True)
    for filename in ("install.sh", "run.sh"):
        target = repository / filename
        shutil.copyfile(ROOT / filename, target)
        target.chmod(0o755)
    shutil.copyfile(ROOT / "tools/setup_environment.py", tools / "setup_environment.py")
    (repository / ".env.example").write_text("ROS_DOMAIN_ID=217\n")
    result = subprocess.run(["bash", str(repository / "install.sh"), "--print-plan", "--no-viser"],
                            cwd=tmp_path, env=dict(os.environ, PYTHONNOUSERSITE="1"),
                            text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert str(repository / ".venv-ik") in result.stdout
    assert "pip" in result.stdout
    assert not (repository / ".env").exists()
    assert not (repository / ".venv-ik").exists()
