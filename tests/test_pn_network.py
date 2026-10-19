"""Reject invalid PN UDP endpoints before entering the native SDK.

These tests never load the PN native library, open a socket or publish ROS.
"""

import asyncio
from contextlib import redirect_stderr
import errno
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

from pn_network import PNNetworkConfig, PNNetworkError, preflight_pn_network, validate_pn_network
import pn_network


DIRECTORY = Path(__file__).resolve().parents[1] / "mocapapi_python"


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, DIRECTORY / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PNNetworkValidationTests(unittest.TestCase):
    def test_default_endpoints_preserve_the_existing_device_contract(self):
        config = validate_pn_network(PNNetworkConfig())
        self.assertEqual((config.local_ip, config.local_port), ("10.42.0.101", 8002))
        self.assertEqual((config.server_ip, config.server_port), ("10.42.0.202", 8080))

    def test_explicit_ipv4_endpoints_and_boundary_ports_are_supported(self):
        config = PNNetworkConfig(local_ip="192.168.137.1", local_port=1,
                                 server_ip="192.168.137.250", server_port=65535)
        self.assertEqual(validate_pn_network(config), config)

    def test_invalid_ipv4_is_rejected_without_socket_creation(self):
        with mock.patch.object(pn_network.socket, "socket") as socket_factory:
            for field in ("local_ip", "server_ip"):
                for address in ("", "not-an-ip", "999.1.2.3", "::1", "localhost"):
                    with self.subTest(field=field, address=address):
                        with self.assertRaises(PNNetworkError):
                            preflight_pn_network(PNNetworkConfig(**{field: address}))
            socket_factory.assert_not_called()

    def test_invalid_ports_are_rejected_without_socket_creation(self):
        with mock.patch.object(pn_network.socket, "socket") as socket_factory:
            for field in ("local_port", "server_port"):
                for port in (0, -1, 65536, 1.5, True):
                    with self.subTest(field=field, port=port):
                        with self.assertRaises(PNNetworkError):
                            preflight_pn_network(PNNetworkConfig(**{field: port}))
            socket_factory.assert_not_called()

    def test_non_unicast_endpoints_and_same_host_are_rejected(self):
        for field in ("local_ip", "server_ip"):
            for address in ("0.0.0.0", "224.0.0.1", "239.255.255.250", "255.255.255.255"):
                with self.subTest(field=field, address=address):
                    with self.assertRaises(PNNetworkError):
                        validate_pn_network(PNNetworkConfig(**{field: address}))
        with self.assertRaises(PNNetworkError):
            validate_pn_network(PNNetworkConfig(local_ip="192.168.137.1", server_ip="192.168.137.1"))


class PNNetworkPreflightTests(unittest.TestCase):
    def setUp(self):
        self.config = PNNetworkConfig(local_ip="192.168.137.1", local_port=18002,
                                      server_ip="192.168.137.250", server_port=8080)
        self.sock = mock.MagicMock()
        self.sock.__enter__.return_value = self.sock

    def test_probe_binds_the_local_endpoint_without_sending_to_device(self):
        with mock.patch.object(pn_network.socket, "socket", return_value=self.sock):
            result = preflight_pn_network(self.config)
        self.assertEqual(result, self.config)
        self.sock.bind.assert_called_once_with(("192.168.137.1", 18002))
        self.sock.connect.assert_not_called()
        self.sock.send.assert_not_called()
        self.sock.sendto.assert_not_called()
        self.assertTrue(self.sock.close.called or self.sock.__exit__.called)

    def test_missing_local_address_is_reported_as_python_error(self):
        self.sock.bind.side_effect = OSError(errno.EADDRNOTAVAIL, "Cannot assign requested address")
        with mock.patch.object(pn_network.socket, "socket", return_value=self.sock):
            with self.assertRaises(PNNetworkError) as raised:
                preflight_pn_network(self.config)
        self.assertIn("192.168.137.1", str(raised.exception))
        self.assertIn("18002", str(raised.exception))
        self.assertTrue(self.sock.close.called or self.sock.__exit__.called)

    def test_occupied_port_is_distinguished_from_missing_address(self):
        self.sock.bind.side_effect = OSError(errno.EADDRINUSE, "Address already in use")
        with mock.patch.object(pn_network.socket, "socket", return_value=self.sock):
            with self.assertRaises(PNNetworkError) as raised:
                preflight_pn_network(self.config)
        message = str(raised.exception).lower()
        self.assertIn("18002", message)
        self.assertTrue("in use" in message or "occupied" in message or "占用" in message)
        self.assertTrue(self.sock.close.called or self.sock.__exit__.called)


class PNNetworkYAMLTests(unittest.TestCase):
    PREFIX = ("# Shared publication settings must be left intact.\n"
              "reference:\n"
              "  topic: /tracking/current_reference\n"
              "  rate_hz: 50\n"
              "  qos_depth: 21\n"
              "  reliability: reliable\n"
              "  transition_steps: 100\n"
              "  source_timeout_ms: 1000\n"
              "  root_z_offset_m: -0.04  # preserve the current height\n"
              "\n"
              "hands:\n"
              "  enabled: true\n"
              "  left_poses: /mocap/hand/left/poses\n"
              "  right_poses: /mocap/hand/right/poses\n")
    FIELDS = {"local_ip": "192.168.137.1", "local_port": 18002,
              "server_ip": "192.168.137.250", "server_port": 18080}

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="pn network yaml ")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "publish config.yaml"

    def write_yaml(self, fields=None, *, with_section=True):
        text = self.PREFIX
        if with_section:
            text += "\npn_network:\n"
            for key, value in (self.FIELDS if fields is None else fields).items():
                text += f"  {key}: {value}\n"
        self.path.write_text(text)
        return self.path

    def assertConfig(self, actual, expected):
        self.assertIsInstance(actual, PNNetworkConfig)
        self.assertEqual({key: getattr(actual, key) for key in self.FIELDS}, expected)

    def test_read_quotes_comments_and_integer_ports_without_hardware(self):
        self.path.write_text(self.PREFIX + (
            "\npn_network:  # only this section is network configuration\n"
            "  local_ip: '192.168.137.1' # PC-side address\n"
            '  server_ip: "192.168.137.250"\n'
            '  local_port: "18002"\n'
            "  server_port: '18080'\n"))
        api = ModuleType("mocap_api")
        api.MCPApplication = mock.Mock()
        with mock.patch.dict(sys.modules, {"mocap_api": api}), \
                mock.patch.object(pn_network.socket, "socket") as socket_factory:
            self.assertEqual(pn_network.read_pn_network_yaml(self.path), self.FIELDS)
            self.assertConfig(pn_network.load_pn_network(self.path, environ={}), self.FIELDS)
        socket_factory.assert_not_called()
        api.MCPApplication.assert_not_called()

    def test_yaml_takes_priority_over_conflicting_environment(self):
        self.write_yaml()
        env = {"PN_LOCAL_IP": "192.168.31.100", "PN_LOCAL_PORT": "28002",
               "PN_SERVER_IP": "192.168.31.202", "PN_SERVER_PORT": "28080"}
        self.assertConfig(pn_network.load_pn_network(self.path, environ=env), self.FIELDS)

    def test_priority_is_resolved_per_field_cli_then_yaml_then_environment(self):
        self.write_yaml({"local_ip": "192.168.137.1", "server_port": 18080})
        env = {"PN_LOCAL_IP": "192.168.31.100", "PN_LOCAL_PORT": "28002",
               "PN_SERVER_IP": "192.168.31.202", "PN_SERVER_PORT": "28080"}
        expected = {"local_ip": "192.168.137.10", "local_port": 28002,
                    "server_ip": "192.168.31.202", "server_port": 18080}
        actual = pn_network.load_pn_network(self.path, overrides={"local_ip": "192.168.137.10"},
                                            environ=env)
        self.assertConfig(actual, expected)

    def test_none_cli_overrides_do_not_hide_yaml_values(self):
        self.write_yaml()
        self.assertConfig(pn_network.load_pn_network(
            self.path, overrides={key: None for key in self.FIELDS}, environ={}), self.FIELDS)

    def test_missing_section_preserves_legacy_defaults_and_environment_fallback(self):
        self.write_yaml(with_section=False)
        self.assertEqual(pn_network.read_pn_network_yaml(self.path), {})
        self.assertEqual(pn_network.load_pn_network(self.path, environ={}), PNNetworkConfig())
        env = {"PN_LOCAL_IP": "192.168.31.100", "PN_SERVER_IP": "192.168.31.202",
               "PN_LOCAL_PORT": "18002", "PN_SERVER_PORT": "18080"}
        expected = {"local_ip": env["PN_LOCAL_IP"], "server_ip": env["PN_SERVER_IP"],
                    "local_port": 18002, "server_port": 18080}
        self.assertConfig(pn_network.load_pn_network(self.path, environ=env), expected)

    def test_default_environment_is_read_only_when_no_explicit_mapping_is_supplied(self):
        self.write_yaml(with_section=False)
        env = {"PN_LOCAL_IP": "192.168.31.100", "PN_SERVER_IP": "192.168.31.202"}
        with mock.patch.dict(os.environ, env, clear=True):
            actual = pn_network.load_pn_network(self.path)
        self.assertEqual(actual.local_ip, env["PN_LOCAL_IP"])
        self.assertEqual(actual.server_ip, env["PN_SERVER_IP"])

    def test_invalid_ip_ports_same_host_and_unknown_network_keys_are_rejected(self):
        cases = [("local_ip", "bad-ip"), ("server_ip", "999.1.2.3"),
                 ("local_port", 0), ("server_port", 65536),
                 ("local_port", "1.5"), ("server_port", "true"),
                 ("server_ip", self.FIELDS["local_ip"]), ("typo_server_ip", "192.168.1.2")]
        with mock.patch.object(pn_network.socket, "socket") as socket_factory:
            for key, value in cases:
                fields = dict(self.FIELDS)
                fields[key] = value
                self.write_yaml(fields)
                with self.subTest(key=key, value=value), self.assertRaises(PNNetworkError):
                    pn_network.load_pn_network(self.path, environ={})
        socket_factory.assert_not_called()

    def test_unknown_override_keys_are_rejected_instead_of_silently_ignored(self):
        self.write_yaml()
        with self.assertRaises(PNNetworkError):
            pn_network.load_pn_network(self.path, overrides={"local_ipp": "192.168.1.2"}, environ={})

    def test_write_updates_only_selected_network_fields_and_preserves_publication_text(self):
        self.path.write_text(self.PREFIX + (
            "\npn_network:\n"
            "  local_ip: 192.168.137.1  # PC-side address\n"
            "  local_port: 18002\n"
            "  server_ip: 192.168.137.250\n"
            "  server_port: 18080\n"
            "\n# Another section must not become part of pn_network.\n"
            "diagnostics:\n"
            "  enabled: false\n"))
        before = self.path.read_text()
        with mock.patch.object(pn_network.socket, "socket") as socket_factory:
            pn_network.write_pn_network_yaml(self.path, {"local_ip": "192.168.137.10"})
        after = self.path.read_text()
        self.assertTrue(after.startswith(self.PREFIX))
        self.assertIn("# PC-side address", after)
        self.assertIn("  local_port: 18002\n", after)
        self.assertIn("  server_ip: 192.168.137.250\n", after)
        self.assertIn("  server_port: 18080\n", after)
        self.assertEqual(after[after.index("# Another section"):], before[before.index("# Another section"):])
        expected = dict(self.FIELDS, local_ip="192.168.137.10")
        self.assertEqual(pn_network.read_pn_network_yaml(self.path), expected)
        socket_factory.assert_not_called()

    def test_write_appends_missing_network_section_without_changing_other_sections(self):
        self.write_yaml(with_section=False)
        pn_network.write_pn_network_yaml(self.path, dict(self.FIELDS))
        self.assertTrue(self.path.read_text().startswith(self.PREFIX))
        self.assertEqual(pn_network.read_pn_network_yaml(self.path), self.FIELDS)
        self.assertEqual(self.path.read_text().count("pn_network:"), 1)

    def test_invalid_update_leaves_the_existing_file_unchanged(self):
        for update in ({"local_ip": "bad-ip"}, {"server_port": 0},
                       {"server_ip": self.FIELDS["local_ip"]}, {"local_ipp": "192.168.137.10"}):
            self.write_yaml()
            before = self.path.read_bytes()
            with self.subTest(update=update), self.assertRaises(PNNetworkError):
                pn_network.write_pn_network_yaml(self.path, update)
            self.assertEqual(self.path.read_bytes(), before)

    def test_failed_atomic_replace_preserves_original_and_cleans_temporary_file(self):
        self.write_yaml()
        before = self.path.read_bytes()
        with mock.patch.object(Path, "replace", side_effect=OSError("synthetic replace failure")):
            with self.assertRaisesRegex(OSError, "synthetic replace failure"):
                pn_network.write_pn_network_yaml(self.path, {"local_ip": "192.168.137.10"})
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_write_preserves_crlf_and_file_permissions(self):
        self.write_yaml()
        original = self.path.read_text().replace("\n", "\r\n").encode()
        self.path.write_bytes(original)
        self.path.chmod(0o640)
        pn_network.write_pn_network_yaml(self.path, {"local_ip": "192.168.137.10"})
        expected = original.replace(b"local_ip: 192.168.137.1\r\n", b"local_ip: 192.168.137.10\r\n")
        self.assertEqual(self.path.read_bytes(), expected)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o640)


@unittest.skipUnless((DIRECTORY / "mocap_main_base.py").is_file(),
    "Vendor sample is external; pn_sdk adapter has separate mock tests")
class PNNetworkSDKBoundaryTests(unittest.TestCase):
    def fake_api(self):
        api = ModuleType("mocap_api")
        api.MCPApplication = mock.Mock()
        api.MCPApplication.return_value.open.return_value = (True, "NoError")
        api.MCPSettings = mock.Mock()
        api.MCPBvhData = SimpleNamespace(Binary=1)
        api.MCPBvhDisplacement = SimpleNamespace(Enable=1)
        api.MCPBvhRotation = SimpleNamespace(YXZ=1)
        return api

    def test_unavailable_bind_does_not_construct_a_native_application(self):
        api = self.fake_api()
        with mock.patch.dict(sys.modules, {"mocap_api": api}):
            base = load_script("network_test_mcp_base", "mocap_main_base.py")
        with mock.patch.object(base, "preflight_pn_network", side_effect=PNNetworkError("synthetic missing address")):
            with self.assertRaisesRegex(PNNetworkError, "synthetic missing address"):
                base.MCPBase(local_ip="192.168.137.1")
        api.MCPApplication.assert_not_called()
        api.MCPSettings.assert_not_called()

    def test_network_override_reaches_both_sdk_udp_settings(self):
        api = self.fake_api()
        with mock.patch.dict(sys.modules, {"mocap_api": api}):
            base = load_script("network_test_mcp_base_success", "mocap_main_base.py")
        config = PNNetworkConfig(local_ip="192.168.137.1", local_port=18002,
                                 server_ip="192.168.137.250", server_port=18080)
        with mock.patch.object(base, "preflight_pn_network", return_value=config):
            base.MCPBase(local_ip=config.local_ip, local_port=config.local_port,
                         server_ip=config.server_ip, server_port=config.server_port)
        settings = api.MCPSettings.return_value
        settings.SetSettingsUDPEx.assert_called_once_with(config.local_ip, config.local_port)
        settings.SetSettingsUDPServer.assert_called_once_with(config.server_ip, config.server_port)
        api.MCPApplication.return_value.set_settings.assert_called_once_with(settings)
        api.MCPApplication.return_value.open.assert_called_once_with()

    def test_existing_no_argument_subclass_construction_remains_supported(self):
        api = self.fake_api()
        with mock.patch.dict(sys.modules, {"mocap_api": api}):
            base = load_script("network_test_mcp_base_legacy", "mocap_main_base.py")
        with mock.patch.object(base, "preflight_pn_network", return_value=PNNetworkConfig()):
            base.MCPBase()
        settings = api.MCPSettings.return_value
        settings.SetSettingsUDPEx.assert_called_once_with("10.42.0.101", 8002)
        settings.SetSettingsUDPServer.assert_called_once_with("10.42.0.202", 8080)

    def test_sdk_reported_open_failure_is_not_treated_as_connected(self):
        api = self.fake_api()
        api.MCPApplication.return_value.open.return_value = (False, "ServerNotReady")
        with mock.patch.dict(sys.modules, {"mocap_api": api}):
            base = load_script("network_test_mcp_base_open_failure", "mocap_main_base.py")
        with mock.patch.object(base, "preflight_pn_network", return_value=PNNetworkConfig()):
            with self.assertRaises(PNNetworkError) as raised:
                base.MCPBase()
        message = str(raised.exception)
        for expected in ("ServerNotReady", "10.42.0.101:8002", "10.42.0.202:8080"):
            self.assertIn(expected, message)


class PNNetworkBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bridge = load_script("pn_network_test_bridge", "mocap_rp1_gmr_reference.py")

    def test_cli_explicit_endpoints_override_defaults(self):
        args = self.bridge.parse_args(["--pn-local-ip", "192.168.137.1", "--pn-local-port", "18002",
                                       "--pn-server-ip", "192.168.137.250", "--pn-server-port", "18080"])
        self.assertEqual((args.pn_local_ip, args.pn_local_port), ("192.168.137.1", 18002))
        self.assertEqual((args.pn_server_ip, args.pn_server_port), ("192.168.137.250", 18080))

    def test_selected_publish_yaml_supplies_live_network_before_device_startup(self):
        with tempfile.TemporaryDirectory(prefix="pn bridge yaml ") as directory:
            path = Path(directory) / "publish.yaml"
            path.write_text(PNNetworkYAMLTests.PREFIX + (
                "\npn_network:\n"
                "  local_ip: 192.168.137.1\n"
                "  local_port: 18002\n"
                "  server_ip: 192.168.137.250\n"
                "  server_port: 18080\n"))
            with mock.patch.object(pn_network.socket, "socket") as socket_factory, \
                    mock.patch.object(self.bridge, "preflight_pn_network") as preflight, \
                    mock.patch.object(self.bridge, "create_live_source") as source, \
                    mock.patch.dict(os.environ, {"PN_LOCAL_IP": "192.168.31.100",
                                                 "PN_SERVER_IP": "192.168.31.202"}, clear=True):
                args = self.bridge.parse_args(["--publish-config", str(path)])
            self.assertEqual((args.pn_local_ip, args.pn_local_port), ("192.168.137.1", 18002))
            self.assertEqual((args.pn_server_ip, args.pn_server_port), ("192.168.137.250", 18080))
            self.assertEqual(args.publish_config, path)
            socket_factory.assert_not_called()
            preflight.assert_not_called()
            source.assert_not_called()

    def test_explicit_cli_overrides_only_the_selected_yaml_network_field(self):
        with tempfile.TemporaryDirectory(prefix="pn bridge override ") as directory:
            path = Path(directory) / "publish.yaml"
            path.write_text(PNNetworkYAMLTests.PREFIX + (
                "\npn_network:\n"
                "  local_ip: 192.168.137.1\n"
                "  local_port: 18002\n"
                "  server_ip: 192.168.137.250\n"
                "  server_port: 18080\n"))
            args = self.bridge.parse_args(["--publish-config", str(path),
                                           "--pn-local-ip", "192.168.137.10"])
        self.assertEqual((args.pn_local_ip, args.pn_local_port), ("192.168.137.10", 18002))
        self.assertEqual((args.pn_server_ip, args.pn_server_port), ("192.168.137.250", 18080))

    def test_legacy_yaml_without_network_uses_environment_then_old_defaults(self):
        with tempfile.TemporaryDirectory(prefix="pn bridge legacy ") as directory:
            path = Path(directory) / "publish.yaml"
            path.write_text(PNNetworkYAMLTests.PREFIX)
            with mock.patch.dict(os.environ, {}, clear=True):
                args = self.bridge.parse_args(["--publish-config", str(path)])
            self.assertEqual((args.pn_local_ip, args.pn_local_port), ("10.42.0.101", 8002))
            self.assertEqual((args.pn_server_ip, args.pn_server_port), ("10.42.0.202", 8080))
            with mock.patch.dict(os.environ, {"PN_LOCAL_IP": "192.168.31.100",
                                              "PN_SERVER_IP": "192.168.31.202",
                                              "PN_LOCAL_PORT": "28002"}, clear=True):
                args = self.bridge.parse_args(["--publish-config", str(path)])
            self.assertEqual((args.pn_local_ip, args.pn_local_port), ("192.168.31.100", 28002))
            self.assertEqual(args.pn_server_ip, "192.168.31.202")

    def test_bad_live_yaml_fails_before_any_native_or_ros_constructor(self):
        with tempfile.TemporaryDirectory(prefix="pn bridge invalid ") as directory:
            path = Path(directory) / "publish.yaml"
            path.write_text(PNNetworkYAMLTests.PREFIX + "\npn_network:\n  local_ip: invalid-ip\n")
            with mock.patch.object(self.bridge, "ROSPublisher") as ros, \
                    mock.patch.object(self.bridge, "SolverWorker") as worker, \
                    mock.patch.object(self.bridge, "IntegratedViser") as viewer, \
                    mock.patch.object(self.bridge, "create_live_source") as source, \
                    mock.patch.object(pn_network.socket, "socket") as socket_factory, \
                    redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    self.bridge.parse_args(["--publish-config", str(path)])
            self.assertEqual(raised.exception.code, 2)
            for factory in (ros, worker, viewer, source, socket_factory):
                factory.assert_not_called()

    def test_replay_does_not_require_a_valid_live_network_in_yaml(self):
        with tempfile.TemporaryDirectory(prefix="pn replay yaml ") as directory:
            path = Path(directory) / "publish.yaml"
            path.write_text(PNNetworkYAMLTests.PREFIX + (
                "\npn_network:\n"
                "  local_ip: invalid-for-offline-use\n"
                "  local_port: 0\n"
                "  server_ip: also-invalid\n"))
            with mock.patch.object(pn_network.socket, "socket") as socket_factory, \
                    mock.patch.object(self.bridge, "create_live_source") as source:
                args = self.bridge.parse_args(["--source", "replay", "--npz", "unused.npz", "--dry-run",
                                               "--publish-config", str(path)])
            self.assertEqual(args.source, "replay")
            self.assertTrue(args.dry_run)
            socket_factory.assert_not_called()
            source.assert_not_called()

    def test_replay_still_rejects_explicit_invalid_network_cli_arguments(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                self.bridge.parse_args(["--source", "replay", "--npz", "unused.npz", "--dry-run",
                                        "--pn-local-ip", "invalid-ip"])
        self.assertEqual(raised.exception.code, 2)

    def test_invalid_cli_ports_exit_with_argparse_error(self):
        for flag in ("--pn-local-port", "--pn-server-port"):
            for port in ("0", "-1", "65536", "1.5"):
                error = io.StringIO()
                with self.subTest(flag=flag, port=port), redirect_stderr(error):
                    with self.assertRaises(SystemExit) as raised:
                        self.bridge.parse_args([flag, port])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("error:", error.getvalue())

    def test_invalid_live_config_does_not_construct_ros_worker_viewer_or_sdk(self):
        args = self.bridge.parse_args(["--pn-local-ip", "192.168.137.1"])
        with mock.patch.object(self.bridge, "preflight_pn_network", side_effect=PNNetworkError("synthetic bind failure")), \
                mock.patch.object(self.bridge, "ROSPublisher") as ros, \
                mock.patch.object(self.bridge, "SolverWorker") as worker, \
                mock.patch.object(self.bridge, "IntegratedViser") as viewer, \
                mock.patch.object(self.bridge, "create_live_source") as source:
            with self.assertRaisesRegex(PNNetworkError, "synthetic bind failure"):
                asyncio.run(self.bridge.run_bridge(args))
        for factory in (ros, worker, viewer, source):
            factory.assert_not_called()

    def test_live_source_forwards_the_config_without_importing_native_sdk(self):
        api = ModuleType("mocap_api")
        api.MCPAvatar = mock.Mock()
        base_module = ModuleType("mocap_main_base")
        captured = []

        class FakeBase:
            def __init__(self, **kwargs):
                captured.append(kwargs)

        base_module.MCPBase = FakeBase
        config = PNNetworkConfig(local_ip="192.168.137.1", local_port=18002,
                                 server_ip="192.168.137.250", server_port=18080)
        with mock.patch.dict(sys.modules, {"mocap_api": api, "mocap_main_base": base_module}):
            self.bridge.create_live_source(mock.Mock(), mock.Mock(), network=config)
        self.assertEqual(captured, [{"local_ip": config.local_ip, "local_port": config.local_port,
                                     "server_ip": config.server_ip, "server_port": config.server_port}])

    def test_ros_startup_failure_closes_open_pn_without_starting_background_processes(self):
        args = self.bridge.parse_args([])
        source = SimpleNamespace(app=mock.Mock(), update=mock.AsyncMock())
        viewer = SimpleNamespace(status={"enabled": True, "started": False, "failure": None},
                                 start=mock.AsyncMock(), close=mock.AsyncMock())
        with mock.patch.object(self.bridge, "preflight_pn_network", return_value=PNNetworkConfig()), \
                mock.patch.object(self.bridge, "create_live_source", return_value=source) as source_factory, \
                mock.patch.object(self.bridge, "ROSPublisher", side_effect=RuntimeError("synthetic ROS startup failure")), \
                mock.patch.object(self.bridge, "IntegratedViser", return_value=viewer), \
                mock.patch.object(self.bridge.SolverWorker, "run", new_callable=mock.AsyncMock) as worker_run, \
                mock.patch.object(self.bridge, "start_cleanup_watchdog", return_value=None) as watchdog, \
                mock.patch.object(self.bridge, "stop_cleanup_watchdog") as stop_watchdog, \
                mock.patch.object(self.bridge.asyncio, "create_subprocess_exec", new_callable=mock.AsyncMock) as spawn, \
                redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "synthetic ROS startup failure"):
                asyncio.run(self.bridge.run_bridge(args))
        source_factory.assert_called_once()
        source.app.close.assert_called_once_with()
        source.update.assert_not_awaited()
        viewer.start.assert_not_awaited()
        viewer.close.assert_awaited_once_with()
        worker_run.assert_not_awaited()
        spawn.assert_not_awaited()
        watchdog.assert_called_once()
        stop_watchdog.assert_called_once_with(None)
        self.assertIsNone(self.bridge._FINAL_CLEANUP_WATCHDOG)


if __name__ == "__main__":
    unittest.main()
