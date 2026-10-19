"""User-supplied SDK loading and compatibility, entirely with fake modules.

No native PN library, socket, hardware, ROS node or real terminal is opened.
"""
import asyncio
from contextlib import redirect_stdout
import io
from pathlib import Path
import runpy
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import pn_sdk
from pn_network import PNNetworkConfig, PNNetworkError


class SDKTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.module_patch = mock.patch.dict(sys.modules)
        self.module_patch.start()
        for name in pn_sdk._NAMES:
            sys.modules.pop(name, None)
        self.env_patch = mock.patch.dict(pn_sdk.os.environ, {}, clear=False)
        self.env_patch.start()
        pn_sdk.os.environ.pop("PN_RP1_SDK_DIR", None)
        self.local_patch = mock.patch.object(pn_sdk, "_LOCAL_DIR", self.directory / "absent")
        self.local_patch.start()
        pn_sdk._compatible_base.cache_clear()

    def tearDown(self):
        pn_sdk._compatible_base.cache_clear()
        self.local_patch.stop()
        self.env_patch.stop()
        self.module_patch.stop()
        self.temporary.cleanup()

    def write_sdk(self, directory, tag):
        directory.mkdir()
        (directory / "mocap_api.py").write_text(f"TAG = {tag!r}\n")
        (directory / "mocap_main_base.py").write_text(
            "from mocap_api import TAG\nclass MCPBase:\n"
            "    def __init__(self, *, local_ip=None, local_port=None, server_ip=None, server_port=None):\n"
            "        self.tag = TAG\n")

    def fake_legacy_sdk(self):
        api = ModuleType("mocap_api")
        api.MCPApplication = mock.Mock()
        api.MCPApplication.return_value.open.return_value = (True, "NoError")
        api.MCPApplication.return_value.poll_next_event.return_value = []
        api.MCPSettings = mock.Mock()
        api.MCPBvhData = SimpleNamespace(Binary=1)
        api.MCPBvhDisplacement = SimpleNamespace(Enable=2)
        api.MCPBvhRotation = SimpleNamespace(YXZ=3)
        api.MCPEventType = SimpleNamespace(AvatarUpdated=10, Notify=11, CommandReply=12, RigidBodyUpdated=13)
        api.MCPReplay = SimpleNamespace(MCPReplay_Response=20, MCPReplay_Running=21, MCPReplay_Result=22)
        api.EMCPCommand = SimpleNamespace(**{name: index for index, name in enumerate((
            "CommandStartCapture", "CommandStopCapture", "CommandCalibrateMotion",
            "CommandResumeOriginalHandsPosture", "CommandClearZeroMotionDrift",
            "CommandResumeOriginalPosture", "CommandZeroPosition"))})
        vendor_init = mock.Mock(side_effect=AssertionError("unsafe vendor initializer called"))

        class VendorBase:
            def __init__(self):
                vendor_init()

            def running_command(self, command):
                self.commands.append(command)

            def handleAvatar(self, handle):
                self.avatars.append(handle)

            def handleNotify(self, value):
                self.notifications.append(value)

            def handleRunning(self, value):
                self.running.append(value)

            def handleResult(self, value):
                self.results.append(value)
                self.current_command = -1

        module = ModuleType("mocap_main_base")
        module.MCPBase = VendorBase
        sys.modules.update(mocap_api=api, mocap_main_base=module)
        return api, VendorBase, vendor_init

    def instance(self):
        api, original, vendor_init = self.fake_legacy_sdk()
        _, adapted = pn_sdk.load_sdk()
        config = PNNetworkConfig("192.168.50.2", 18002, "192.168.50.3", 18080)
        with mock.patch.object(pn_sdk, "preflight_pn_network", return_value=config) as preflight:
            instance = adapted(local_ip=config.local_ip, local_port=config.local_port,
                               server_ip=config.server_ip, server_port=config.server_port)
        preflight.assert_called_once_with(config)
        instance.commands, instance.avatars, instance.notifications = [], [], []
        instance.running, instance.results = [], []
        return instance, api, original, vendor_init


class SDKLoaderTests(SDKTestCase):
    def test_import_is_lazy_even_when_sdk_imports_would_raise(self):
        original_import = __import__

        def guarded_import(name, *args, **kwargs):
            if name in pn_sdk._NAMES:
                raise AssertionError("SDK imported during module load")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guarded_import):
            runpy.run_path(pn_sdk.__file__, run_name="pn_sdk_lazy_test")
        for name in pn_sdk._NAMES:
            self.assertNotIn(name, sys.modules)

    def test_existing_network_aware_class_is_returned_unchanged(self):
        api = ModuleType("mocap_api")

        class ExistingBase:
            def __init__(self, *, local_ip=None, local_port=None, server_ip=None, server_port=None):
                self.received = (local_ip, local_port, server_ip, server_port)

        module = ModuleType("mocap_main_base")
        module.MCPBase = ExistingBase
        sys.modules.update(mocap_api=api, mocap_main_base=module)
        with mock.patch.object(pn_sdk, "_sdk_directory") as find, mock.patch.object(pn_sdk, "preflight_pn_network") as preflight:
            returned_api, base = pn_sdk.load_sdk()
            self.assertIs(returned_api, api)
            self.assertIs(base, ExistingBase)
            self.assertEqual(base(local_ip="selected").received[0], "selected")
        find.assert_not_called()
        preflight.assert_not_called()

    def test_local_pair_takes_precedence_over_explicit_external_pair(self):
        local, external = self.directory / "local", self.directory / "external"
        self.write_sdk(local, "local")
        self.write_sdk(external, "external")
        with mock.patch.object(pn_sdk, "_LOCAL_DIR", local), mock.patch.dict(pn_sdk.os.environ, {"PN_RP1_SDK_DIR": str(external)}):
            api, base = pn_sdk.load_sdk()
        self.assertEqual(api.TAG, "local")
        self.assertEqual(base().tag, "local")

    def test_external_directory_loads_both_modules_and_restores_sys_path(self):
        external = self.directory / "external"
        self.write_sdk(external, "external")
        original_path = sys.path[:]
        with mock.patch.dict(pn_sdk.os.environ, {"PN_RP1_SDK_DIR": str(external)}):
            api, base = pn_sdk.load_sdk()
        self.assertEqual(api.TAG, "external")
        self.assertEqual(base().tag, "external")
        self.assertEqual(sys.path, original_path)

    def test_missing_sdk_explains_external_install_and_offline_alternative(self):
        with self.assertRaises(pn_sdk.PNSDKError) as raised:
            pn_sdk.load_sdk()
        for phrase in ("PN_RP1_SDK_DIR", "mocap_api.py", "mocap_main_base.py", "Offline"):
            self.assertIn(phrase, str(raised.exception))

    def test_explicit_incomplete_directory_is_rejected_without_searching(self):
        external = self.directory / "external"
        external.mkdir()
        (external / "mocap_api.py").write_text("raise AssertionError('must not import')")
        with mock.patch.dict(pn_sdk.os.environ, {"PN_RP1_SDK_DIR": str(external)}):
            with self.assertRaisesRegex(pn_sdk.PNSDKError, "both mocap_api.py and mocap_main_base.py"):
                pn_sdk.load_sdk()

    def test_native_import_error_is_friendly_and_rolls_back_partial_modules(self):
        external = self.directory / "external"
        self.write_sdk(external, "external")
        (external / "mocap_main_base.py").write_text("raise OSError('synthetic shared library unavailable')\n")
        original_path = sys.path[:]
        with mock.patch.dict(pn_sdk.os.environ, {"PN_RP1_SDK_DIR": str(external)}):
            with self.assertRaises(pn_sdk.PNSDKError) as raised:
                pn_sdk.load_sdk()
        self.assertIn("lib libraries", str(raised.exception))
        self.assertIn("synthetic shared library unavailable", str(raised.exception))
        self.assertEqual(sys.path, original_path)
        for name in pn_sdk._NAMES:
            self.assertNotIn(name, sys.modules)


class SDKAdapterTests(SDKTestCase):
    def test_preflight_and_explicit_endpoints_precede_sdk_open_without_vendor_init(self):
        instance, api, original, unsafe_init = self.instance()
        self.assertIsInstance(instance, original)
        unsafe_init.assert_not_called()
        settings = api.MCPSettings.return_value
        settings.SetSettingsUDPEx.assert_called_once_with("192.168.50.2", 18002)
        settings.SetSettingsUDPServer.assert_called_once_with("192.168.50.3", 18080)
        settings.set_bvh_rotation.assert_called_once_with(api.MCPBvhRotation.YXZ)
        api.MCPApplication.return_value.set_settings.assert_called_once_with(settings)
        api.MCPApplication.return_value.open.assert_called_once_with()
        self.assertEqual((instance.current_command, instance.capture_key, instance.connent_key), (-1, False, False))

    def test_failed_network_preflight_never_constructs_sdk_objects(self):
        api, _, unsafe_init = self.fake_legacy_sdk()
        _, base = pn_sdk.load_sdk()
        with mock.patch.object(pn_sdk, "preflight_pn_network", side_effect=PNNetworkError("synthetic bind failure")):
            with self.assertRaisesRegex(PNNetworkError, "synthetic bind failure"):
                base()
        api.MCPApplication.assert_not_called()
        api.MCPSettings.assert_not_called()
        unsafe_init.assert_not_called()

    def test_sdk_open_failure_closes_created_application_and_reports_endpoints(self):
        api, _, _ = self.fake_legacy_sdk()
        api.MCPApplication.return_value.open.return_value = (False, "synthetic failure")
        _, base = pn_sdk.load_sdk()
        with mock.patch.object(pn_sdk, "preflight_pn_network", return_value=PNNetworkConfig()):
            with self.assertRaisesRegex(PNNetworkError, "synthetic failure"):
                base()
        api.MCPApplication.return_value.close.assert_called_once_with()

    def test_keyboard_mapping_calls_inherited_command_handler(self):
        instance, api, _, _ = self.instance()
        for key in ("N", "f", "c", "r", "0", "o", "z"):
            self.assertTrue(instance.dispatch_key(key))
        self.assertEqual(instance.commands, list(range(7)))
        self.assertFalse(instance.dispatch_key("a"))
        self.assertFalse(instance.dispatch_key("x"))

    def test_poll_keeps_latest_avatar_and_dispatches_vendor_control_handlers(self):
        instance, api, _, _ = self.instance()
        running = SimpleNamespace(_replay=api.MCPReplay.MCPReplay_Running)
        result = SimpleNamespace(_replay=api.MCPReplay.MCPReplay_Result)
        notification = object()
        event = lambda kind, **kwargs: SimpleNamespace(event_type=kind, event_data=SimpleNamespace(**kwargs))
        instance.app.poll_next_event.return_value = [
            event(api.MCPEventType.AvatarUpdated, avatar_handle=101),
            event(api.MCPEventType.Notify, notifyData=notification),
            event(api.MCPEventType.CommandReply, commandRespond=running),
            event(api.MCPEventType.AvatarUpdated, avatar_handle=102),
            event(api.MCPEventType.CommandReply, commandRespond=result),
        ]
        instance._poll_latest()
        self.assertEqual(instance.avatars, [102])
        self.assertEqual(instance.notifications, [notification])
        self.assertEqual(instance.running, [running])
        self.assertEqual(instance.results, [result])
        self.assertTrue(instance.capture_key)
        self.assertTrue(instance.connent_key)

    def test_calibrating_or_pending_commands_suppress_avatar_frames(self):
        instance, api, _, _ = self.instance()
        instance.current_command = api.EMCPCommand.CommandCalibrateMotion
        instance.app.poll_next_event.return_value = [SimpleNamespace(
            event_type=api.MCPEventType.AvatarUpdated, event_data=SimpleNamespace(avatar_handle=101))]
        instance._poll_latest()
        self.assertEqual(instance.avatars, [])
        self.assertFalse(instance.capture_key)

    def test_update_uses_100hz_deadlines_and_yields_after_overrun(self):
        instance, _, _, _ = self.instance()
        clock = SimpleNamespace(now=0.)
        sleeps = []

        def poll():
            if not sleeps:
                clock.now += .015

        async def sleep(delay):
            sleeps.append(delay)
            clock.now += delay
            if len(sleeps) == 2:
                raise asyncio.CancelledError

        with mock.patch.object(instance, "_poll_latest", side_effect=poll), \
                mock.patch.object(pn_sdk.asyncio, "get_running_loop", return_value=SimpleNamespace(time=lambda: clock.now)), \
                mock.patch.object(pn_sdk.asyncio, "sleep", side_effect=sleep):
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(instance.update())
        self.assertEqual(instance.POLL_FREQUENCY_HZ, 100.)
        self.assertEqual(sleeps[0], 0.)
        self.assertAlmostEqual(sleeps[1], .01)

    def test_standalone_terminal_uses_derived_dispatch_and_cleans_up(self):
        instance, _, _, _ = self.instance()
        selected_keys = []
        instance.dispatch_key = selected_keys.append

        async def run():
            loop = asyncio.get_running_loop()

            def add_reader(fd, callback):
                loop.call_soon(callback)
                loop.call_soon(callback)

            with mock.patch.object(loop, "add_reader", side_effect=add_reader), \
                    mock.patch.object(loop, "remove_reader") as remove, \
                    mock.patch.object(pn_sdk.sys, "stdin", SimpleNamespace(isatty=lambda: True, fileno=lambda: 77)), \
                    mock.patch.object(pn_sdk.os, "read", side_effect=[b"A", b"\x1b"]), \
                    mock.patch("termios.tcgetattr", return_value="original"), \
                    mock.patch("termios.tcsetattr") as restore, mock.patch("tty.setcbreak"):
                await instance.main_async()
            remove.assert_called_once_with(77)
            restore.assert_called_once()

        asyncio.run(run())
        self.assertEqual(selected_keys, ["a"])
        instance.app.close.assert_called_once_with()

    def test_non_terminal_update_failure_propagates_and_closes_sdk(self):
        instance, _, _, _ = self.instance()
        instance.update = mock.AsyncMock(side_effect=RuntimeError("synthetic polling failure"))
        with mock.patch.object(pn_sdk.sys, "stdin", SimpleNamespace(isatty=lambda: False)), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "synthetic polling failure"):
                asyncio.run(instance.main_async())
        instance.app.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
