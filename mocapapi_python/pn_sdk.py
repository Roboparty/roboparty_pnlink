"""Lazy access to a user-supplied PN SDK, without redistributing SDK files.

Importing this module is safe for --help and offline tools. ``load_sdk`` is
called only for live input. An existing local SDK pair takes precedence over
PN_RP1_SDK_DIR; no home-directory search, download or connection is performed.
"""
from __future__ import annotations

import asyncio
from contextlib import suppress
from functools import lru_cache
import importlib.util
import inspect
import os
from pathlib import Path
import sys

from pn_network import PNNetworkConfig, PNNetworkError, preflight_pn_network

_LOCAL_DIR = Path(__file__).resolve().parent
_NAMES = ("mocap_api", "mocap_main_base")


class PNSDKError(RuntimeError):
    """The user-supplied SDK or its native dependency could not be loaded."""


def _sdk_directory():
    if all((_LOCAL_DIR / (name + ".py")).is_file() for name in _NAMES):
        return _LOCAL_DIR
    configured = os.environ.get("PN_RP1_SDK_DIR")
    if configured:
        directory = Path(configured).expanduser().resolve()
        if all((directory / (name + ".py")).is_file() for name in _NAMES):
            return directory
        raise PNSDKError(
            "PN_RP1_SDK_DIR must name the SDK sample directory containing "
            f"both mocap_api.py and mocap_main_base.py: {directory}")
    raise PNSDKError(
        "PN SDK is required only for live input. Install the official SDK sample "
        "and set PN_RP1_SDK_DIR to the directory containing mocap_api.py, "
        "mocap_main_base.py and its lib/ libraries. Offline replay needs no SDK.")


def _import_sdk_module(name, directory):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, directory / (name + ".py"))
    if spec is None or spec.loader is None:
        raise PNSDKError(f"Cannot load SDK module {name} from {directory}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _accepts_network_kwargs(base):
    try:
        parameters = inspect.signature(base.__init__).parameters
    except (TypeError, ValueError):
        return False
    if any(p.kind == p.VAR_KEYWORD for p in parameters.values()):
        return True
    return all(name in parameters and parameters[name].kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        for name in ("local_ip", "local_port", "server_ip", "server_port"))


def load_sdk():
    """Return ``(mocap_api_module, network-compatible MCPBase class)``.

    Already imported modules and the existing local SDK remain unchanged.
    Loading the SDK imports its library bindings; constructing the returned
    base is the step that performs network preflight and opens live input.
    """
    if all(name in sys.modules for name in _NAMES):
        api, base_module = (sys.modules[name] for name in _NAMES)
    else:
        directory = _sdk_directory()
        previous = {name: sys.modules.get(name) for name in _NAMES}
        original_path = sys.path[:]
        try:
            sys.path.insert(0, str(directory))
            api = _import_sdk_module("mocap_api", directory)
            base_module = _import_sdk_module("mocap_main_base", directory)
        except Exception as exc:
            for name, old in previous.items():
                if old is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old
            raise PNSDKError(
                f"Cannot import PN SDK from {directory}: {exc}. Check SDK "
                "architecture/lib libraries and install the SDK sample's Python "
                "dependencies (docutils, pynput). For a headless sample import, "
                "PYNPUT_BACKEND=dummy can be used; this adapter reads terminal keys.") from exc
        finally:
            sys.path[:] = original_path
    base = getattr(base_module, "MCPBase", None)
    if not isinstance(base, type):
        raise PNSDKError("mocap_main_base.py must export the official MCPBase class")
    return api, base if _accepts_network_kwargs(base) else _compatible_base(api, base)


@lru_cache(maxsize=8)
def _compatible_base(api, vendor_base):
    """Small public-API adapter; vendor command handlers remain inherited."""
    class NetworkMCPBase(vendor_base):
        POLL_FREQUENCY_HZ = 100.0

        def __init__(self, *, local_ip="10.42.0.101", local_port=8002,
                     server_ip="10.42.0.202", server_port=8080):
            # Never call a vendor initializer that may bind a hardcoded address.
            self.network = preflight_pn_network(PNNetworkConfig(
                local_ip=local_ip, local_port=local_port,
                server_ip=server_ip, server_port=server_port))
            self.current_command = -1
            self.capture_key = False
            self.connent_key = False  # Public vendor sample spelling.
            self._master_info_printed = False
            self.app = api.MCPApplication()
            try:
                settings = api.MCPSettings()
                settings.set_bvh_data(api.MCPBvhData.Binary)
                settings.set_bvh_transformation(api.MCPBvhDisplacement.Enable)
                settings.set_bvh_rotation(api.MCPBvhRotation.YXZ)
                settings.SetSettingsUDPEx(self.network.local_ip, self.network.local_port)
                settings.SetSettingsUDPServer(self.network.server_ip, self.network.server_port)
                self.app.set_settings(settings)
                opened, error = self.app.open()
                if not opened:
                    raise PNNetworkError(
                        f"PN SDK open failed: {error}; local="
                        f"{self.network.local_ip}:{self.network.local_port}, target="
                        f"{self.network.server_ip}:{self.network.server_port}")
            except Exception:
                with suppress(Exception):
                    self.app.close()
                raise

        def dispatch_key(self, key):
            commands = {
                "n": "CommandStartCapture", "f": "CommandStopCapture",
                "c": "CommandCalibrateMotion", "r": "CommandResumeOriginalHandsPosture",
                "0": "CommandClearZeroMotionDrift", "o": "CommandResumeOriginalPosture",
                "z": "CommandZeroPosition",
            }
            name = commands.get(key.lower())
            if name is None:
                return False
            self.running_command(getattr(api.EMCPCommand, name))
            return True

        def _poll_latest(self):
            latest_avatar = None
            for event in self.app.poll_next_event():
                self.connent_key = True
                kind, data = event.event_type, event.event_data
                if kind == api.MCPEventType.AvatarUpdated:
                    if self.current_command == -1:
                        latest_avatar = data.avatar_handle
                elif kind == api.MCPEventType.Notify:
                    self.handleNotify(data.notifyData)
                elif kind == api.MCPEventType.CommandReply:
                    response = data.commandRespond
                    if response._replay == api.MCPReplay.MCPReplay_Running:
                        self.handleRunning(response)
                    elif response._replay == api.MCPReplay.MCPReplay_Result:
                        self.handleResult(response)
            if latest_avatar is not None and self.current_command == -1:
                self.capture_key = True
                self.handleAvatar(latest_avatar)

        async def update(self):
            loop = asyncio.get_running_loop()
            deadline = loop.time()
            period = 1.0 / self.POLL_FREQUENCY_HZ
            while True:
                self._poll_latest()
                deadline += period
                delay = deadline - loop.time()
                if delay < 0:
                    deadline = loop.time()
                    delay = 0.0
                await asyncio.sleep(delay)

        async def main_async(self):
            """Focused-terminal lifecycle used by the legacy/recorder entries."""
            loop = asyncio.get_running_loop()
            stop = asyncio.Event()
            update_task = asyncio.create_task(self.update())
            update_task.add_done_callback(lambda _: stop.set())
            fd, terminal_state = None, None
            try:
                if sys.stdin.isatty():
                    import termios
                    import tty
                    fd = sys.stdin.fileno()
                    terminal_state = termios.tcgetattr(fd)
                    tty.setcbreak(fd)

                    def read_keys():
                        data = os.read(fd, 16)
                        if not data or data == b"\x1b":
                            stop.set()
                        elif not data.startswith(b"\x1b"):
                            for code in data:
                                if code not in (10, 13):
                                    try:
                                        self.dispatch_key(chr(code).lower())
                                    except Exception as exc:
                                        print(f"PN command failed: {exc}", file=sys.stderr)
                    loop.add_reader(fd, read_keys)
                else:
                    print("stdin is not a terminal; PN keyboard commands are disabled")
                await stop.wait()
                if update_task.done():
                    update_task.result()
            finally:
                try:
                    if fd is not None and terminal_state is not None:
                        loop.remove_reader(fd)
                        termios.tcsetattr(fd, termios.TCSADRAIN, terminal_state)
                finally:
                    update_task.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await update_task
                    with suppress(Exception):
                        self.app.close()

    NetworkMCPBase.__name__ = "PNNetworkMCPBase"
    return NetworkMCPBase
