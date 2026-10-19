"""Validate PN UDP endpoints before entering the native SDK."""

from dataclasses import dataclass
import errno
import ipaddress
import os
from pathlib import Path
import re
import shlex
import socket
import tempfile


class PNNetworkError(RuntimeError):
    """An endpoint cannot safely be passed to the PN SDK."""


@dataclass(frozen=True)
class PNNetworkConfig:
    local_ip: str = "10.42.0.101"
    local_port: int = 8002
    server_ip: str = "10.42.0.202"
    server_port: int = 8080


PN_NETWORK_FIELDS = ('local_ip', 'local_port', 'server_ip', 'server_port')


def _port_value(value, key):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise PNNetworkError(f'pn_network.{key} must be an integer port')
    try:
        return int(value)
    except ValueError as exc:
        raise PNNetworkError(f'pn_network.{key} must be an integer port: {value!r}') from exc


def _network_section_span(lines):
    start, end = None, len(lines)
    for index, line in enumerate(lines):
        content = line.split('#', 1)[0].strip()
        if not content or line[:1].isspace():
            continue
        if start is not None and end == len(lines):
            end = index
        key, separator, value = content.partition(':')
        if separator and key == 'pn_network':
            if start is not None:
                raise PNNetworkError('duplicate pn_network YAML section')
            if value.strip():
                raise PNNetworkError('pn_network must be a two-level YAML mapping')
            start = index
    return start, end


def _read_network_lines(lines):
    start, end = _network_section_span(lines)
    if start is None:
        return {}
    values = {}
    for line in lines[start+1:end]:
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        match = re.fullmatch(r' +([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*?)\s*', line.rstrip('\r\n'))
        if not match:
            raise PNNetworkError('pn_network must contain scalar key: value settings')
        key, value = match.groups()
        if key not in PN_NETWORK_FIELDS:
            raise PNNetworkError(f'unknown pn_network setting: {key}')
        if key in values:
            raise PNNetworkError(f'duplicate pn_network setting: {key}')
        try:
            tokens = shlex.split(value, comments=True, posix=True)
        except ValueError as exc:
            raise PNNetworkError(f'invalid pn_network.{key}: {exc}') from exc
        if len(tokens) > 1:
            raise PNNetworkError(f'pn_network.{key} must be a scalar value')
        value = tokens[0] if tokens else ''
        values[key] = _port_value(value, key) if key.endswith('_port') else value
    return values


def read_pn_network_yaml(path):
    """Read only the optional two-level pn_network section; no ROS/SDK/socket."""
    return _read_network_lines(Path(path).read_text(encoding='utf-8').splitlines(keepends=True))


def load_pn_network(path, *, overrides=None, environ=None):
    """Resolve each endpoint: explicit override > YAML > old env > defaults.

    YAML remains authoritative even when an older .env contains PN addresses.
    This function validates values only; the live source separately probes bind.
    """
    overrides = {key: value for key, value in (overrides or {}).items() if value is not None}
    unknown = set(overrides).difference(PN_NETWORK_FIELDS)
    if unknown:
        raise PNNetworkError('unknown PN network override: ' + ', '.join(sorted(unknown)))
    environment = os.environ if environ is None else environ
    configured = read_pn_network_yaml(path)
    defaults = PNNetworkConfig()
    fields = {}
    for key in PN_NETWORK_FIELDS:
        value = overrides.get(key, configured.get(key, environment.get('PN_' + key.upper(), getattr(defaults, key))))
        fields[key] = _port_value(value, key) if key.endswith('_port') else value
    return validate_pn_network(PNNetworkConfig(**fields))


def write_pn_network_yaml(path, updates):
    """Atomically update selected PN endpoints, preserving other YAML bytes."""
    path = Path(path)
    unknown = set(updates).difference(PN_NETWORK_FIELDS)
    if unknown:
        raise PNNetworkError('unknown PN network setting: ' + ', '.join(sorted(unknown)))
    network = load_pn_network(path, overrides=updates, environ={})
    if not updates:
        return
    lines = path.read_bytes().decode('utf-8').splitlines(keepends=True)
    newline = '\r\n' if any(line.endswith('\r\n') for line in lines) else '\n'
    start, end = _network_section_span(lines)
    pending = dict(updates)
    if start is None:
        if lines and not lines[-1].endswith(('\n', '\r')):
            lines[-1] += newline
        lines += [newline, 'pn_network:' + newline]
        end = len(lines)
    else:
        for index in range(start+1, end):
            match = re.match(r'( +)([A-Za-z_][A-Za-z0-9_]*)\s*:', lines[index])
            if match and match.group(2) in pending:
                key = match.group(2)
                comment = '  #' + lines[index].split('#', 1)[1].rstrip('\r\n') if '#' in lines[index] else ''
                lines[index] = f'{match.group(1)}{key}: {getattr(network, key)}{comment}{newline}'
                pending.pop(key)
        if end and not lines[end-1].endswith(('\n', '\r')):
            lines[end-1] += newline
    lines[end:end] = [f'  {key}: {getattr(network, key)}{newline}' for key in pending]
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name + '.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(''.join(lines).encode('utf-8'))
        temporary.chmod(path.stat().st_mode & 0o777)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_pn_network(config: PNNetworkConfig) -> PNNetworkConfig:
    endpoints = {}
    for role in ("local", "server"):
        value = getattr(config, f"{role}_ip")
        try:
            address = ipaddress.IPv4Address(value)
        except (ipaddress.AddressValueError, TypeError) as exc:
            raise PNNetworkError(f"PN {role} IP must be an IPv4 address: {value!r}") from exc
        if address.is_unspecified or address.is_multicast or int(address) == 0xffffffff:
            raise PNNetworkError(f"PN {role} IP must be a concrete unicast IPv4 address: {value!r}")
        port = getattr(config, f"{role}_port")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise PNNetworkError(f"PN {role} port must be between 1 and 65535: {port!r}")
        endpoints[f"{role}_ip"] = str(address)
        endpoints[f"{role}_port"] = port
    if endpoints["local_ip"] == endpoints["server_ip"]:
        raise PNNetworkError("PN local IP and PN-Link server IP must differ; local IP belongs to this PC")
    return PNNetworkConfig(**endpoints)


def preflight_pn_network(config: PNNetworkConfig) -> PNNetworkConfig:
    """Probe only the local UDP bind; no packets or SDK commands are sent."""
    config = validate_pn_network(config)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind((config.local_ip, config.local_port))
    except OSError as exc:
        endpoint = f"{config.local_ip}:{config.local_port}"
        if exc.errno == errno.EADDRNOTAVAIL:
            detail = (f"this PC does not own {config.local_ip}; select its actual PN-facing address "
                      "with --pn-local-ip, or restore the PN network interface configuration")
        elif exc.errno == errno.EADDRINUSE:
            detail = "the UDP port is already in use; close the other PN receiver or choose --pn-local-port"
        else:
            detail = str(exc)
        raise PNNetworkError(f"PN UDP cannot bind local {endpoint}: {detail}. "
                             f"PN-Link target is {config.server_ip}:{config.server_port}.") from exc
    return config
