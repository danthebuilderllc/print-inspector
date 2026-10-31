"""Network guard: Print Inspector never communicates over the network.

The application has no network code (no HTTP client, no Qt Network module, no
update check, licence check or telemetry). As a second line of defence the
guard installed at start-up refuses every outbound connection and DNS lookup
made through Python's ``socket`` module, logs it and remembers it, so that a
library which unexpectedly tried to go online would fail loudly instead of
silently sending data. Loopback (127.0.0.1, ::1) stays allowed: it never leaves
the computer and Python itself uses it internally on Windows (``socketpair``).

The guard covers Python code only. Native code (Qt, OpenCV, Tesseract) is
covered by excluding Qt's network module from the build and by the operating
system checks described in docs/WINDOWS_DEPLOYMENT.md (firewall rule, no network).
"""
from __future__ import annotations

import ipaddress
import logging
import socket
import threading
from dataclasses import dataclass
from datetime import datetime

log = logging.getLogger(__name__)

_INET_FAMILIES = (socket.AF_INET, socket.AF_INET6)
_LOCAL_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost", ""}


class NetworkBlocked(OSError):
    """An outbound network operation was refused by the offline guard."""


@dataclass(frozen=True)
class BlockedAttempt:
    operation: str  # connect, sendto, getaddrinfo, ...
    target: str
    time: str


_lock = threading.Lock()
_attempts: list[BlockedAttempt] = []
_originals: dict[str, object] = {}


def _is_loopback(host: object) -> bool:
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if not isinstance(host, str):
        return False
    host = host.strip("[]").split("%")[0].lower()
    if host in _LOCAL_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _refuse(operation: str, target: object) -> None:
    attempt = BlockedAttempt(operation, str(target), datetime.now().astimezone().isoformat(timespec="seconds"))
    with _lock:
        _attempts.append(attempt)
    log.warning("Blocked outbound network %s to %s: Print Inspector works offline only", operation, target)
    raise NetworkBlocked(f"Network access is disabled in Print Inspector ({operation} {target})")


def _check_address(sock: socket.socket, operation: str, address: object) -> None:
    if sock.family not in _INET_FAMILIES:
        return  # local IPC (e.g. AF_UNIX) is not network communication
    host = address[0] if isinstance(address, tuple) and address else address
    if not _is_loopback(host):
        _refuse(operation, address)


def _guarded_connect(self: socket.socket, address) -> None:
    _check_address(self, "connect", address)
    return _originals["connect"](self, address)


def _guarded_connect_ex(self: socket.socket, address) -> int:
    _check_address(self, "connect", address)
    return _originals["connect_ex"](self, address)


def _guarded_sendto(self: socket.socket, data, *args):
    _check_address(self, "sendto", args[-1] if args else None)
    return _originals["sendto"](self, data, *args)


def _guarded_getaddrinfo(host, *args, **kwargs):
    if not _is_loopback(host):
        _refuse("name lookup", host)
    return _originals["getaddrinfo"](host, *args, **kwargs)


def _guarded_gethostbyname(host):
    if not _is_loopback(host):
        _refuse("name lookup", host)
    return _originals["gethostbyname"](host)


def install_network_guard() -> None:
    """Refuse outbound connections and DNS lookups from now on (idempotent)."""
    with _lock:
        if _originals:
            return
        _originals.update(
            connect=socket.socket.connect, connect_ex=socket.socket.connect_ex, sendto=socket.socket.sendto,
            getaddrinfo=socket.getaddrinfo, gethostbyname=socket.gethostbyname,
        )
    socket.socket.connect = _guarded_connect
    socket.socket.connect_ex = _guarded_connect_ex
    socket.socket.sendto = _guarded_sendto
    socket.getaddrinfo = _guarded_getaddrinfo
    socket.gethostbyname = _guarded_gethostbyname
    log.info("Offline mode: outbound network access is blocked")


def uninstall_network_guard() -> None:
    """Restore the original socket functions (used by the tests)."""
    with _lock:
        if not _originals:
            return
        originals = dict(_originals)
        _originals.clear()
    socket.socket.connect = originals["connect"]
    socket.socket.connect_ex = originals["connect_ex"]
    socket.socket.sendto = originals["sendto"]
    socket.getaddrinfo = originals["getaddrinfo"]
    socket.gethostbyname = originals["gethostbyname"]


def guard_installed() -> bool:
    return bool(_originals)


def blocked_attempts() -> list[BlockedAttempt]:
    """Every network operation refused since the start (empty in normal operation)."""
    with _lock:
        return list(_attempts)


def clear_blocked_attempts() -> None:
    with _lock:
        _attempts.clear()
