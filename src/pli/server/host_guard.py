"""PLI: Host and Origin guard.

A PLI server listens on 127.0.0.1 without an API key. Two attacks coming from a browser reach it:
- DNS rebinding: a page points its own name at 127.0.0.1 and reads the replies;
  the request then arrives with a Host that is not a local address;
- the drive-by request: a page sends a "simple" POST to http://127.0.0.1:port; it does not read
  the reply, but the computation starts; the browser adds an Origin header.

The guard refuses (403) a Host missing from the allowed list when the server listens on a local address, and any request
carrying an Origin whose host is not allowed, whatever the listening address. API clients (Claude Code, curl,
Python, a local relay) send no Origin; the relay rewrites Host to 127.0.0.1:port.

Off by default, like every PLI item (no pli.env: TensorFold 0.6.6 behaviour). PLI_HOST_GUARD=1 switches it on;
PLI_ALLOWED_HOSTS="name1,name2" adds allowed names (Host and Origin), besides the local names and the listening address.
"""

from __future__ import annotations

import ipaddress
import os
from urllib.parse import urlsplit

_LOCAL_NAMES = ("localhost", "127.0.0.1", "::1")


def _host(value: str) -> str:
    """The host name of a Host header ("name:port", "[::1]:port"), lowercase, without port or brackets."""

    value = value.strip().lower()
    if value.startswith("["):
        return value[1:].split("]", 1)[0]
    if value.count(":") == 1:
        return value.split(":", 1)[0]
    return value  # without port, or bare IPv6


def _local(name: str) -> bool:
    if name in _LOCAL_NAMES or name.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


class HostGuard:
    def __init__(self, listen: str, allowed: tuple[str, ...] = ()) -> None:
        self.listen = _host(str(listen or ""))
        self.allowed = frozenset(n for n in (_host(a) for a in allowed) if n)
        # Host is checked only on a local listening address: on 0.0.0.0 or a network address, clients call
        # the server by names not known in advance, and DNS rebinding targets only local addresses.
        self.checks_host = _local(self.listen)

    def _allowed(self, name: str) -> bool:
        return bool(name) and (_local(name) or name == self.listen or name in self.allowed)

    def refuse(self, headers) -> str | None:
        """The reason for the refusal, or None if the request passes."""

        if self.checks_host:
            host = headers.get("Host")
            if host is not None and not self._allowed(_host(host)):
                return f"Host {host!r} not allowed"
        origin = headers.get("Origin")
        if origin is not None:
            name = (urlsplit(origin.strip()).hostname or "") if origin.strip().lower() != "null" else ""
            if not self._allowed(name):
                return f"Origin {origin!r} not allowed"
        return None


def configure(args, environ=os.environ) -> HostGuard | None:
    if environ.get("PLI_HOST_GUARD", "") not in ("1", "yes", "on", "oui"):
        return None
    allowed = tuple(n.strip() for n in environ.get("PLI_ALLOWED_HOSTS", "").split(",") if n.strip())
    guard = HostGuard(getattr(args, "host", "127.0.0.1"), allowed)
    print(f"[tensorfold] PLI Host/Origin guard on (Host checked: {'yes' if guard.checks_host else 'no'}"
          f"{', also allowed: ' + ', '.join(sorted(guard.allowed)) if guard.allowed else ''})", flush=True)
    return guard
