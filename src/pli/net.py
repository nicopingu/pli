"""HTTP(S) requests for PLI's own calls (PLI 5.6.3).

``urllib.request.urlopen`` follows every scheme it knows, ``file://`` and ``ftp://`` included: a URL taken from an
option or a setting could make PLI read a local file. ``urlopen`` here opens ``http://`` and ``https://`` only, with
the same proxies, redirections and errors as ``urllib.request.urlopen``; any other scheme raises ``URLError``
(« unknown url type »). Tests patch ``pli.net.urlopen``.
"""

from __future__ import annotations

import socket
import urllib.request
from typing import Any

_HANDLERS = (urllib.request.ProxyHandler, urllib.request.UnknownHandler, urllib.request.HTTPHandler,
             urllib.request.HTTPSHandler, urllib.request.HTTPDefaultErrorHandler, urllib.request.HTTPRedirectHandler,
             urllib.request.HTTPErrorProcessor)


def _opener() -> urllib.request.OpenerDirector:
    opener = urllib.request.OpenerDirector()
    for handler in _HANDLERS:
        opener.add_handler(handler())
    return opener


_OPENER = _opener()


def urlopen(request: Any, data: bytes | None = None, timeout: float = socket._GLOBAL_DEFAULT_TIMEOUT) -> Any:  # noqa: SLF001
    """Like ``urllib.request.urlopen``, for ``http://`` and ``https://`` URLs only."""

    return _OPENER.open(request, data, timeout)


__all__ = ["urlopen"]
