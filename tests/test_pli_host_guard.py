"""PLI (patch batch 20261016-220013-4533): Host/Origin guard in front of both HTTP servers."""

from contextlib import contextmanager
import http.client
import json
import threading
from types import SimpleNamespace

import pytest

from tensorfold.server import host_guard
from tensorfold.server.host_guard import HostGuard
from tensorfold.server.http import Server, make_handler as mlx_handler


def test_off_by_default():
    assert host_guard.configure(SimpleNamespace(host="127.0.0.1"), environ={}) is None
    assert host_guard.configure(SimpleNamespace(host="127.0.0.1"), environ={"PLI_HOST_GUARD": "0"}) is None


def test_on_and_allowed_names():
    guard = host_guard.configure(SimpleNamespace(host="127.0.0.1"),
                                 environ={"PLI_HOST_GUARD": "1", "PLI_ALLOWED_HOSTS": "client.local, SERVER"})
    assert guard.checks_host and guard.allowed == {"client.local", "server"}


@pytest.mark.parametrize("host", ["127.0.0.1:8033", "localhost:8033", "LOCALHOST", "[::1]:8033", "::1",
                                  "127.0.0.2", "app.localhost:9"])
def test_local_host_allowed(host):
    assert HostGuard("127.0.0.1").refuse({"Host": host}) is None


@pytest.mark.parametrize("host", ["evil.example:8033", "evil.example", "192.168.1.50:8033", "server.local"])
def test_foreign_host_refused_on_local_listen(host):
    assert HostGuard("127.0.0.1").refuse({"Host": host})


def test_extra_allowed_host():
    assert HostGuard("127.0.0.1", ("server.local",)).refuse({"Host": "server.local:8033"}) is None


def test_host_free_on_network_listen():
    guard = HostGuard("0.0.0.0")
    assert not guard.checks_host
    assert guard.refuse({"Host": "server.local:8033"}) is None


def test_missing_host_allowed():
    assert HostGuard("127.0.0.1").refuse({}) is None


@pytest.mark.parametrize("listen", ["127.0.0.1", "0.0.0.0"])
@pytest.mark.parametrize("origin", ["https://evil.example", "http://192.168.1.10:3000", "null", "file://"])
def test_foreign_origin_refused(listen, origin):
    assert HostGuard(listen).refuse({"Host": "127.0.0.1:8033", "Origin": origin})


@pytest.mark.parametrize("origin", ["http://127.0.0.1:8033", "http://localhost:3000", "http://[::1]:8033"])
def test_local_origin_allowed(origin):
    assert HostGuard("127.0.0.1").refuse({"Host": "127.0.0.1:8033", "Origin": origin}) is None


@contextmanager
def serving(backend, guard):
    app = SimpleNamespace(served="fixture", served_name="fixture", model_ids=["fixture"], max_batch_size=1)
    app.auth = None
    app.host_guard = guard
    server = Server(("127.0.0.1", 0), mlx_handler(app))
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
        assert not thread.is_alive()


def request(port, path, headers=None, method="GET", body=None):
    client = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        client.putrequest(method, path, skip_host=True)
        for name, value in (headers or {}).items():
            client.putheader(name, value)
        if body is not None:
            client.putheader("Content-Length", str(len(body)))
        client.endheaders(body)
        response = client.getresponse()
        return response.status, response.read().decode()
    finally:
        client.close()


@pytest.mark.parametrize("backend", ["mlx"])
@pytest.mark.parametrize("headers", [{"Host": "evil.example:1234"},
                                     {"Host": "127.0.0.1", "Origin": "https://evil.example"}])
def test_refused_before_any_route(backend, headers):
    with serving(backend, HostGuard("127.0.0.1")) as port:
        code, body = request(port, "/v1/chat/completions", headers, method="POST", body=b'{"messages": []}')
    assert code == 403
    assert json.loads(body)["error"]["code"] == "forbidden_origin"


@pytest.mark.parametrize("backend", ["mlx"])
def test_local_client_passes(backend):
    with serving(backend, HostGuard("127.0.0.1")) as port:
        code, body = request(port, "/health", {"Host": "127.0.0.1"})
    assert code == 200 and json.loads(body)


@pytest.mark.parametrize("backend", ["mlx"])
def test_without_guard_nothing_changes(backend):
    with serving(backend, None) as port:
        code, _ = request(port, "/health", {"Host": "evil.example", "Origin": "https://evil.example"})
    assert code == 200
