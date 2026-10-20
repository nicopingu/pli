"""PLI (lot de patch 20261016-220013-4533) : garde Host/Origin devant les deux serveurs HTTP."""

from contextlib import contextmanager
import http.client
import json
import threading
from types import SimpleNamespace

import pytest

from tensorfold.server import host_guard
from tensorfold.server.host_guard import HostGuard
from tensorfold.server.http import Server, make_handler as mlx_handler


def test_eteinte_par_defaut():
    assert host_guard.configure(SimpleNamespace(host="127.0.0.1"), environ={}) is None
    assert host_guard.configure(SimpleNamespace(host="127.0.0.1"), environ={"PLI_GARDE_HOTE": "0"}) is None


def test_allumee_et_noms_admis():
    garde = host_guard.configure(SimpleNamespace(host="127.0.0.1"),
                                 environ={"PLI_GARDE_HOTE": "1", "PLI_HOTES_ADMIS": "client.local, SERVEUR"})
    assert garde.verifie_host and garde.admis == {"client.local", "serveur"}


@pytest.mark.parametrize("host", ["127.0.0.1:8033", "localhost:8033", "LOCALHOST", "[::1]:8033", "::1",
                                  "127.0.0.2", "app.localhost:9"])
def test_host_local_admis(host):
    assert HostGuard("127.0.0.1").refuse({"Host": host}) is None


@pytest.mark.parametrize("host", ["evil.example:8033", "evil.example", "192.168.1.50:8033", "serveur.local"])
def test_host_etranger_refuse_sur_ecoute_locale(host):
    assert HostGuard("127.0.0.1").refuse({"Host": host})


def test_host_admis_en_plus():
    assert HostGuard("127.0.0.1", ("serveur.local",)).refuse({"Host": "serveur.local:8033"}) is None


def test_host_libre_sur_ecoute_reseau():
    garde = HostGuard("0.0.0.0")
    assert not garde.verifie_host
    assert garde.refuse({"Host": "serveur.local:8033"}) is None


def test_host_absent_admis():
    assert HostGuard("127.0.0.1").refuse({}) is None


@pytest.mark.parametrize("ecoute", ["127.0.0.1", "0.0.0.0"])
@pytest.mark.parametrize("origin", ["https://evil.example", "http://192.168.1.10:3000", "null", "file://"])
def test_origin_etranger_refuse(ecoute, origin):
    assert HostGuard(ecoute).refuse({"Host": "127.0.0.1:8033", "Origin": origin})


@pytest.mark.parametrize("origin", ["http://127.0.0.1:8033", "http://localhost:3000", "http://[::1]:8033"])
def test_origin_local_admis(origin):
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
def test_refus_avant_toute_route(backend, headers):
    with serving(backend, HostGuard("127.0.0.1")) as port:
        code, body = request(port, "/v1/chat/completions", headers, method="POST", body=b'{"messages": []}')
    assert code == 403
    assert json.loads(body)["error"]["code"] == "forbidden_origin"


@pytest.mark.parametrize("backend", ["mlx"])
def test_client_local_passe(backend):
    with serving(backend, HostGuard("127.0.0.1")) as port:
        code, body = request(port, "/health", {"Host": "127.0.0.1"})
    assert code == 200 and json.loads(body)


@pytest.mark.parametrize("backend", ["mlx"])
def test_sans_garde_rien_ne_change(backend):
    with serving(backend, None) as port:
        code, _ = request(port, "/health", {"Host": "evil.example", "Origin": "https://evil.example"})
    assert code == 200
