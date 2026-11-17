"""HTTP server generation routes (PLI 5, step 1): behind PLI_GEN=1, simulated child processes."""

from __future__ import annotations

import http.client
import json
import socket
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import pli
from pli.server.authentication import KeyStore
from pli.server.http import Server, make_handler

SRC = str(Path(pli.__file__).resolve().parent.parent)


def port_libre() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def serveur(app):
    srv = Server(("127.0.0.1", 0), make_handler(app))
    fil = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    fil.start()
    try:
        yield srv.server_port
    finally:
        srv.shutdown()
        srv.server_close()
        fil.join(2)
        gen = getattr(app, "gen", None)
        if gen is not None and hasattr(gen, "arreter"):
            gen.arreter()


def demande(port, chemin, corps=b"{}", entetes=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    try:
        c.request("POST", chemin, body=corps, headers={"Content-Type": "application/json", **(entetes or {})})
        r = c.getresponse()
        return r.status, r.getheader("Content-Type"), r.read()
    finally:
        c.close()


def appli(**plus):
    return SimpleNamespace(served="f", served_name="f", model_ids=["f"], max_batch_size=1, auth=None, **plus)


@pytest.fixture
def config(tmp_path, monkeypatch):
    def fam(nom):
        return {"argv": ["{python}", "-m", "pli.gen.fils_simule", "--port", "{port}"], "port": port_libre(),
                "memoire_go": 0, "sante_delai_s": 20}
    chemin = tmp_path / "gen.json"
    chemin.write_text(json.dumps({"reserve_go": 0, "familles": {"audio": fam("audio"), "image": fam("image")}}))
    monkeypatch.setenv("PLI_GEN_CONFIG", str(chemin))
    monkeypatch.setenv("PLI_GEN_RESERVE_GO", "0")
    monkeypatch.setenv("PYTHONPATH", SRC)
    return chemin


def test_sans_pli_gen_les_routes_de_generation_sont_inconnues(monkeypatch, config):
    monkeypatch.delenv("PLI_GEN", raising=False)
    with serveur(appli()) as port:
        for chemin in ("/v1/images/generations", "/v1/audio/speech", "/v1/audio/transcriptions", "/v1/images/edits"):
            code, _, corps = demande(port, chemin)
            assert code == 404 and json.loads(corps) == {"error": {"message": f"unknown path {chemin}"}}


def test_avec_pli_gen_les_quatre_routes_arrivent_au_fils(monkeypatch, config):
    monkeypatch.setenv("PLI_GEN", "1")
    with serveur(appli()) as port:
        code, _, corps = demande(port, "/v1/images/generations", b'{"prompt": "chat"}')
        assert code == 200 and json.loads(corps)["data"][0]["b64_json"]
        assert demande(port, "/v1/images/edits")[0] == 200
        code, type_, corps = demande(port, "/v1/audio/speech")
        assert code == 200 and type_ == "audio/wav" and corps.startswith(b"RIFF")
        assert json.loads(demande(port, "/v1/audio/transcriptions")[2]) == {"text": "simulated text"}


def test_meme_authentification_que_les_autres_routes_v1(monkeypatch, config):
    monkeypatch.setenv("PLI_GEN", "1")
    app = appli()
    app.auth = KeyStore(["cle-test"])
    with serveur(app) as port:
        code, _, corps = demande(port, "/v1/images/generations")
        assert code == 401 and json.loads(corps)["error"]["code"] == "invalid_api_key"
        assert demande(port, "/v1/images/generations", entetes={"Authorization": "Bearer cle-test"})[0] == 200
        # without /v1 the route is not routed: it therefore does not bypass the guard
        assert demande(port, "/images/generations", entetes={"Authorization": "Bearer cle-test"})[0] == 404


def test_les_demandes_du_modele_general_posent_et_levent_le_drapeau(monkeypatch, config):
    monkeypatch.setenv("PLI_GEN", "1")
    appels = []
    gen = SimpleNamespace(general_debut=lambda: appels.append("debut"), general_fin=lambda: appels.append("fin"))
    with serveur(appli(gen=gen)) as port:
        demande(port, "/v1/chat/completions", b"{}")       # refused by the general model, the flag turned anyway
        assert appels == ["debut", "fin"]
        demande(port, "/v1/messages", b"{}")
        assert appels == ["debut", "fin", "debut", "fin"]
