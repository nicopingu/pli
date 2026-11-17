"""PLI 5.4.1 (11/15/2026): placement computed on the requested model, inactive families stopped before launch (including audio,
sauf réserve vidéo-voix active), nouvel essai après un 503 « memoire » du fils ; attente bornée du 27B au démarrage."""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path

import pytest

import pli
from pli.gen.superviseur import GO, Famille, Superviseur

SRC = str(Path(pli.__file__).resolve().parent.parent)
JSON = {"Content-Type": "application/json"}


def port_libre() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def attendre(condition, delai=10.0):
    fin = time.monotonic() + delai
    while time.monotonic() < fin:
        if condition():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def catalogues(tmp_path):
    image = tmp_path / "catalogue-image.json"
    image.write_text(json.dumps({"defaut": "qwen21", "modeles": {"qwen21": {"classe": "qwen21", "memoire_go": 26},
                                                                  "krea2": {"classe": "krea2", "memoire_go": 29},
                                                                  "flux": {"classe": "flux3klein"}}}))
    voix = tmp_path / "voix.json"
    voix.write_text(json.dumps({"voix": {"frederique": {"modele": "drama3", "memoire_go": 24, "alias": ["drama-d2"]},
                                         "drama-d1": {"modele": "drama3"},
                                         "kokoro-siwis": {"modele": "kokoro", "memoire_go": 3}},
                                "defaut": {"*": "kokoro-siwis"}}))
    return image, voix


def famille(nom, *options, **kw) -> Famille:
    port = port_libre()
    argv = [sys.executable, "-m", "pli.gen.fils_simule", "--port", "{port}", *map(str, options)]
    return Famille(nom=nom, argv=argv, port=port, sante_delai_s=20, **kw)


@pytest.fixture
def fabrique(tmp_path, monkeypatch):
    monkeypatch.setenv("PLI_GEN_PIDS", str(tmp_path / "pids"))
    faits: list[Superviseur] = []

    def faire(familles, **kw):
        kw.setdefault("echange_octets", lambda: 0)
        kw.setdefault("reserve_octets", 0)
        kw.setdefault("delai_kill_s", 3.0)
        kw.setdefault("env", {"PYTHONPATH": SRC})
        kw.setdefault("drapeau", str(tmp_path / "general.drapeau"))
        sup = Superviseur({f.nom: f for f in familles}, **kw)
        faits.append(sup)
        return sup

    yield faire
    for sup in faits:
        sup.arreter()


def corps(**d) -> bytes:
    return json.dumps(d).encode()


def test_besoin_du_modele_demande_lu_dans_le_catalogue(fabrique, catalogues):
    image, voix = catalogues
    img = famille("image", "--catalogue", image, memoire_octets=26 * GO)
    aud = famille("audio", "--voix", voix, memoire_octets=16 * GO)
    sup = fabrique([img, aud], memoire_disponible=lambda: 64 * GO)
    b = lambda fam, chemin, c: sup.besoin_demande(fam, "POST", chemin, c)      # noqa: E731
    assert b(img, "/v1/images/generations", corps(prompt="x", model="krea2")) == 29 * GO
    assert b(img, "/v1/images/generations", corps(prompt="x")) == 26 * GO               # catalog default
    assert b(img, "/v1/images/generations", corps(prompt="x", model="inconnu")) == 0
    assert b(img, "/v1/images/generations", corps(prompt="x", model="flux")) == 0       # without memory_go: the family
    assert b(aud, "/v1/audio/speech", corps(input="x", voice="frederique")) == 24 * GO
    assert b(aud, "/v1/audio/speech", corps(input="x", voice="drama-d2")) == 24 * GO    # alias
    assert b(aud, "/v1/audio/speech", corps(input="x", voice="drama-d1")) == 24 * GO    # drama3 model default
    assert b(aud, "/v1/audio/speech", corps(input="x")) == 3 * GO                       # "*" default: Kokoro
    assert b(aud, "/v1/audio/transcriptions", b"--multipart--") == 0
    assert sup.besoin_demande(img, "GET", "/v1/images/generations", b"") == 0


def _dispo_selon(sup, nom, avec, sans):
    def lire():
        etat = sup._etats[nom]
        vivant = etat.fils is not None and etat.fils.proc.poll() is None
        return avec if vivant else sans
    return lire


def test_le_modele_demande_fait_arreter_l_audio_inactif_avant_le_lancement(fabrique, catalogues):
    image, voix = catalogues
    img = famille("image", "--catalogue", image, memoire_octets=26 * GO, reserve_octets=4 * GO)
    aud = famille("audio", "--voix", voix, memoire_octets=16 * GO)
    sup = fabrique([img, aud])
    sup.memoire_disponible = _dispo_selon(sup, "audio", 30 * GO, 50 * GO)
    assert sup.relayer("audio", "POST", "/v1/audio/speech", JSON, corps(input="x"))[0] == 200
    # qwen21: 26 + 4 = 30 GB, enough next to the voice: audio remains (as before)
    assert sup.relayer("image", "POST", "/v1/images/generations", JSON, corps(prompt="x", model="qwen21"))[0] == 200
    assert sup.etat_fils("audio")["vivant"] is True
    sup._arreter_fils(sup._etats["image"])
    # Krea 2: 29 + 4 = 33 GB: the inactive voice is stopped BEFORE launch (before: 26 + 4 counted, launched without stop)
    assert sup.relayer("image", "POST", "/v1/images/generations", JSON, corps(prompt="x", model="krea2"))[0] == 200
    assert sup.etat_fils("audio")["vivant"] is False and sup.etat_fils("image")["vivant"] is True


def test_refus_net_si_le_modele_demande_ne_tient_pas(fabrique, catalogues):
    image, _ = catalogues
    img = famille("image", "--catalogue", image, memoire_octets=26 * GO, reserve_octets=4 * GO)
    sup = fabrique([img], memoire_disponible=lambda: 31 * GO)
    code, _, brut = sup.relayer("image", "POST", "/v1/images/generations", JSON, corps(prompt="x", model="krea2"))
    erreur = json.loads(brut)["error"]
    assert code == 503 and erreur["type"] == "memoire" and erreur["requis_octets"] == 33 * GO
    assert sup.etat_fils("image")["vivant"] is False


def test_un_503_memoire_du_fils_arrete_les_inactifs_puis_reessaie(fabrique, catalogues):
    image, voix = catalogues
    img = famille("image", "--catalogue", image, memoire_octets=26 * GO, reserve_octets=4 * GO)
    aud = famille("audio", "--voix", voix, memoire_octets=16 * GO)
    sup = fabrique([img, aud], memoire_disponible=lambda: 64 * GO)
    assert sup.relayer("image", "POST", "/v1/images/generations", JSON, corps(prompt="x"))[0] == 200
    assert sup.relayer("audio", "POST", "/v1/audio/speech", JSON, corps(input="x"))[0] == 200
    sup.memoire_disponible = _dispo_selon(sup, "audio", 20 * GO, 50 * GO)
    vrai, appels = sup._http, []

    def http(fam, methode, chemin, en_tetes, c):
        appels.append(fam.nom)
        if fam.nom == "image" and appels.count("image") == 1:
            return 503, JSON, json.dumps({"error": {"type": "memoire", "message": "29 Go requis"}}).encode()
        return vrai(fam, methode, chemin, en_tetes, c)

    sup._http = http
    assert sup.relayer("image", "POST", "/v1/images/generations", JSON, corps(prompt="x", model="krea2"))[0] == 200
    assert appels == ["image", "image"] and sup.etat_fils("audio")["vivant"] is False


def test_un_503_sans_rien_a_arreter_est_rendu_tel_quel(fabrique, catalogues):
    image, _ = catalogues
    img = famille("image", "--catalogue", image, memoire_octets=26 * GO)
    sup = fabrique([img], memoire_disponible=lambda: 64 * GO)
    assert sup.relayer("image", "POST", "/v1/images/generations", JSON, corps(prompt="x"))[0] == 200
    sup.memoire_disponible = lambda: 20 * GO
    appels = []
    sup._http = lambda *a: appels.append(1) or (503, JSON, b'{"error": {"type": "memoire", "message": "m"}}')
    assert sup.relayer("image", "POST", "/v1/images/generations", JSON, corps(prompt="x", model="krea2"))[0] == 503
    assert appels == [1]


def test_reserve_video_voix_active_l_audio_n_est_pas_arrete(fabrique, catalogues, tmp_path):
    image, voix = catalogues
    img = famille("image", "--catalogue", image, memoire_octets=26 * GO, reserve_octets=4 * GO)
    aud = famille("audio", "--voix", voix, memoire_octets=16 * GO)
    vid = Famille(nom="video", port=port_libre(), sante_delai_s=20, delai_decharge_s=300, argv=[
        sys.executable, "-m", "pli.gen.fils_video", "--port", "{port}", "--simuler", "--sorties", str(tmp_path / "v")])
    sup = fabrique([img, aud, vid], memoire_disponible=lambda: 64 * GO, reserve_voix=True)
    assert sup.relayer("video", "GET", "/v1/videos/video_0000000000000000", {}, b"")[0] == 404     # fils vidéo vivant
    assert sup.relayer("audio", "POST", "/v1/audio/speech", JSON, corps(input="x"))[0] == 200
    sup.memoire_disponible = _dispo_selon(sup, "audio", 30 * GO, 50 * GO)
    open(sup.drapeau_video, "w").close()                       # a video is running
    code, _, brut = sup.relayer("image", "POST", "/v1/images/generations", JSON, corps(prompt="x", model="krea2"))
    assert code == 503 and json.loads(brut)["error"]["type"] == "memoire"
    assert sup.etat_fils("audio")["vivant"] is True
    os.unlink(sup.drapeau_video)                               # video finished: audio can leave
    assert sup.relayer("image", "POST", "/v1/images/generations", JSON, corps(prompt="x", model="krea2"))[0] == 200
    assert sup.etat_fils("audio")["vivant"] is False


# bounded wait for the 27B at startup (app.py)

def test_la_fenetre_qui_ne_tient_pas_attend_puis_reessaie(monkeypatch):
    from pli.server import prompt_memory
    from tests.test_memory_window import app

    vrai, essais = prompt_memory.PromptMemory.fit_window, []

    def fit(self, window, fit_):
        essais.append(window)
        if len(essais) == 1:
            raise ValueError("a 8,192-token context window does not fit this server's memory budget")
        return vrai(self, window, fit_)

    monkeypatch.setattr(prompt_memory.PromptMemory, "fit_window", fit)
    sizes = []
    vrai_sized = prompt_memory.PromptMemory.sized
    monkeypatch.setattr(prompt_memory.PromptMemory, "sized", lambda self, *a, **k: sizes.append(1) or vrai_sized(self, *a, **k))
    debut = time.monotonic()
    served = app(int(1.5 * 2**30), context_window=8192, fit_context=True, fit_wait_s=5, fit_step_s=0.2)
    try:
        assert served.context_window == 8192 and len(essais) == 2 and len(sizes) == 2
        assert time.monotonic() - debut >= 0.2
    finally:
        served.close()


def test_le_refus_vient_apres_l_attente_bornee():
    from tests.test_memory_window import app

    debut = time.monotonic()
    with pytest.raises(ValueError, match="262,144-token context window"):
        app(int(1.5 * 2**30), context_window=262144, fit_wait_s=0.9, fit_step_s=0.3)
    assert 0.6 <= time.monotonic() - debut < 10


def test_sans_attente_le_refus_est_immediat_comme_avant():
    from tests.test_memory_window import app

    debut = time.monotonic()
    with pytest.raises(ValueError):
        app(int(1.5 * 2**30), context_window=262144)
    assert time.monotonic() - debut < 5
