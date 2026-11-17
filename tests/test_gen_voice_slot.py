"""Room for the voice assistant during a video (14/11/2026): general model's parallelism reduced, voice memory reserved, voice never blocked.

Simulated child processes, no models: (a) video_parallele_general key, (b) video_reserve_voix, (c) video_pause_voix.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import pli
from pli.gen import config as gen_config
from pli.gen.fils_video import GO, Moteur
from pli.gen.memoire_general import MemoireGeneral
from pli.gen.superviseur import Famille, Superviseur

SRC = str(Path(pli.__file__).resolve().parent.parent)
JSON = {"content-type": "application/json"}
VARIABLES = ("PLI_GEN_DRAPEAU", "PLI_GEN_DRAPEAU_AUTRES", "PLI_GEN_DRAPEAU_VOIX", "PLI_GEN_DRAPEAU_VIDEO",
             "PLI_GEN_DRAPEAU_AUDIO", "PLI_GEN_BESOIN", "PLI_GEN_VOIX_OCTETS", "PLI_GEN_VOIX_RESERVE_OCTETS",
             "PLI_GEN_ATTENTE_MEMOIRE_S", "FAUX_H3_PAS_S", "FAUX_H3_ECHEC")


def attendre(condition, delai=20.0):
    fin = time.monotonic() + delai
    while time.monotonic() < fin:
        valeur = condition()
        if valeur:
            return valeur
        time.sleep(0.02)
    raise AssertionError("condition not met in time")


def etat_processus(pid: int) -> str:
    return subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()


def port_libre() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def env(tmp_path, monkeypatch):
    for nom in VARIABLES:
        monkeypatch.delenv(nom, raising=False)
    return tmp_path


@pytest.fixture
def moteurs():
    faits: list[Moteur] = []

    def faire(env, **kw):
        m = Moteur("", "", str(env / "s"), simuler=True, **kw)
        faits.append(m)
        return m
    yield faire
    for m in faits:
        m.arreter()


@pytest.fixture
def sup_gen(env, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", SRC)
    monkeypatch.setenv("PLI_GEN_PIDS", str(env / "pids"))
    faits: list[Superviseur] = []

    def faire(audio=None, memoire=lambda: 1 << 40, decharge=0.0, **kw):
        familles = {"video": Famille(nom="video", port=port_libre(), sante_delai_s=20, delai_decharge_s=decharge, argv=[
            sys.executable, "-m", "pli.gen.fils_video", "--port", "{port}", "--simuler",
            "--sorties", str(env / "sorties")], memoire_octets=12 * GO)}
        if audio is not None:
            familles["audio"] = Famille(nom="audio", port=port_libre(), sante_delai_s=20, argv=[
                sys.executable, "-m", "pli.gen.fils_simule", "--port", "{port}", *audio],
                memoire_octets=16 * GO)
        kw.setdefault("reserve_octets", 6 * GO)
        s = Superviseur(familles, memoire_disponible=memoire, echange_octets=lambda: 0,
                        drapeau=str(env / "general.drapeau"), intervalle_s=0.05, **kw)
        faits.append(s)
        return s
    yield faire
    for s in faits:
        s.arreter()


def video(sup, corps=b'{"prompt":"x"}'):
    code, _, brut = sup.relayer("video", "POST", "/v1/videos", JSON, corps)
    assert code == 202, brut
    return json.loads(brut)["id"]


def etat_video(sup, id_):
    return json.loads(sup.relayer("video", "GET", f"/v1/videos/{id_}", {}, b"")[2])


# (a) general model parallelism during a video

def test_limiter_parallele_sur_le_scheduler_puis_retour():
    sched = SimpleNamespace(lanes=4)
    m = MemoireGeneral(store=None, runtime=None, on_engine=None, scheduler=sched)
    m.limiter_parallele(1)
    assert sched.lanes == 1
    m.limiter_parallele(8)                     # never more than the original count
    assert sched.lanes == 4
    m.limiter_parallele(None)
    assert sched.lanes == 4
    MemoireGeneral(store=None, runtime=None, on_engine=None).limiter_parallele(1)     # without scheduler: nothing


def test_parallele_reduit_a_chaud_les_demandes_attendent_sans_refus():
    from tests.test_lane_server import SlowFakeEngine, make_app

    app = make_app(lanes=2, engine_factory=SlowFakeEngine)
    m = MemoireGeneral(store=None, runtime=None, on_engine=None, scheduler=app.scheduler)
    pic = [0]
    fin = threading.Event()

    def mesurer():
        while not fin.is_set():
            pic[0] = max(pic[0], app.scheduler.engine.active_count + len(app.scheduler._fills))
            time.sleep(0.0005)

    def lot(n, resultats):
        fils = [threading.Thread(target=lambda i=i: resultats.append(
            app.chat([{"role": "user", "content": f"question {i}"}], max_tokens=40))) for i in range(n)]
        for f in fils:
            f.start()
        for f in fils:
            f.join(60)

    try:
        veille = threading.Thread(target=mesurer, daemon=True)
        veille.start()
        m.limiter_parallele(1)
        reduits: list = []
        lot(3, reduits)
        pic_reduit = pic[0]
        m.limiter_parallele(None)
        pic[0] = 0
        normaux: list = []
        lot(3, normaux)
        fin.set()
        veille.join(5)
        assert len(reduits) == 3 and all(r.get("content") for r in reduits)      # none refused
        assert pic_reduit == 1                                                   # only one at a time
        assert len(normaux) == 3 and app.scheduler.lanes == 2 and pic[0] == 2
    finally:
        fin.set()
        app.close()


def test_le_superviseur_reduit_le_general_tant_que_la_video_tourne(sup_gen, monkeypatch):
    monkeypatch.setenv("FAUX_H3_PAS_S", "0.15")
    appels: list = []
    sup = sup_gen(parallele=appels.append, parallele_video=1)
    id_ = video(sup)
    attendre(lambda: appels == [1])
    assert etat_video(sup, id_)["status"] == "in_progress"
    attendre(lambda: etat_video(sup, id_)["status"] == "completed")
    attendre(lambda: appels == [1, None])


def test_sans_cle_rien_ne_change(sup_gen, monkeypatch):
    monkeypatch.setenv("FAUX_H3_PAS_S", "0.05")
    appels: list = []
    sup = sup_gen(audio=[], parallele=appels.append)
    assert sup._env_video() == {}
    id_ = video(sup)
    attendre(lambda: etat_video(sup, id_)["status"] == "completed")
    time.sleep(0.3)
    assert appels == [] and not os.path.exists(sup.drapeau_voix) and not os.path.exists(sup.drapeau_audio)


def test_configuration_des_cles_video(tmp_path, monkeypatch):
    familles = {"audio": {"argv": ["x"], "port": 1, "memoire_go": 16, "reserve_go": 6},
                "video": {"argv": ["x"], "port": 2, "memoire_go": 12}}
    chemin = tmp_path / "gen.json"
    monkeypatch.setenv("PLI_GEN_PIDS", str(tmp_path / "pids"))
    monkeypatch.delenv("PLI_GEN_RESERVE_GO", raising=False)
    chemin.write_text(json.dumps({"familles": familles}))
    monkeypatch.setenv("PLI_GEN_CONFIG", str(chemin))
    sup = gen_config.superviseur_depuis_env()
    try:
        assert (sup.parallele_video, sup.reserve_voix, sup.pause_voix) == (None, False, False)
    finally:
        sup.arreter()
    chemin.write_text(json.dumps({"familles": familles, "video_parallele_general": 1, "video_reserve_voix": True,
                                  "video_pause_voix": True, "video_attente_memoire_s": 900}))
    sup = gen_config.superviseur_depuis_env()
    try:
        assert (sup.parallele_video, sup.reserve_voix, sup.pause_voix, sup.attente_memoire_s) == (1, True, True, 900)
        plus = sup._env_video()
        assert plus["PLI_GEN_VOIX_OCTETS"] == str(22 * GO) and plus["PLI_GEN_VOIX_RESERVE_OCTETS"] == str(6 * GO)
        assert plus["PLI_GEN_DRAPEAU_VIDEO"] == sup.drapeau_video and plus["PLI_GEN_DRAPEAU_VOIX"] == sup.drapeau_voix
    finally:
        sup.arreter()
    chemin.write_text(json.dumps({"familles": familles, "video_parallele_general": 0}))
    with pytest.raises(ValueError):
        gen_config.superviseur_depuis_env()
    chemin.write_text(json.dumps({"familles": familles, "video_reserve_voix": "oui"}))
    with pytest.raises(ValueError):
        gen_config.superviseur_depuis_env()


# (b) voice reserve

def reserve(env, monkeypatch, attente_s=None):
    audio = env / "general.drapeau.audio"
    besoin = env / "general.drapeau.besoin"
    monkeypatch.setenv("PLI_GEN_VOIX_OCTETS", str(22 * GO))
    monkeypatch.setenv("PLI_GEN_VOIX_RESERVE_OCTETS", str(6 * GO))
    monkeypatch.setenv("PLI_GEN_DRAPEAU_AUDIO", str(audio))
    monkeypatch.setenv("PLI_GEN_BESOIN", str(besoin))
    if attente_s is not None:
        monkeypatch.setenv("PLI_GEN_ATTENTE_MEMOIRE_S", str(attente_s))
    return audio, besoin


def test_la_video_attend_la_memoire_de_la_voix_puis_demarre(env, moteurs, monkeypatch):
    _, besoin = reserve(env, monkeypatch)
    libre = [30 * GO]
    m = moteurs(env, memoire_disponible=lambda: libre[0])
    t = m.creer({"prompt": "x"})                        # 30 GB cover the video alone: accepted
    attendre(lambda: t.attente == "memoire_voix")
    attendre(besoin.exists)
    assert json.loads(besoin.read_text())["octets"] == 4 * GO     # 12 + 16 + 6 - 30
    time.sleep(0.6)
    assert t.status == "queued" and m.vue(t)["attente"] == "memoire_voix"
    libre[0] = 34 * GO
    attendre(lambda: t.status == "completed")
    assert not besoin.exists() and t.attente is None


def test_voix_deja_chargee_seule_la_reserve_compte(env, moteurs, monkeypatch):
    audio, besoin = reserve(env, monkeypatch)
    audio.write_text("")
    m = moteurs(env, memoire_disponible=lambda: 18 * GO)        # 12 + 6
    t = m.creer({"prompt": "x"})
    attendre(lambda: t.status == "completed")
    assert not besoin.exists()


def test_la_memoire_ne_vient_pas_le_travail_echoue_en_memoire(env, moteurs, monkeypatch):
    reserve(env, monkeypatch, attente_s=0.6)
    m = moteurs(env, memoire_disponible=lambda: 30 * GO)
    t = m.creer({"prompt": "x"})
    attendre(lambda: t.status == "failed", 10)
    assert t.error_type == "memoire" and "il manque 4.0 Go" in t.error


def test_sans_reserve_la_video_demarre_comme_avant(env, moteurs):
    m = moteurs(env, memoire_disponible=lambda: 13 * GO)
    t = m.creer({"prompt": "x"})
    attendre(lambda: t.status == "completed")


def test_le_superviseur_fait_rendre_le_general_pour_la_voix(sup_gen):
    rendus: list = []
    sup = sup_gen(audio=[], reserve_voix=True, liberer=lambda n: rendus.append(n) or n, rendre=lambda: rendus.append("rendu"),
                  delai_besoin_s=0.5, decharge=300)
    assert sup.relayer("video", "GET", "/v1/videos/video_0000000000000000", {}, b"")[0] == 404    # child process started
    with open(sup.fichier_besoin, "w") as f:
        json.dump({"octets": 4 * GO, "id": "x"}, f)
    attendre(lambda: rendus[:1] == [4 * GO])
    attendre(lambda: "video" in sup._liberees)
    os.unlink(sup.fichier_besoin)
    n = len(rendus)
    time.sleep(0.8)
    assert len(rendus) == n                             # no more need: no more release


def test_le_drapeau_de_la_voix_chargee_suit_le_fils_audio(sup_gen):
    sup = sup_gen(audio=[], reserve_voix=True, decharge=300)
    assert sup.relayer("video", "GET", "/v1/videos/video_0000000000000000", {}, b"")[0] == 404
    time.sleep(0.3)
    assert not os.path.exists(sup.drapeau_audio)
    assert sup.relayer("audio", "POST", "/v1/audio/speech", JSON, b"{}")[0] == 200
    attendre(lambda: os.path.exists(sup.drapeau_audio))


def test_reserve_bout_en_bout_le_fils_video_attend_et_le_general_rend(sup_gen, env, monkeypatch):
    monkeypatch.setenv("FAUX_H3_PAS_S", "0.05")
    rendus: list = []
    sup = sup_gen(audio=[], reserve_voix=True, liberer=lambda n: rendus.append(n) or 0, delai_besoin_s=0.3)
    # the video child process measures 30 GB (--memoire-disponible-go): 4 GB missing for video + voice + reserve
    sup.familles["video"].argv += ["--memoire-disponible-go", "30"]
    id_ = video(sup)
    attendre(lambda: etat_video(sup, id_)["attente"] == "memoire_voix")
    attendre(lambda: rendus and rendus[0] == 4 * GO)
    sup.relayer("video", "DELETE", f"/v1/videos/{id_}", {}, b"")
    attendre(lambda: etat_video(sup, id_)["status"] == "cancelled")


# (c) the voice passes during the video pause

def test_h3_en_pause_pendant_le_drapeau_de_la_voix(env, moteurs, monkeypatch):
    voix = env / "general.drapeau.voix"
    monkeypatch.setenv("PLI_GEN_DRAPEAU_VOIX", str(voix))
    monkeypatch.setenv("FAUX_H3_PAS_S", "0.1")
    m = moteurs(env)
    t = m.creer({"prompt": "x"})
    attendre(lambda: t.phase is not None)
    voix.write_text("")
    attendre(lambda: t.en_pause)
    pid = m.proc.pid
    attendre(lambda: etat_processus(pid).startswith("T"), 5)
    voix.unlink()
    attendre(lambda: t.status == "completed")
    assert t.pauses == 1 and not t.en_pause


def test_une_voix_pendant_une_video_passe_et_met_h3_en_pause(sup_gen, monkeypatch):
    monkeypatch.setenv("FAUX_H3_PAS_S", "0.2")
    sup = sup_gen(audio=["--etapes", "4", "--duree-etape", "0.25"], pause_voix=True)
    id_ = video(sup)
    attendre(lambda: etat_video(sup, id_)["status"] == "in_progress" and etat_video(sup, id_).get("phase"))
    reponse: list = []
    fil = threading.Thread(target=lambda: reponse.append(sup.relayer("audio", "POST", "/v1/audio/speech", JSON, b"{}")))
    fil.start()
    attendre(lambda: os.path.exists(sup.drapeau_voix))
    attendre(lambda: etat_video(sup, id_)["en_pause"], 5)
    fil.join(20)
    assert reponse and reponse[0][0] == 200 and not os.path.exists(sup.drapeau_voix)
    v = attendre(lambda: (lambda e: e if e["status"] == "completed" else None)(etat_video(sup, id_)))
    assert v["pauses"] >= 1


def test_sans_pause_voix_la_voix_tourne_a_cote_de_h3(sup_gen, monkeypatch):
    monkeypatch.setenv("FAUX_H3_PAS_S", "0.2")
    sup = sup_gen(audio=["--etapes", "2", "--duree-etape", "0.25"])
    id_ = video(sup)
    attendre(lambda: etat_video(sup, id_)["status"] == "in_progress")
    assert sup.relayer("audio", "POST", "/v1/audio/speech", JSON, b"{}")[0] == 200
    v = attendre(lambda: (lambda e: e if e["status"] == "completed" else None)(etat_video(sup, id_)))
    assert v["pauses"] == 0


def test_avec_reserve_de_voix_le_lancement_du_fils_video_ne_compte_pas_la_reserve(sup_gen):
    sans = sup_gen(audio=[], memoire=lambda: 13 * GO)
    code, _, corps = sans.relayer("video", "GET", "/v1/videos/video_0000000000000000", {}, b"")
    assert code == 503 and json.loads(corps)["error"]["type"] == "memoire"            # 12 + 6 > 13: as before
    avec = sup_gen(audio=[], memoire=lambda: 13 * GO, reserve_voix=True)
    assert avec.relayer("video", "GET", "/v1/videos/video_0000000000000000", {}, b"")[0] == 404   # started; the job will wait
