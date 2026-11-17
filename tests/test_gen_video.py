"""Génération vidéo (PLI 5) : fils vidéo simulé, général chargé (streaming ou refus), drapeau, arrêt, orphelins, /health."""

from __future__ import annotations

import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import pli
from pli.gen import fils_video
from pli.gen.superviseur import Famille, Superviseur
from pli.server.http import Server, make_handler

SRC = str(Path(pli.__file__).resolve().parent.parent)


def port_libre() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def appel(port, methode, chemin, corps=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    try:
        c.request(methode, chemin, body=json.dumps(corps).encode() if corps is not None else None,
                  headers={"Content-Type": "application/json"})
        r = c.getresponse()
        return r.status, r.getheader("Content-Type"), r.read()
    finally:
        c.close()


def attendre(condition, delai=20.0):
    fin = time.monotonic() + delai
    while time.monotonic() < fin:
        valeur = condition()
        if valeur:
            return valeur
        time.sleep(0.02)
    raise AssertionError("condition not met in time")


def famille_video(tmp_path, **plus):
    return Famille(
        nom="video", port=port_libre(), memoire_octets=0, sante_delai_s=20, delai_decharge_s=plus.pop("delai", 300),
        argv=[sys.executable, "-m", "pli.gen.fils_video", "--port", "{port}", "--simuler",
              "--sorties", str(tmp_path / "sorties")], **plus)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", SRC)
    monkeypatch.setenv("PLI_GEN_PIDS", str(tmp_path / "pids"))
    monkeypatch.delenv("PLI_GEN_VIDEO_AVEC_GENERAL", raising=False)
    monkeypatch.delenv("PLI_GEN_DRAPEAU", raising=False)
    return tmp_path


@pytest.fixture
def superviseur(env):
    sups = []

    def faire(**plus):
        s = Superviseur({"video": famille_video(env, **plus.pop("famille", {}))}, reserve_octets=0,
                        memoire_disponible=lambda: 1 << 40, echange_octets=lambda: 0, **plus)
        sups.append(s)
        return s
    yield faire
    for s in sups:
        s.arreter()


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
        if gen is not None:
            gen.arreter()


@pytest.fixture
def config(env, monkeypatch):
    chemin = env / "gen.json"
    chemin.write_text(json.dumps({"reserve_go": 0, "familles": {"video": {
        "argv": ["{python}", "-m", "pli.gen.fils_video", "--port", "{port}", "--simuler",
                 "--sorties", str(env / "sorties")],
        "port": port_libre(), "memoire_go": 0, "sante_delai_s": 20, "delai_decharge_s": 300}}}))
    monkeypatch.setenv("PLI_GEN_CONFIG", str(chemin))
    monkeypatch.setenv("PLI_GEN_RESERVE_GO", "0")
    monkeypatch.setenv("PLI_GEN", "1")


def appli(**plus):
    return SimpleNamespace(served="f", served_name="f", model_ids=["f"], max_batch_size=1, auth=None, **plus)


def attendre_statut(port, id_, voulu, delai=20.0):
    def lire():
        etat = json.loads(appel(port, "GET", f"/v1/videos/{id_}")[2])
        return etat if etat["status"] in voulu else None
    return attendre(lire, delai)


# end-to-end by the HTTP server

def test_travail_simule_de_bout_en_bout(config):
    with serveur(appli()) as port:
        code, _, corps = appel(port, "POST", "/v1/videos", {"prompt": "a hummingbird", "seconds": 2, "size": "512x512",
                                                              "steps": 5, "seed": 7, "reuse": 2, "layers": 45})
        cree = json.loads(corps)
        assert code == 202 and cree["status"] == "queued" and cree["id"].startswith("video_")
        fini = attendre_statut(port, cree["id"], {"completed", "failed"})
        assert fini["status"] == "completed" and fini["progress"] == 1.0 and fini["error"] is None
        code, type_, contenu = appel(port, "GET", f"/v1/videos/{cree['id']}/content")
        assert code == 200 and type_ == "video/mp4"
        ligne = contenu.decode()
        assert ligne.startswith("MP4-SIMULE") and "--seconds 2" in ligne and "--width 512 --height 512" in ligne
        assert "--steps 5" in ligne and "--seed 7" in ligne and "--reuse 2" in ligne and "--layers 45" in ligne
        assert appel(port, "GET", "/v1/videos/video_0000000000000000")[0] == 404
        assert appel(port, "GET", "/v1/videos/inconnu")[0] == 404


def test_demandes_mal_formees_refusees_en_400(config):
    with serveur(appli()) as port:
        for corps in ({}, {"prompt": "x", "seconds": 1, "frames": 5}, {"prompt": "x", "size": "grand"},
                      {"prompt": "x", "steps": "a lot"}, {"prompt": "x", "first_frame": "!!not-base64"}):
            assert appel(port, "POST", "/v1/videos", corps)[0] == 400, corps


def test_images_de_depart_et_de_fin_transmises(config, env):
    import base64
    img = base64.b64encode(b"png").decode()
    with serveur(appli()) as port:
        cree = json.loads(appel(port, "POST", "/v1/videos", {"prompt": "x", "frames": 22, "first_frame": img,
                                                             "last_frame": "data:image/png;base64," + img})[2])
        attendre_statut(port, cree["id"], {"completed"})
        ligne = appel(port, "GET", f"/v1/videos/{cree['id']}/content")[2].decode()
        assert "--frames 22" in ligne and ".first_frame" in ligne and ".last_frame" in ligne


def test_acceptee_a_cote_du_general_en_streaming(config):
    with serveur(appli(scheduler=object())) as port:
        code, _, corps = appel(port, "POST", "/v1/videos", {"prompt": "x"})
        assert code == 202, corps
        id_ = json.loads(corps)["id"]
        attendre_statut(port, id_, {"completed"})
        assert "--ssd-streaming" in appel(port, "GET", f"/v1/videos/{id_}/content")[2].decode()


def test_refus_503_quand_le_general_est_charge_sans_streaming(config, env, monkeypatch):
    chemin = env / "gen.json"
    donnees = json.loads(chemin.read_text())
    donnees["familles"]["video"]["argv"] += ["--ssd-streaming", "non"]
    chemin.write_text(json.dumps(donnees))
    with serveur(appli(scheduler=object())) as port:
        code, _, corps = appel(port, "POST", "/v1/videos", {"prompt": "x"})
        assert code == 503 and json.loads(corps)["error"]["type"] == "general_charge"
        monkeypatch.setenv("PLI_GEN_VIDEO_AVEC_GENERAL", "1")
        code, _, corps = appel(port, "POST", "/v1/videos", {"prompt": "x"})
        assert code == 202
        id_ = json.loads(corps)["id"]
        attendre_statut(port, id_, {"completed"})
        assert "--ssd-streaming" not in appel(port, "GET", f"/v1/videos/{id_}/content")[2].decode()
    with serveur(appli()) as port:                       # general model unloaded: accepted without streaming
        assert appel(port, "POST", "/v1/videos", {"prompt": "x"})[0] == 202


def test_delete_par_le_serveur_annule_le_travail(config, monkeypatch):
    monkeypatch.setenv("FAUX_H3_PAS_S", "0.5")
    with serveur(appli()) as port:
        id_ = json.loads(appel(port, "POST", "/v1/videos", {"prompt": "x"})[2])["id"]
        attendre_statut(port, id_, {"in_progress"})
        code, _, corps = appel(port, "DELETE", f"/v1/videos/{id_}")
        assert code == 200 and json.loads(corps)["status"] == "cancelled"
        assert appel(port, "GET", f"/v1/videos/{id_}")[0] == 200


def test_sans_pli_gen_les_routes_video_sont_inconnues(config, monkeypatch):
    monkeypatch.delenv("PLI_GEN")
    with serveur(appli()) as port:
        assert appel(port, "GET", "/v1/videos/video_0123456789abcdef")[0] == 404
        assert appel(port, "POST", "/v1/videos", {"prompt": "x"})[0] == 404


def test_health_avec_et_sans_pli_gen(config, monkeypatch):
    with serveur(appli()) as port:
        avant = json.loads(appel(port, "GET", "/health")[2])["gen"]["video"]
        assert avant["etat"] == "arrete" and avant["pid"] is None and avant["memoire_octets"] == 0
        cree = json.loads(appel(port, "POST", "/v1/videos", {"prompt": "x"})[2])
        attendre_statut(port, cree["id"], {"completed"})
        apres = json.loads(appel(port, "GET", "/health")[2])["gen"]["video"]
        assert apres["etat"] == "pret" and isinstance(apres["pid"], int) and apres["dernier_usage"] > 0
    monkeypatch.delenv("PLI_GEN")
    with serveur(appli()) as port:
        assert "gen" not in json.loads(appel(port, "GET", "/health")[2])


# video child process only

def moteur(env, **plus):
    return fils_video.Moteur("", "", str(env / "s"), simuler=True, **plus)


def test_le_drapeau_retarde_le_demarrage_du_travail_suivant(env, monkeypatch):
    drapeau = env / "general.drapeau"
    drapeau.write_text("")
    monkeypatch.setenv("PLI_GEN_DRAPEAU", str(drapeau))
    m = moteur(env)
    try:
        t = m.creer({"prompt": "x"})
        time.sleep(0.6)
        assert t.status == "queued"
        drapeau.unlink()
        attendre(lambda: t.status == "completed")
    finally:
        m.arreter()


def test_drapeau_apparu_pendant_un_travail_le_met_en_pause_et_retient_le_suivant(env, monkeypatch):
    drapeau = env / "general.drapeau"
    monkeypatch.setenv("PLI_GEN_DRAPEAU", str(drapeau))
    monkeypatch.setenv("FAUX_H3_PAS_S", "0.1")
    m = moteur(env)
    try:
        a = m.creer({"prompt": "a"})
        attendre(lambda: a.status == "in_progress")
        drapeau.write_text("")
        b = m.creer({"prompt": "b"})
        attendre(lambda: a.en_pause)
        time.sleep(0.4)
        assert a.status == "in_progress" and b.status == "queued"
        drapeau.unlink()
        attendre(lambda: b.status == "completed")
        assert a.status == "completed" and a.pauses == 1
    finally:
        m.arreter()


def test_progression_lue_dans_la_sortie_de_h3_et_echec_rapporte(env, monkeypatch):
    monkeypatch.setenv("FAUX_H3_PAS_S", "0.15")
    m = moteur(env)
    try:
        t = m.creer({"prompt": "x"})
        attendre(lambda: t.phase and t.phase["nom"] == "denoise" and 0 < t.progress < 1)
        attendre(lambda: t.status == "completed")
        assert t.progress == 1.0
        monkeypatch.setenv("FAUX_H3_ECHEC", "1")
        monkeypatch.setenv("FAUX_H3_PAS_S", "0")
        u = m.creer({"prompt": "y"})
        attendre(lambda: u.status == "failed")
        assert "code 2" in u.error and "echec simule" in u.error
    finally:
        m.arreter()


def test_un_travail_interrompu_est_relu_comme_echec(env):
    m = moteur(env)
    t = m.creer({"prompt": "x"})
    attendre(lambda: t.status == "completed")
    m.arreter()
    ancien = json.loads((env / "s" / f"{t.id}.json").read_text())
    ancien["status"] = "in_progress"
    (env / "s" / f"{t.id}.json").write_text(json.dumps(ancien))
    m2 = moteur(env)
    try:
        assert m2.travaux[t.id].status == "failed" and "interrupted" in m2.travaux[t.id].error
    finally:
        m2.arreter()


# superviseur

def h3_vivants(env):
    return subprocess.run(["pgrep", "-f", str(env / "sorties")], capture_output=True, text=True).stdout.split()


def test_arret_propre_tue_le_fils_et_son_h3(superviseur, env, monkeypatch):
    monkeypatch.setenv("FAUX_H3_PAS_S", "2")
    sup = superviseur()
    code, _, corps = sup.relayer("video", "POST", "/v1/videos", {"content-type": "application/json"}, b'{"prompt":"x"}')
    assert code == 202
    attendre(lambda: len(h3_vivants(env)) >= 2)            # the child and its fake h3
    sup.arreter()
    attendre(lambda: not h3_vivants(env), 15)
    assert sup.etat_fils("video")["vivant"] is False


def test_delai_zero_arrete_le_fils_apres_le_travail_pas_pendant(superviseur, env, monkeypatch):
    monkeypatch.setenv("FAUX_H3_PAS_S", "0.25")
    sup = superviseur(famille={"delai": 0}, intervalle_s=0.1)
    code, _, corps = sup.relayer("video", "POST", "/v1/videos", {"content-type": "application/json"}, b'{"prompt":"x"}')
    id_ = json.loads(corps)["id"]
    time.sleep(1.0)                                     # sweeps pass during the job
    assert sup.etat_fils("video")["vivant"] is True
    attendre(lambda: sup.etat_fils("video")["vivant"] is False)
    assert (env / "sorties" / f"{id_}.mp4").exists()


def test_nettoyage_des_orphelins_d_un_lancement_precedent(env):
    fam = famille_video(env)
    argv = [sys.executable, "-c", "import time; time.sleep(60)", str(env), "{port}"]
    fam.argv = argv
    orphelin = subprocess.Popen([a.replace("{port}", str(fam.port)) for a in argv], start_new_session=True)
    # Just after Popen, ps may still show the parent command (before exec): wait for the real command.
    attendre(lambda: "time.sleep(60)" in subprocess.run(["ps", "-ww", "-o", "command=", "-p", str(orphelin.pid)],
                                                         capture_output=True, text=True).stdout, 5)
    os.makedirs(env / "pids")
    (env / "pids" / f"video-{fam.port}.pid").write_text(
        json.dumps({"pid": orphelin.pid, "argv": [a.replace("{port}", str(fam.port)) for a in argv]}))
    sup = Superviseur({"video": fam}, reserve_octets=0)
    try:
        assert sup.orphelins_tues == ["video"]
        assert orphelin.wait(timeout=10) is not None
        assert not (env / "pids" / f"video-{fam.port}.pid").exists()
    finally:
        sup.arreter()
        if orphelin.poll() is None:
            orphelin.kill()


def test_un_pid_reutilise_par_un_autre_programme_est_laisse(env):
    fam = famille_video(env)
    autre = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    os.makedirs(env / "pids")
    (env / "pids" / f"video-{fam.port}.pid").write_text(json.dumps({"pid": autre.pid, "argv": fam.argv}))
    sup = Superviseur({"video": fam}, reserve_octets=0)
    try:
        assert sup.orphelins_tues == []
        assert autre.poll() is None
    finally:
        sup.arreter()
        autre.kill()
        autre.wait()


def test_le_general_charge_se_decide_sur_l_appli():
    from pli.gen import routes
    assert routes.general_charge(SimpleNamespace(scheduler=object())) is True
    assert routes.general_charge(SimpleNamespace()) is False
    assert routes.general_charge(SimpleNamespace(scheduler=object(), general_charge=lambda: False)) is False
