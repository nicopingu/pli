"""PLI Video 5 next to the 27B: SSD streaming, queue, SIGSTOP pause on the flag, cancellation, merged LoRAs (simulated)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from pli.gen import fils_video
from pli.gen.fils_video import GO, ErreurDemande, Moteur, valider_catalogue


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


@pytest.fixture
def env(tmp_path, monkeypatch):
    for nom in ("PLI_GEN_DRAPEAU", "PLI_GEN_DRAPEAU_AUTRES", "FAUX_H3_PAS_S", "FAUX_H3_ECHEC", "FAUSSE_FUSION_PAS_S",
                "FAUSSE_FUSION_OCTETS", "FAUSSE_FUSION_ECHEC"):
        monkeypatch.delenv(nom, raising=False)
    return tmp_path


@pytest.fixture
def moteurs():
    faits: list[Moteur] = []

    def faire(env, cat=None, **kw):
        m = Moteur("", "", str(env / "s"), simuler=True, catalogue=cat, **kw)
        faits.append(m)
        return m
    yield faire
    for m in faits:
        m.arreter()


def catalogue(env, **plus):
    base = env / "MiniMax-H3"
    base.mkdir(exist_ok=True)
    loras = {}
    for nom in ("a", "b", "c", "garde"):
        f = env / "lora" / f"{nom}.safetensors"
        f.parent.mkdir(exist_ok=True)
        f.write_bytes(b"lora")
        loras[nom] = {"chemin": str(f), "force": 1.0, "mode": "sauvegardee" if nom == "garde" else "a_la_demande",
                      "taille_go": 100_000 / GO}
    c = {"defaut": "h3", "modeles": {"h3": {"chemin": str(base)}}, "lora": loras,
         "dossier_fusions": str(env / "sauvegardes"), "cache_fusions": str(env / "cache"), "cache_go": 250_000 / GO,
         "fusion": {"marge_go": 0}}
    c.update(plus)
    return valider_catalogue(c)


def argv_h3(m: Moteur, t) -> str:
    return open(m._chemin(t.id, ".mp4"), encoding="utf-8").read()


# streaming and memory

def test_streaming_par_defaut_et_besoin_memoire_par_modele(env, moteurs):
    m = moteurs(env, memoire_disponible=lambda: 13 * GO)
    t = m.creer({"prompt": "x"}, general_charge=True)          # next to the general model: accepted in streaming
    attendre(lambda: t.status == "completed")
    assert "--ssd-streaming" in argv_h3(m, t) and t.params["ssd_streaming"] is True
    m2 = moteurs(env, memoire_disponible=lambda: 11 * GO)
    with pytest.raises(ErreurDemande) as e:
        m2.creer({"prompt": "x"})
    assert e.value.code == 503 and e.value.type == "memoire" and "12.0 GB required" in e.value.message


def test_sans_streaming_refus_si_general_charge_et_besoin_resident(env, moteurs):
    m = moteurs(env, ssd_streaming=False, memoire_disponible=lambda: 50 * GO)
    with pytest.raises(ErreurDemande) as e:
        m.creer({"prompt": "x"}, general_charge=True)
    assert e.value.code == 503 and e.value.type == "general_charge"
    t = m.creer({"prompt": "x"})
    attendre(lambda: t.status == "completed")
    assert "--ssd-streaming" not in argv_h3(m, t)
    m2 = moteurs(env, ssd_streaming=False, memoire_disponible=lambda: 30 * GO)
    with pytest.raises(ErreurDemande) as e:
        m2.creer({"prompt": "x"})
    assert e.value.type == "memoire" and "43.0 GB required" in e.value.message


# file

def test_file_de_deux_videos_une_a_la_fois(env, moteurs, monkeypatch):
    monkeypatch.setenv("FAUX_H3_PAS_S", "0.08")
    m = moteurs(env)
    a, b = m.creer({"prompt": "a"}), m.creer({"prompt": "b"})
    attendre(lambda: a.status == "in_progress")
    vue = m.vue(b)
    assert vue["status"] == "queued" and vue["position"] == 1 and vue["attente"] == "file"
    while a.status == "in_progress":
        assert b.status == "queued"
        time.sleep(0.02)
    attendre(lambda: b.status == "completed")
    assert a.status == "completed" and a.completed_at <= b.completed_at


def test_une_image_ou_une_voix_en_cours_retient_le_demarrage(env, moteurs, monkeypatch):
    autres = env / "general.drapeau.autres"
    autres.write_text("")
    monkeypatch.setenv("PLI_GEN_DRAPEAU_AUTRES", str(autres))
    m = moteurs(env)
    t = m.creer({"prompt": "x"})
    attendre(lambda: t.attente == "image_ou_voix")
    time.sleep(0.3)
    assert t.status == "queued"
    autres.unlink()
    attendre(lambda: t.status == "completed")


# preemption

def test_pause_sigstop_pendant_le_drapeau_puis_reprise(env, moteurs, monkeypatch):
    drapeau = env / "general.drapeau"
    monkeypatch.setenv("PLI_GEN_DRAPEAU", str(drapeau))
    monkeypatch.setenv("FAUX_H3_PAS_S", "0.1")
    m = moteurs(env)
    t = m.creer({"prompt": "x"})
    attendre(lambda: t.phase is not None)
    drapeau.write_text("")
    attendre(lambda: t.en_pause)
    pid = m.proc.pid
    attendre(lambda: etat_processus(pid).startswith("T"), 5)
    time.sleep(0.1)
    phase = dict(t.phase)
    time.sleep(0.5)
    assert t.phase == phase and t.status == "in_progress"      # nothing progresses during the pause
    drapeau.unlink()
    attendre(lambda: t.status == "completed")
    assert t.pauses == 1 and t.secondes_pause >= 0.4 and not t.en_pause
    assert m.vue(t)["pauses"] == 1


def test_annulation_pendant_une_pause(env, moteurs, monkeypatch):
    drapeau = env / "general.drapeau"
    monkeypatch.setenv("PLI_GEN_DRAPEAU", str(drapeau))
    monkeypatch.setenv("FAUX_H3_PAS_S", "0.2")
    m = moteurs(env)
    t = m.creer({"prompt": "x"})
    attendre(lambda: t.phase is not None)
    drapeau.write_text("")
    attendre(lambda: t.en_pause)
    proc = m.proc
    debut = time.monotonic()
    assert m.annuler(t.id) is True
    attendre(lambda: t.status == "cancelled", 5)
    assert proc.poll() is not None and proc.returncode == -15    # SIGTERM received: the process has been resumed
    assert time.monotonic() - debut < 5 and t.error_type == "annulee"
    file_suivante = m.creer({"prompt": "y"})                   # the queue continues (flag still there: waiting)
    time.sleep(0.2)
    assert file_suivante.status == "queued" and file_suivante.attente == "general"
    assert m.annuler(file_suivante.id) is True and file_suivante.status == "cancelled"


def test_arret_propre_pendant_une_pause(env, moteurs, monkeypatch):
    drapeau = env / "general.drapeau"
    monkeypatch.setenv("PLI_GEN_DRAPEAU", str(drapeau))
    monkeypatch.setenv("FAUX_H3_PAS_S", "0.2")
    m = moteurs(env)
    t = m.creer({"prompt": "x"})
    attendre(lambda: t.phase is not None)
    drapeau.write_text("")
    attendre(lambda: t.en_pause)
    proc = m.proc
    debut = time.monotonic()
    m.arreter()
    assert proc.poll() is not None and proc.returncode == -15 and time.monotonic() - debut < 5
    assert t.status == "failed" and "interrupted" in t.error


# LoRA and fusions

def test_fusion_a_la_demande_puis_reutilisation_du_cache(env, moteurs, monkeypatch):
    monkeypatch.setenv("FAUSSE_FUSION_PAS_S", "0.1")
    m = moteurs(env, catalogue(env))
    t = m.creer({"prompt": "x", "lora": [{"nom": "a"}]})
    attendre(lambda: t.etape == "fusion" and m.vue(t)["phase"]["nom"] == "fusion")
    attendre(lambda: t.status == "completed")
    assert t.fusion["source"] == "nouvelle" and t.fusion["mode"] == "a_la_demande"
    assert os.path.dirname(t.fusion["chemin"]) == str(env / "cache") and t.secondes_fusion >= 0.3
    assert f"-d {t.fusion['chemin']} " in argv_h3(m, t)
    u = m.creer({"prompt": "y", "lora": [{"nom": "a", "force": 1.0}]})
    attendre(lambda: u.status == "completed")
    assert u.fusion == {"chemin": t.fusion["chemin"], "source": "cache", "mode": "a_la_demande"}
    assert u.secondes_fusion < 0.3
    v = m.creer({"prompt": "z", "lora": [{"nom": "a", "force": 0.5}]})   # autre force : autre fusion
    attendre(lambda: v.status == "completed")
    assert v.fusion["source"] == "nouvelle" and v.fusion["chemin"] != t.fusion["chemin"]


def test_eviction_du_moins_recemment_utilise(env, moteurs, monkeypatch):
    monkeypatch.setenv("FAUSSE_FUSION_OCTETS", "100000")          # two fusions fit in 250,000 bytes
    m = moteurs(env, catalogue(env))

    def rendre(nom):
        t = m.creer({"prompt": nom, "lora": [{"nom": nom}]})
        attendre(lambda: t.status in fils_video.FINIS)
        assert t.status == "completed", t.error
        time.sleep(0.05)
        return t
    a, b = rendre("a"), rendre("b")
    assert rendre("a").fusion["source"] == "cache"               # a becomes the most recent
    c = rendre("c")
    assert os.path.isdir(a.fusion["chemin"]) and os.path.isdir(c.fusion["chemin"])
    assert not os.path.exists(b.fusion["chemin"])                 # b, the least recently used, is evicted


def test_fusion_sauvegardee_reutilisee_et_jamais_evincee(env, moteurs, monkeypatch):
    monkeypatch.setenv("FAUSSE_FUSION_OCTETS", "100000")
    cat = catalogue(env)
    existante = env / "modeles" / "MiniMax-H3-style"         # fusion done manually, declared in "fusions"
    existante.mkdir(parents=True)
    (existante / fils_video.META_FUSION).write_text(json.dumps(
        {"modele": cat["modeles"]["h3"]["chemin"], "loras": [{"fichier": cat["lora"]["b"]["chemin"], "force": 1.0}]}))
    cat["fusions"] = [str(env / "modeles")]
    m = moteurs(env, cat)
    t = m.creer({"prompt": "x", "lora": [{"nom": "b"}]})
    attendre(lambda: t.status == "completed")
    assert t.fusion["source"] == "sauvegardee" and t.fusion["chemin"] == str(existante)
    assert not os.path.exists(env / "cache")                       # aucune fusion faite
    g = m.creer({"prompt": "x", "lora": [{"nom": "garde"}]})       # save mode: out of cache, kept
    attendre(lambda: g.status == "completed")
    assert g.fusion["source"] == "nouvelle" and os.path.dirname(g.fusion["chemin"]) == str(env / "sauvegardes")
    for nom in ("a", "c", "b"):
        u = m.creer({"prompt": nom, "lora": [{"nom": nom}]})
        attendre(lambda: u.status == "completed")
    assert os.path.isdir(g.fusion["chemin"]) and os.path.isdir(existante)
    h = m.creer({"prompt": "x", "lora": [{"nom": "garde"}]})
    attendre(lambda: h.status == "completed")
    assert h.fusion == {"chemin": g.fusion["chemin"], "source": "sauvegardee", "mode": "sauvegardee"}


def test_place_disque_insuffisante(env, moteurs, monkeypatch):
    drapeau = env / "general.drapeau"
    monkeypatch.setenv("PLI_GEN_DRAPEAU", str(drapeau))
    place = {"libre": 50_000}
    m = moteurs(env, catalogue(env), place_libre=lambda d: place["libre"])
    with pytest.raises(ErreurDemande) as e:
        m.creer({"prompt": "x", "lora": [{"nom": "garde"}]})
    assert e.value.code == 507 and e.value.type == "disque" and "insufficient disk space" in e.value.message
    place["libre"] = 10**9
    drapeau.write_text("")                                         # the job waits in the queue
    t = m.creer({"prompt": "x", "lora": [{"nom": "a"}], "seed": 1})
    place["libre"] = 0                                             # space was missing between creation and fusion
    drapeau.unlink()
    attendre(lambda: t.status in fils_video.FINIS)
    assert t.status == "failed" and t.error_type == "disque"
    trop = catalogue(env, cache_go=50_000 / GO)
    m2 = moteurs(env, trop)
    with pytest.raises(ErreurDemande) as e:
        m2.creer({"prompt": "x", "lora": [{"nom": "a"}]})
    assert e.value.code == 507 and "cache_go" in e.value.message


def test_demandes_lora_mal_formees(env, moteurs):
    m = moteurs(env, catalogue(env))
    for lora in ([{"nom": "a"}, {"nom": "b"}], [{"nom": "inconnu"}], {"nom": "a"}, [{"nom": "a", "force": "fort"}]):
        with pytest.raises(ErreurDemande) as e:
            m.creer({"prompt": "x", "lora": lora})
        assert e.value.code == 400, lora


def test_annulation_pendant_la_fusion_efface_le_partiel(env, moteurs, monkeypatch):
    monkeypatch.setenv("FAUSSE_FUSION_PAS_S", "0.3")
    m = moteurs(env, catalogue(env))
    t = m.creer({"prompt": "x", "lora": [{"nom": "a"}]})
    attendre(lambda: t.etape == "fusion" and m.proc is not None)
    assert m.annuler() is True
    attendre(lambda: t.status == "cancelled", 5)
    assert not [n for n in os.listdir(env / "cache") if n.endswith(".partiel")]
    assert m.fusions_du_cache() == []


# supervisor and server

def port_libre() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def sup_video(env, monkeypatch):
    import pli
    from pli.gen.superviseur import Famille, Superviseur
    src = str(Path(pli.__file__).resolve().parent.parent)
    monkeypatch.setenv("PYTHONPATH", src)
    monkeypatch.setenv("PLI_GEN_PIDS", str(env / "pids"))
    faits = []

    def faire(memoire=lambda: 1 << 40, image=None):
        familles = {"video": Famille(nom="video", port=port_libre(), sante_delai_s=20, argv=[
            sys.executable, "-m", "pli.gen.fils_video", "--port", "{port}", "--simuler",
            "--sorties", str(env / "sorties")])}
        if image is not None:
            familles["image"] = Famille(nom="image", port=port_libre(), sante_delai_s=20, argv=[
                sys.executable, "-m", "pli.gen.fils_simule", "--port", "{port}", *image], memoire_octets=20 * GO)
        s = Superviseur(familles, reserve_octets=0, memoire_disponible=memoire, echange_octets=lambda: 0,
                        drapeau=str(env / "general.drapeau"))
        faits.append(s)
        return s
    yield faire
    for s in faits:
        s.arreter()


JSON = {"content-type": "application/json"}


def test_une_image_en_cours_pose_le_drapeau_des_autres_familles(sup_video, env):
    sup = sup_video(image=["--etapes", "4", "--duree-etape", "0.25"])
    autres = Path(sup.drapeau_autres)
    fil = threading.Thread(target=sup.relayer, args=("image", "POST", "/v1/images/generations", JSON, b"{}"))
    fil.start()
    attendre(autres.exists)
    code, _, corps = sup.relayer("video", "POST", "/v1/videos", JSON, b'{"prompt":"x"}')
    id_ = json.loads(corps)["id"]
    time.sleep(0.3)
    etat = json.loads(sup.relayer("video", "GET", f"/v1/videos/{id_}", {}, b"")[2])
    assert code == 202 and etat["status"] == "queued" and etat["attente"] == "image_ou_voix"
    fil.join(20)
    assert not autres.exists()
    attendre(lambda: json.loads(sup.relayer("video", "GET", f"/v1/videos/{id_}", {}, b"")[2])["status"] == "completed")


def test_delete_annule_puis_supprime_le_travail(sup_video, monkeypatch):
    monkeypatch.setenv("FAUX_H3_PAS_S", "0.5")
    sup = sup_video()
    id_ = json.loads(sup.relayer("video", "POST", "/v1/videos", JSON, b'{"prompt":"x"}')[2])["id"]
    attendre(lambda: json.loads(sup.relayer("video", "GET", f"/v1/videos/{id_}", {}, b"")[2])["status"] == "in_progress")
    code, _, corps = sup.relayer("video", "DELETE", f"/v1/videos/{id_}", {}, b"")
    assert code == 200 and json.loads(corps)["status"] == "cancelled"
    code, _, corps = sup.relayer("video", "DELETE", f"/v1/videos/{id_}", {}, b"")
    assert code == 200 and json.loads(corps) == {"id": id_, "object": "video.deleted", "deleted": True}
    assert sup.relayer("video", "GET", f"/v1/videos/{id_}", {}, b"")[0] == 404


def test_une_video_en_cours_n_est_pas_arretee_pour_faire_place(sup_video, monkeypatch):
    monkeypatch.setenv("FAUX_H3_PAS_S", "0.3")
    sup = sup_video(memoire=lambda: 10 * GO, image=[])
    assert sup.relayer("video", "POST", "/v1/videos", JSON, b'{"prompt":"x"}')[0] == 202
    code, _, corps = sup.relayer("image", "POST", "/v1/images/generations", JSON, b"{}")
    assert code == 503 and json.loads(corps)["error"]["type"] == "memoire"
    assert sup.etat_fils("video")["vivant"] is True
