"""Supervisor of generation child processes (PLI 5, step 1): simulated children, no model."""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

import pli
from pli.gen.config import familles_depuis
from pli.gen.superviseur import (GO, MO, Famille, Superviseur, analyser_swapusage, analyser_vm_stat)

SRC = str(Path(pli.__file__).resolve().parent.parent)
SAIN = lambda: 64 * GO          # noqa: E731 - comfortable available memory
JSON = {"Content-Type": "application/json"}

VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                     4494.
Pages active:                                1585511.
Pages inactive:                              1582398.
Pages speculative:                              1080.
Pages throttled:                                   0.
Pages wired down:                             538175.
Pages purgeable:                                  90.
"Translation faults":                     4194906116.
"""


def port_libre() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def famille(nom="audio", *options, **kw) -> Famille:
    port = port_libre()
    argv = [sys.executable, "-m", "pli.gen.fils_simule", "--port", "{port}", *map(str, options)]
    return Famille(nom=nom, argv=argv, port=port, sante_delai_s=20, **kw)


@pytest.fixture
def fabrique():
    faits: list[Superviseur] = []

    def faire(familles, **kw):
        kw.setdefault("memoire_disponible", SAIN)
        kw.setdefault("echange_octets", lambda: 0)
        kw.setdefault("reserve_octets", 0)
        kw.setdefault("delai_kill_s", 3.0)
        kw.setdefault("env", {"PYTHONPATH": SRC})
        sup = Superviseur({f.nom: f for f in familles}, **kw)
        faits.append(sup)
        return sup

    yield faire
    for sup in faits:
        sup.arreter()


def appel(sup, nom, chemin):
    return sup.relayer(nom, "POST", chemin, JSON, b"{}")


def attendre(condition, delai=10.0):
    fin = time.monotonic() + delai
    while time.monotonic() < fin:
        if condition():
            return True
        time.sleep(0.02)
    return False


def test_le_fils_demarre_au_premier_appel_seulement(fabrique):
    sup = fabrique([famille("audio")])
    assert not sup.etat_fils("audio")["vivant"]
    code, en_tetes, corps = appel(sup, "audio", "/v1/audio/transcriptions")
    assert code == 200 and json.loads(corps) == {"text": "simulated text"}
    assert sup.etat_fils("audio")["vivant"]
    code, en_tetes, corps = appel(sup, "audio", "/v1/audio/speech")
    assert code == 200 and en_tetes["Content-Type"] == "audio/wav" and corps.startswith(b"RIFF")


def test_famille_inconnue(fabrique):
    sup = fabrique([famille("audio")])
    assert appel(sup, "video", "/v1/videos")[0] == 404


def test_decharge_apres_le_delai(fabrique):
    sup = fabrique([famille("audio", delai_decharge_s=0.4)], intervalle_s=0.05)
    assert appel(sup, "audio", "/v1/audio/transcriptions")[0] == 200
    proc = sup._etats["audio"].fils.proc
    assert sup.etat_fils("audio")["vivant"]
    assert attendre(lambda: proc.poll() is not None)
    assert not sup.etat_fils("audio")["vivant"]
    # a following request restarts the child
    assert appel(sup, "audio", "/v1/audio/transcriptions")[0] == 200
    assert sup.etat_fils("audio")["pid"] != proc.pid


def test_refus_503_quand_la_memoire_manque_sans_lancer(fabrique):
    fam = famille("image", memoire_octets=10 * GO)
    sup = fabrique([fam], memoire_disponible=lambda: 12 * GO, reserve_octets=6 * GO)
    code, _, corps = appel(sup, "image", "/v1/images/generations")
    erreur = json.loads(corps)["error"]
    assert code == 503 and erreur["type"] == "memoire"
    assert erreur["disponible_octets"] == 12 * GO and erreur["requis_octets"] == 16 * GO
    assert sup._etats["image"].fils is None                   # no process started
    with socket.socket() as s:                                  # and nothing is listening on the port
        assert s.connect_ex(("127.0.0.1", fam.port)) != 0


def test_la_memoire_suffisante_laisse_lancer(fabrique):
    sup = fabrique([famille("image", memoire_octets=10 * GO)], memoire_disponible=lambda: 16 * GO,
                   reserve_octets=6 * GO)
    assert appel(sup, "image", "/v1/images/generations")[0] == 200


def _dispo_selon_audio(etat):
    def dispo():
        sup = etat.get("sup")
        f = sup._etats["audio"].fils if sup is not None else None
        return 10 * GO if f is not None and f.proc.poll() is None else 30 * GO
    return dispo


def test_une_famille_inactive_est_arretee_pour_laisser_la_place(fabrique):
    etat = {}
    sup = fabrique([famille("audio", memoire_octets=4 * GO), famille("image", memoire_octets=20 * GO)],
                   memoire_disponible=_dispo_selon_audio(etat), reserve_octets=4 * GO)
    etat["sup"] = sup
    assert appel(sup, "audio", "/v1/audio/transcriptions")[0] == 200
    proc = sup._etats["audio"].fils.proc
    code, _, corps = appel(sup, "image", "/v1/images/generations")
    assert code == 200, corps
    assert sup._etats["audio"].fils is None and attendre(lambda: proc.poll() is not None)


def test_une_famille_occupee_n_est_pas_arretee(fabrique):
    etat = {}
    sup = fabrique([famille("audio", memoire_octets=4 * GO), famille("image", memoire_octets=20 * GO)],
                   memoire_disponible=_dispo_selon_audio(etat), reserve_octets=4 * GO)
    etat["sup"] = sup
    assert appel(sup, "audio", "/v1/audio/transcriptions")[0] == 200
    sup._etats["audio"].actifs = 1                            # a voice request in progress
    code, _, corps = appel(sup, "image", "/v1/images/generations")
    assert code == 503 and json.loads(corps)["error"]["type"] == "memoire"
    assert sup._etats["audio"].fils is not None and sup._etats["audio"].fils.proc.poll() is None
    sup._etats["audio"].actifs = 0


def test_plantage_du_fils_502_puis_relance(fabrique):
    sup = fabrique([famille("audio", "--planter-apres", 1)])
    assert appel(sup, "audio", "/v1/audio/transcriptions")[0] == 200
    sup._etats["audio"].fils.proc.wait(10)                      # the child killed itself after its first response
    code, _, corps = appel(sup, "audio", "/v1/audio/transcriptions")
    erreur = json.loads(corps)["error"]
    assert code == 502 and erreur["message"] == "child process stopped" and erreur["code_sortie"] == 3
    assert appel(sup, "audio", "/v1/audio/transcriptions")[0] == 200       # restarted


def test_preemption_entre_etapes_par_le_drapeau(fabrique):
    sup = fabrique([famille("image", "--etapes", 3, "--duree-etape", 0.05)])
    sup.general_debut()
    assert os.path.exists(sup.drapeau)
    resultat = []
    fil = threading.Thread(target=lambda: resultat.append(appel(sup, "image", "/v1/images/generations")))
    fil.start()
    time.sleep(1.0)
    assert resultat == []                                      # as long as the flag exists, no response
    sup.general_fin()
    assert not os.path.exists(sup.drapeau)
    fil.join(10)
    code, _, corps = resultat[0]
    assert code == 200 and json.loads(corps)["simule"]["attentes"] > 0


def test_le_drapeau_tient_tant_qu_une_demande_generale_dure(fabrique):
    sup = fabrique([famille("image")])
    sup.general_debut()
    sup.general_debut()
    sup.general_fin()
    assert os.path.exists(sup.drapeau)
    sup.general_fin()
    assert not os.path.exists(sup.drapeau)
    sup.general_fin()                                          # an extra end does not break anything
    assert not os.path.exists(sup.drapeau)


def test_hausse_de_l_echange_arrete_le_fils_image(fabrique):
    echange = [0]
    sup = fabrique([famille("image", "--etapes", 100, "--duree-etape", 0.1)], echange_octets=lambda: echange[0],
                   intervalle_s=0.05, seuil_echange_octets=100 * MO)
    resultat = []
    fil = threading.Thread(target=lambda: resultat.append(appel(sup, "image", "/v1/images/generations")))
    debut = time.monotonic()
    fil.start()
    assert attendre(lambda: sup.etat_fils("image")["actifs"] == 1)
    proc = sup._etats["image"].fils.proc
    echange[0] = 500 * MO
    fil.join(8)
    code, _, corps = resultat[0]
    erreur = json.loads(corps)["error"]
    assert code == 502 and erreur["type"] == "echange" and "swap" in erreur["message"]
    assert time.monotonic() - debut < 8 and proc.poll() is not None
    assert not sup.etat_fils("image")["vivant"]


def test_hausse_de_l_echange_arrete_aussi_le_fils_audio(fabrique):
    echange = [0]
    sup = fabrique([famille("audio", "--etapes", 100, "--duree-etape", 0.1)], echange_octets=lambda: echange[0],
                   intervalle_s=0.05, seuil_echange_octets=100 * MO)
    resultat = []
    fil = threading.Thread(target=lambda: resultat.append(appel(sup, "audio", "/v1/images/generations")))
    fil.start()
    assert attendre(lambda: sup.etat_fils("audio")["actifs"] == 1)
    echange[0] = 500 * MO
    fil.join(8)
    code, _, corps = resultat[0]
    assert code == 502 and json.loads(corps)["error"]["type"] == "echange"
    assert not sup.etat_fils("audio")["vivant"]


def test_echange_avec_memoire_disponible_ne_tue_pas(fabrique):
    echange, dispo = [0], [30 * GO]
    sup = fabrique([famille("audio", "--etapes", 40, "--duree-etape", 0.05)], echange_octets=lambda: echange[0],
                   memoire_disponible=lambda: dispo[0], intervalle_s=0.05, seuil_echange_octets=100 * MO,
                   dispo_min_echange_octets=4 * GO)
    res = {}
    t = threading.Thread(target=lambda: res.setdefault("r", appel(sup, "audio", "/v1/images/generations")))
    t.start()
    assert attendre(lambda: sup._etats["audio"].actifs == 1)
    echange[0] = 5 * GO                                   # load spike, memory still available
    t.join(20)
    assert res["r"][0] == 200


def test_echange_et_memoire_basse_tue(fabrique):
    echange, dispo = [0], [30 * GO]
    sup = fabrique([famille("audio", "--etapes", 100, "--duree-etape", 0.1)], echange_octets=lambda: echange[0],
                   memoire_disponible=lambda: dispo[0], intervalle_s=0.05, seuil_echange_octets=100 * MO,
                   dispo_min_echange_octets=4 * GO)
    res = {}
    t = threading.Thread(target=lambda: res.setdefault("r", appel(sup, "audio", "/v1/images/generations")))
    t.start()
    assert attendre(lambda: sup._etats["audio"].actifs == 1)
    echange[0], dispo[0] = 5 * GO, 2 * GO
    t.join(20)
    assert res["r"][0] == 502 and json.loads(res["r"][2])["error"]["type"] == "echange"


def test_surveiller_echange_faux_laisse_vivre_le_fils(fabrique):
    echange = [0]
    sup = fabrique([famille("audio", "--etapes", 20, "--duree-etape", 0.05, surveiller_echange=False)],
                   echange_octets=lambda: echange[0], seuil_echange_octets=MO)
    resultat = []
    fil = threading.Thread(target=lambda: resultat.append(appel(sup, "audio", "/v1/images/generations")))
    fil.start()
    assert attendre(lambda: sup.etat_fils("audio")["actifs"] == 1)
    echange[0] = 10 * GO
    sup.balayer()
    fil.join(10)
    assert resultat[0][0] == 200


def test_reserve_propre_a_la_famille(fabrique):
    # 10 Go available: the global reserve of 6 Go refuses (8 + 6 = 14), the family's 1 Go reserve accepts (8 + 1 = 9)
    sup = fabrique([famille("audio", memoire_octets=8 * GO)], memoire_disponible=lambda: 10 * GO, reserve_octets=6 * GO)
    code, _, corps = appel(sup, "audio", "/v1/audio/transcriptions")
    assert code == 503 and json.loads(corps)["error"]["type"] == "memoire"
    sup = fabrique([famille("audio", memoire_octets=8 * GO, reserve_octets=1 * GO)],
                   memoire_disponible=lambda: 10 * GO, reserve_octets=6 * GO)
    assert appel(sup, "audio", "/v1/audio/transcriptions")[0] == 200


def test_analyse_de_vm_stat_et_de_swapusage():
    attendu = (4494 + 1582398 + 1080 + 90) * 16384
    assert analyser_vm_stat(VM_STAT) == attendu
    assert analyser_swapusage("vm.swapusage: total = 2048.00M  used = 522.94M  free = 1525.06M  (encrypted)") == \
        int(522.94 * MO)
    assert analyser_swapusage("vm.swapusage: total = 2048,00M  used = 1,5G  free = 1525,06M  (encrypted)") == \
        int(1.5 * GO)
    with pytest.raises(ValueError):
        analyser_vm_stat("rien")


def test_configuration_des_familles():
    familles = familles_depuis({"familles": {
        "image": {"argv": ["{python}", "-m", "x", "--port", "{port}"], "port": 18101, "memoire_go": 8,
                  "delai_decharge_s": 120}}})
    image = familles["image"]
    assert image.argv[0] == sys.executable and image.memoire_octets == 8 * GO and image.delai_decharge_s == 120
    with pytest.raises(ValueError):
        familles_depuis({"familles": {"musique": {"argv": ["x"], "port": 1}}})


def test_client_parti_appelle_annuler_sur_le_fils(tmp_path):
    """The client closes its connection during the relay: the child receives POST /annuler (client_parti probe)."""

    marque = tmp_path / "annule"
    script = tmp_path / "faux_fils.py"
    script.write_text(
        "import sys, time, pathlib\n"
        "from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer\n"
        "class H(BaseHTTPRequestHandler):\n"
        "    def log_message(self, *a): pass\n"
        "    def _rep(self, c, b=b'{}'):\n"
        "        self.send_response(c); self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)\n"
        "    def do_GET(self): self._rep(200)\n"
        "    def do_POST(self):\n"
        "        n = int(self.headers.get('Content-Length') or 0); self.rfile.read(n)\n"
        f"        if self.path == '/annuler': pathlib.Path({str(marque)!r}).write_text('oui'); self._rep(200); return\n"
        "        time.sleep(2); self._rep(200)\n"
        "ThreadingHTTPServer(('127.0.0.1', int(sys.argv[1])), H).serve_forever()\n")
    port = port_libre()
    fam = Famille(nom="audio", argv=[sys.executable, str(script), "{port}"], port=port, sante_delai_s=20)
    sup = Superviseur({"audio": fam}, memoire_disponible=SAIN, echange_octets=lambda: 0, reserve_octets=0,
                      delai_kill_s=3.0)
    debut = time.monotonic()
    try:
        sup.relayer("audio", "POST", "/v1/audio/speech", JSON, b"{}",
                    client_parti=lambda: time.monotonic() - debut > 0.7)
        assert marque.exists()
    finally:
        sup.arreter()


def test_le_delai_depasse_appelle_annuler_sur_le_fils(tmp_path):
    """The relay gives up (504): the child receives POST /annuler. Fake child: slow, it logs the received call."""

    marque = tmp_path / "annule"
    script = tmp_path / "faux_fils.py"
    script.write_text(
        "import sys, time, pathlib\n"
        "from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer\n"
        "class H(BaseHTTPRequestHandler):\n"
        "    def log_message(self, *a): pass\n"
        "    def _rep(self, c, b=b'{}'):\n"
        "        self.send_response(c); self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)\n"
        "    def do_GET(self): self._rep(200)\n"
        "    def do_POST(self):\n"
        "        n = int(self.headers.get('Content-Length') or 0); self.rfile.read(n)\n"
        f"        if self.path == '/annuler': pathlib.Path({str(marque)!r}).write_text('oui'); self._rep(200); return\n"
        "        time.sleep(3); self._rep(200)\n"
        "ThreadingHTTPServer(('127.0.0.1', int(sys.argv[1])), H).serve_forever()\n")
    port = port_libre()
    fam = Famille(nom="audio", argv=[sys.executable, str(script), "{port}"], port=port, sante_delai_s=20, relais_delai_s=0.4)
    sup = Superviseur({"audio": fam}, memoire_disponible=SAIN, echange_octets=lambda: 0, reserve_octets=0,
                      delai_kill_s=3.0)
    try:
        code, _, corps = sup.relayer("audio", "POST", "/v1/audio/speech", JSON, b"{}")
        assert code == 504 and json.loads(corps)["error"]["type"] == "delai"
        assert attendre(marque.exists, 3.0)
    finally:
        sup.arreter()


# ------------------------------------------------------------------------- PLI 5.6.4: room asked, general model set aside

def test_la_place_est_demandee_a_la_file_puis_le_fils_lance(fabrique, tmp_path):
    """Short of memory: the request file is written, the supervisor waits; memory comes back (the queue paused a
    passage), the child starts and the file stays while it lives, then goes."""

    fichier = tmp_path / "besoin-place.json"
    dispo = [12 * GO]
    vu = {}

    def memoire():
        if fichier.exists() and "lu" not in vu:
            vu["lu"] = json.loads(fichier.read_text())
            dispo[0] = 20 * GO                                    # the queue made room
        return dispo[0]

    fam = famille("image", memoire_octets=10 * GO, delai_decharge_s=0.5)
    sup = fabrique([fam], memoire_disponible=memoire, reserve_octets=6 * GO, intervalle_s=0.05,
                   place_fichier=str(fichier), place_attente_s=10)
    assert appel(sup, "image", "/v1/images/generations")[0] == 200
    assert vu["lu"]["famille"] == "image" and vu["lu"]["manque_octets"] == 4 * GO and vu["lu"]["requis_octets"] == 16 * GO
    assert fichier.exists() and sup.places_demandees == 1           # kept while the child lives
    assert vu["lu"]["attente"] is True and json.loads(fichier.read_text())["attente"] is False   # memory obtained
    proc = sup._etats["image"].fils.proc
    assert attendre(lambda: proc.poll() is not None)               # unloaded after 0.5 s idle
    assert attendre(lambda: not fichier.exists())


def test_sans_place_au_bout_du_delai_refus_et_fichier_retire(fabrique, tmp_path):
    fichier = tmp_path / "besoin-place.json"
    sup = fabrique([famille("image", memoire_octets=10 * GO)], memoire_disponible=lambda: 12 * GO,
                   reserve_octets=6 * GO, place_fichier=str(fichier), place_attente_s=1.5)
    code, _, corps = appel(sup, "image", "/v1/images/generations")
    assert code == 503 and json.loads(corps)["error"]["type"] == "memoire"
    assert not fichier.exists() and sup._etats["image"].fils is None


def test_le_general_mis_de_cote_quand_la_file_ne_suffit_pas_puis_rendu(fabrique, tmp_path):
    """Order: room from the queue first, then the general model set aside; it comes back once the child is gone."""

    fichier = tmp_path / "besoin-place.json"
    dispo, appels = [12 * GO], []

    def mettre_de_cote(octets):
        appels.append(("cote", octets, fichier.exists()))
        dispo[0] = 30 * GO
        return 18 * GO

    fam = famille("image", memoire_octets=10 * GO, delai_decharge_s=0.5)
    sup = fabrique([fam], memoire_disponible=lambda: dispo[0], reserve_octets=6 * GO, intervalle_s=0.05,
                   place_fichier=str(fichier), place_attente_s=1.0, mettre_de_cote=mettre_de_cote,
                   rendre=lambda: appels.append(("rendu",)))
    assert appel(sup, "image", "/v1/images/generations")[0] == 200
    assert appels[0] == ("cote", 4 * GO, True) and sup.mises_de_cote == 1      # asked of the queue before
    proc = sup._etats["image"].fils.proc
    assert attendre(lambda: proc.poll() is not None)
    assert attendre(lambda: ("rendu",) in appels)


def test_sans_fichier_de_place_rien_ne_change(fabrique, tmp_path):
    sup = fabrique([famille("image", memoire_octets=10 * GO)], memoire_disponible=lambda: 12 * GO,
                   reserve_octets=6 * GO)
    assert appel(sup, "image", "/v1/images/generations")[0] == 503 and sup.places_demandees == 0


def test_un_fichier_laisse_par_un_serveur_arrete_est_retire(fabrique, tmp_path):
    fichier = tmp_path / "besoin-place.json"
    fichier.write_text("{}")
    fabrique([famille("image")], place_fichier=str(fichier))
    assert not fichier.exists()


def test_configuration_place(tmp_path, monkeypatch):
    from pli.gen import config

    f = tmp_path / "gen.json"
    f.write_text(json.dumps({"place_fichier": str(tmp_path / "p.json"), "place_attente_s": 30,
                             "familles": {"image": {"argv": ["x"], "port": 1, "memoire_go": 1}}}))
    monkeypatch.setenv("PLI_GEN_CONFIG", str(f))
    sup = config.superviseur_depuis_env()
    try:
        assert sup.place_fichier == str(tmp_path / "p.json") and sup.place_attente_s == 30
    finally:
        sup.arreter()


def test_la_sante_donne_le_modele_servi_par_le_fils(fabrique):
    sup = fabrique([famille("image")])
    assert sup.etat_general()["image"]["modele"] is None
    assert sup.relayer("image", "POST", "/v1/images/generations", JSON, b'{"model": "qwen21", "prompt": "x"}')[0] == 200
    g = sup.etat_general()["image"]
    assert g["etat"] == "pret" and g["modele"] == "qwen21" and g["actifs"] == 0
