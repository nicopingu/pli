"""Image generation child process (PLI 5) in --simuler mode: OpenAI API, catalog, preemption, reloading, supervisor."""

from __future__ import annotations

import base64
import http.client
import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import pli
from pli.gen import fils_image
from pli.gen.superviseur import GO, Famille, Superviseur

SCRIPT = Path(pli.__file__).resolve().parent / "gen" / "fils_image.py"
EXEMPLE = Path(pli.__file__).resolve().parent.parent.parent / "docs" / "gen" / "catalogue-image.exemple.json"
JSON = {"Content-Type": "application/json"}

CATALOGUE = {
    "defaut": "qwen21",
    "modeles": {
        "qwen21": {"classe": "qwen21", "chemin": "/x/qwen21", "etapes": 6, "guidance": 4.0, "taille": "256x256"},
        "qwen2512": {"classe": "qwen2512", "chemin": "/x/qwen2512", "etapes": 5, "guidance": 4.0},
        "flux3klein": {"classe": "flux3klein", "chemin": "/x/flux3", "etapes": 4},
    },
    "lora": {
        "lightning": {"chemin": "/x/l.safetensors", "base": ["qwen21"], "horaire": "lightning1", "etapes": 4, "guidance": 1.0},
        "omni": {"chemin": "/x/o.safetensors", "base": ["qwen21"], "force": 0.8},
        "q4p": {"chemin": "/x/q.safetensors", "base": ["qwen2512"],
                "cibles": [{"source": "img_mod.1", "modele": "img_mod_linear"}]},
    },
}


def port_libre() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def appel(port, methode, chemin, corps=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        c.request(methode, chemin, body=None if corps is None else json.dumps(corps), headers=JSON)
        r = c.getresponse()
        return r.status, json.loads(r.read() or b"null")
    finally:
        c.close()


def gen(port, **kw):
    return appel(port, "POST", "/v1/images/generations", {"prompt": "a cat", **kw})


@pytest.fixture
def catalogue(tmp_path):
    chemin = tmp_path / "catalogue.json"
    chemin.write_text(json.dumps(CATALOGUE))
    return chemin


@pytest.fixture
def fils(tmp_path, catalogue):
    """Launches fils_image.py --simuler with the current Python; returns (port, flag file)."""

    lances = []

    def lancer(*options):
        port = port_libre()
        drapeau = tmp_path / "general.drapeau"
        env = {**os.environ, "PLI_GEN_DRAPEAU": str(drapeau)}
        env.pop("PYTHONPATH", None)          # standalone: must not require pli
        proc = subprocess.Popen([sys.executable, "-I", str(SCRIPT), "--port", str(port), "--catalogue", str(catalogue),
                                 "--simuler", *map(str, options)], env=env, stdin=subprocess.DEVNULL)
        lances.append(proc)
        fin = time.monotonic() + 20
        while time.monotonic() < fin:
            try:
                if appel(port, "GET", "/sante")[0] == 200:
                    return port, drapeau
            except OSError:
                time.sleep(0.05)
        raise AssertionError("child not ready")

    yield lancer
    for p in lances:
        p.terminate()
        p.wait(5)


def test_generation_json_openai_et_png_valide(fils):
    port, _ = fils()
    code, rep = gen(port, size="512x256", seed=3)
    assert code == 200 and isinstance(rep["created"], int)
    png = base64.b64decode(rep["data"][0]["b64_json"], validate=True)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    largeur, hauteur = struct.unpack(">II", png[16:24])
    assert (largeur, hauteur) == (64, 64)         # the simulation caps at 64 pixels per side
    assert rep["pli"]["etapes"] == 6 and rep["pli"]["chargements"] == 1


def test_refus_400_et_501(fils):
    port, _ = fils()
    refus = [
        {"lora": [{"nom": "inconnu"}]},                                  # hors catalogue
        {"lora": [{"nom": "q4p"}]},                                      # base qwen2512 on qwen21
        {"lora": ["omni", "omni"]},                                      # doublon
        {"lora": [{"nom": "lightning"}], "steps": 8},                    # schedule at 4 steps
        {"lora": [{"nom": "lightning"}], "guidance": 4},                 # schedule without guidance
        {"model": "nul"}, {"n": 2}, {"response_format": "url"}, {"size": "100x100"},
        {"model": "flux3klein", "negative_prompt": "flou"}, {"model": "flux3klein", "guidance": 3},
        {"prompt": ""},
    ]
    for corps in refus:
        code, rep = gen(port, **corps)
        assert code == 400, (corps, rep)
        assert rep["error"]["message"]
    assert appel(port, "POST", "/v1/images/edits", {"prompt": "x"})[0] == 501
    assert appel(port, "POST", "/v1/images/inconnu", {})[0] == 404
    code, rep = gen(port, model="qwen2512", lora=[{"nom": "q4p", "force": 0.5}])
    assert code == 200 and rep["pli"]["etapes"] == 5
    code, rep = gen(port, lora=[{"nom": "lightning"}])             # enforced schedule: 4 steps
    assert code == 200 and rep["pli"]["etapes"] == 4 and rep["pli"]["horaire"] == "lightning1"


def test_preemption_attend_le_drapeau(fils):
    port, drapeau = fils("--duree-etape", 0.01)
    drapeau.write_text("")
    sortie = {}
    fil = threading.Thread(target=lambda: sortie.update(r=gen(port)))
    fil.start()
    time.sleep(0.5)
    assert fil.is_alive() and "r" not in sortie                      # flag set: no response
    drapeau.unlink()
    fil.join(15)
    code, rep = sortie["r"]
    assert code == 200 and rep["pli"]["attentes"] > 0                # flag removed: response
    code, rep = gen(port)
    assert rep["pli"]["attentes"] == 0                               # chemin libre : aucune attente


def test_changement_de_lora_recharge(fils):
    port, _ = fils()
    chargements = lambda r: r[1]["pli"]["chargements"]               # noqa: E731
    assert chargements(gen(port)) == 1
    assert chargements(gen(port, seed=9, steps=3)) == 1              # same key: no reloading
    assert chargements(gen(port, lora=["omni"])) == 2                # LoRA added
    assert chargements(gen(port, lora=[{"nom": "omni", "force": 0.8}])) == 2   # default force identical
    assert chargements(gen(port, lora=[{"nom": "omni", "force": 0.5}])) == 3   # force changed
    assert chargements(gen(port)) == 4                               # LoRA removed
    assert chargements(gen(port, model="qwen2512")) == 5             # other model
    etat = appel(port, "GET", "/etat")[1]
    assert etat["chargements"] == 5 and etat["cle"][0] == "qwen2512"


def test_catalogue_exemple_valide():
    cat = fils_image.charger_catalogue(str(EXEMPLE))
    assert set(cat["modeles"]) == {"qwen21", "qwen2512", "flux3klein", "krea2"}
    assert cat["modeles"]["krea2"]["quantize"] == 8
    assert {n for n, lo in cat["lora"].items() if lo["base"] == ["krea2"]} == {"style-krea2"}
    assert cat["lora"]["lightning-v1"]["horaire"] == "lightning1"
    assert cat["lora"]["style-qwen2512"]["base"] == ["qwen2512"]
    for entree in (cat["modeles"], cat["lora"]):
        assert all(v["chemin"].startswith("/chemin/pli/") for v in entree.values())   # aucun chemin de machine


def test_catalogue_invalide():
    mauvais = json.loads(json.dumps(CATALOGUE))
    mauvais["lora"]["omni"]["base"] = ["absent"]
    with pytest.raises(ValueError):
        fils_image.valider_catalogue(mauvais)
    mauvais = json.loads(json.dumps(CATALOGUE))
    mauvais["modeles"]["qwen21"]["classe"] = "sdxl"
    with pytest.raises(ValueError):
        fils_image.valider_catalogue(mauvais)


def test_bout_a_bout_superviseur_vers_fils_image(tmp_path, catalogue):
    port = port_libre()
    argv = [sys.executable, "-I", str(SCRIPT), "--port", "{port}", "--catalogue", str(catalogue), "--simuler",
            "--duree-etape", "0.01"]
    fam = Famille(nom="image", argv=argv, port=port, sante_delai_s=20)
    sup = Superviseur({"image": fam}, memoire_disponible=lambda: 64 * GO, echange_octets=lambda: 0,
                      reserve_octets=0, delai_kill_s=3.0)
    try:
        corps = json.dumps({"prompt": "a cat", "model": "qwen2512"}).encode()
        code, _, donnees = sup.relayer("image", "POST", "/v1/images/generations", JSON, corps)
        rep = json.loads(donnees)
        assert code == 200 and base64.b64decode(rep["data"][0]["b64_json"])[:4] == b"\x89PNG"
        mauvais = json.dumps({"prompt": "x", "lora": ["nul"]}).encode()
        assert sup.relayer("image", "POST", "/v1/images/generations", JSON, mauvais)[0] == 400
        # preemption by the supervisor: the general model is working, the child waits between two steps
        sup.general_debut()
        sortie = {}
        fil = threading.Thread(target=lambda: sortie.update(
            r=sup.relayer("image", "POST", "/v1/images/generations", JSON, corps)))
        fil.start()
        time.sleep(0.5)
        assert fil.is_alive()
        sup.general_fin()
        fil.join(15)
        code, _, donnees = sortie["r"]
        assert code == 200 and json.loads(donnees)["pli"]["attentes"] > 0
    finally:
        sup.arreter()


def test_sorties_gardees_sur_disque(fils, tmp_path):
    """Each produced image is kept with its request."""
    sorties = tmp_path / "sorties"
    port, _ = fils("--sorties", sorties)
    code, rep = gen(port, seed=7)
    assert code == 200
    chemin = rep["pli"]["fichier"]
    assert chemin.startswith(str(sorties)) and chemin.endswith(".png")
    with open(chemin, "rb") as f:
        assert f.read() == base64.b64decode(rep["data"][0]["b64_json"])
    with open(chemin[:-4] + ".json", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["demande"]["prompt"] == "a cat" and meta["demande"]["seed"] == 7
    code, rep2 = gen(port, seed=7)                 # same second possible: no overwriting
    assert code == 200 and rep2["pli"]["fichier"] != chemin


def test_sans_sorties_aucune_copie(fils):
    port, _ = fils()
    code, rep = gen(port)
    assert code == 200 and "fichier" not in rep["pli"]


def test_refus_503_memoire_par_modele(fils):
    # the test catalog has no memoire_go: defaults by class (qwen21 26 GB, flux3klein 16 GB)
    port, _ = fils("--memoire-disponible-go", 20)
    code, rep = gen(port)
    assert code == 503 and rep["error"]["type"] == "memoire" and "26.0 GB required" in rep["error"]["message"]
    code, rep = gen(port, model="flux3klein")
    assert code == 200                                                # 16 GB required, 20 disponibles


def test_memoire_go_du_catalogue_remplace_le_defaut(tmp_path):
    cat = fils_image.valider_catalogue({"modeles": {"a": {"classe": "qwen21", "chemin": "x", "memoire_go": 7}}})
    assert fils_image.besoin_go(cat, "a") == 7.0
    with pytest.raises(ValueError):
        fils_image.valider_catalogue({"modeles": {"a": {"classe": "qwen21", "chemin": "x", "memoire_go": 0}}})


KREA = {"defaut": "krea2",
        "modeles": {"krea2": {"classe": "krea2", "chemin": "/x/krea2", "quantize": 8, "horaire": "er_sde",
                              "etapes": 8, "guidance": 1.0}},
        "lora": {"style-krea2": {"chemin": "/x/k.safetensors", "base": ["krea2"]}}}


def test_catalogue_krea2():
    cat = fils_image.valider_catalogue(json.loads(json.dumps(KREA)))
    assert fils_image.besoin_go(cat, "krea2") == 29.0
    for cle, valeur in (("horaire", "lightning1"), ("quantize", 7), ("cibles", [{"source": "a", "modele": "a"}])):
        mauvais = json.loads(json.dumps(KREA))
        mauvais["modeles"]["krea2"][cle] = valeur
        with pytest.raises(ValueError):
            fils_image.valider_catalogue(mauvais)
    mauvais = json.loads(json.dumps(KREA))
    mauvais["lora"]["style-krea2"]["cibles"] = [{"source": "a", "modele": "a"}]
    with pytest.raises(ValueError):
        fils_image.valider_catalogue(mauvais)


def test_demandes_krea2():
    cat = fils_image.valider_catalogue(json.loads(json.dumps(KREA)))
    p = fils_image.resoudre({"prompt": "a cat", "lora": [{"nom": "style-krea2", "force": 0.7}]}, cat)
    assert (p.modele, p.horaire, p.etapes, p.guidance, p.loras) == ("krea2", "er_sde", 8, 1.0, [("style-krea2", 0.7)])
    assert fils_image.resoudre({"prompt": "a cat", "guidance": 3.0, "negative_prompt": "blurry"}, cat).negatif == "blurry"
    with pytest.raises(fils_image.ErreurDemande) as e:       # negative ignored by Krea 2 at guidance 1.0: hard refusal
        fils_image.resoudre({"prompt": "a cat", "negative_prompt": "blurry"}, cat)
    assert e.value.code == 400


def test_annulation_au_milieu_d_une_generation_puis_le_fils_resservit(fils):
    port, _ = fils("--duree-etape", 0.1)
    res = {}
    t = threading.Thread(target=lambda: res.update(r=gen(port, steps=40)))        # 40 steps of 0.1 s
    t0 = time.monotonic()
    t.start()
    time.sleep(0.5)
    assert appel(port, "POST", "/annuler") == (200, {"annulee": True})
    t.join(10)
    code, corps = res["r"]
    assert code == 499 and corps["error"]["type"] == "annulee"
    assert time.monotonic() - t0 < 3                                               # well before 4 s
    code, corps = gen(port, steps=2)
    assert code == 200 and corps["data"][0]["b64_json"]
    assert appel(port, "POST", "/annuler") == (200, {"annulee": False})           # plus rien en cours


def test_evaluer_embeddings_avant_le_saver():
    """The callback evaluates all prompt cache tables (positive and negative), not the rest."""

    class Tab:
        shape, dtype = (1, 2), "bf16"

    class Faux:
        def __init__(self):
            self.evalues = []

        def eval(self, *a):
            self.evalues.extend(a)

    a, b, c = Tab(), Tab(), Tab()
    modele = type("M", (), {"prompt_cache": {"chat": (a, b), "flou": (c, None), "x": "texte"}})()
    mx = Faux()
    fils_image._EvaluerEmbeddings(mx, modele).call_before_loop(1, "chat", None, None)
    assert mx.evalues == [a, b, c]
    fils_image._EvaluerEmbeddings(mx, type("M", (), {})()).call_before_loop(1, "chat", None, None)   # no cache: nothing
    assert mx.evalues == [a, b, c]
