"""Private LoRA catalogues (PLI_GEN_CATALOGUE_PRIVE, TOML files): reading, merging, applied defaults (without model)."""

from __future__ import annotations

import json
import os
import time

import pytest

from pli.gen import catalogue_prive, fils_image, fils_video
from pli.gen.config import verifier_catalogues_prives
from pli.gen.superviseur import Famille

TOML = """
[image.prive-krea]
fichier = "/x/prive-krea.safetensors"
base = "krea2"
force = 0.9
defauts = {passes = 6, taille = "768x768"}

[image.prive-qwen]
fichier = "/x/prive-qwen.safetensors"
base = ["qwen2512"]
cibles = [{source = "img_mod.1", modele = "img_mod_linear"}]
defauts = {passes = 20, guidance = 3.5}

[video.prive-video]
fichier = "/x/prive-video.safetensors"
mode = "sauvegardee"
taille_go = 33
defauts = {passes = 8, taille = "640x384", secondes = 3}
"""


def ecrire(tmp_path, texte, nom="prive.toml"):
    f = tmp_path / nom
    f.write_text(texte, encoding="utf-8")
    return str(f)


def env_prive(*chemins):
    return {catalogue_prive.ENV: os.pathsep.join(chemins)}


def catalogue_image(tmp_path):
    c = {"defaut": "qwen21",
         "modeles": {"qwen21": {"classe": "qwen21", "chemin": "/m/q21", "etapes": 40, "guidance": 4.0},
                     "qwen2512": {"classe": "qwen2512", "chemin": "/m/q2512", "etapes": 30, "guidance": 4.0},
                     "krea2": {"classe": "krea2", "chemin": "/m/k", "horaire": "er_sde", "etapes": 8, "guidance": 1.0}},
         "lora": {"public": {"chemin": "/x/public.safetensors", "base": ["krea2"]}}}
    f = tmp_path / "catalogue-image.json"
    f.write_text(json.dumps(c))
    return str(f)


def test_absent_rien_ne_change(tmp_path):
    assert catalogue_prive.chemins({}) == [] and catalogue_prive.chemins({catalogue_prive.ENV: " "}) == []
    cat = fils_image.charger_catalogue(catalogue_image(tmp_path), environ={})
    assert set(cat["lora"]) == {"public"}


def test_lecture_et_conversion(tmp_path):
    e = catalogue_prive.lire([ecrire(tmp_path, TOML)])
    assert set(e["image"]) == {"prive-krea", "prive-qwen"} and set(e["video"]) == {"prive-video"}
    k = e["image"]["prive-krea"]
    assert k == {"chemin": "/x/prive-krea.safetensors", "base": ["krea2"], "force": 0.9, "prive": True,
                 "defauts": {"passes": 6, "taille": "768x768"}}
    assert e["image"]["prive-qwen"]["cibles"] == [{"source": "img_mod.1", "modele": "img_mod_linear"}]
    v = e["video"]["prive-video"]
    assert v["base"] == ["h3"] and v["mode"] == "sauvegardee" and v["taille_go"] == 33.0
    assert v["defauts"] == {"passes": 8, "taille": "640x384", "secondes": 3.0}


@pytest.mark.parametrize("texte, motif", [
    ('[image.a]\nfichier = "/x"\nbase = "krea2"\npasses = 8\n', "unknown key(s) passes"),
    ('[image.a]\nbase = "krea2"\n', "fichier"),
    ('[image.a]\nfichier = "/x"\n', "base"),
    ('[image.a]\nfichier = "/x"\nbase = "krea2"\ndefauts = {passes = 0}\n', "passes"),
    ('[image.a]\nfichier = "/x"\nbase = "krea2"\ndefauts = {taille = "grand"}\n', "taille"),
    ('[image.a]\nfichier = "/x"\nbase = "krea2"\ndefauts = {secondes = 2}\n', "unknown key(s)"),
    ('[video.a]\nfichier = "/x"\nmode = "toujours"\n', "mode"),
    ('[audio.a]\nfichier = "/x"\n', "unknown section(s)"),
    ('[image.a\n', "invalid TOML"),
])
def test_erreurs_claires(tmp_path, texte, motif):
    with pytest.raises(ValueError) as e:
        catalogue_prive.lire([ecrire(tmp_path, texte)])
    assert motif in str(e.value)


def test_nom_public_ou_double_refuse(tmp_path):
    double = ecrire(tmp_path, '[image.public]\nfichier = "/x/p"\nbase = "krea2"\n', "double.toml")
    with pytest.raises(ValueError) as e:
        fils_image.charger_catalogue(catalogue_image(tmp_path), environ=env_prive(double))
    assert "already in the public catalog" in str(e.value)
    a = ecrire(tmp_path, '[image.x]\nfichier = "/a"\nbase = "krea2"\n', "a.toml")
    b = ecrire(tmp_path, '[image.x]\nfichier = "/b"\nbase = "krea2"\n', "b.toml")
    with pytest.raises(ValueError) as e:
        catalogue_prive.lire([a, b])
    assert "defined twice" in str(e.value)


def test_image_defauts_appliques_sans_les_imposer(tmp_path):
    cat = fils_image.charger_catalogue(catalogue_image(tmp_path), environ=env_prive(ecrire(tmp_path, TOML)))
    assert {"public", "prive-krea", "prive-qwen"} <= set(cat["lora"])
    p = fils_image.resoudre({"prompt": "x", "model": "krea2", "lora": [{"nom": "prive-krea"}]}, cat)
    assert p.etapes == 6 and p.loras == [("prive-krea", 0.9)] and (p.largeur, p.hauteur) == (768, 768)
    p2 = fils_image.resoudre({"prompt": "x", "model": "krea2", "steps": 4, "size": "512x512",
                              "lora": [{"nom": "prive-krea", "force": 1.2}]}, cat)
    assert p2.etapes == 4 and p2.loras == [("prive-krea", 1.2)] and p2.largeur == 512     # the request takes precedence
    q = fils_image.resoudre({"prompt": "x", "model": "qwen2512", "lora": ["prive-qwen"]}, cat)
    assert q.etapes == 20 and q.guidance == 3.5
    sans = fils_image.resoudre({"prompt": "x", "model": "krea2"}, cat)
    assert sans.etapes == 8 and sans.largeur == 1024                          # without LoRA: the model


def test_video_defauts_appliques(tmp_path, monkeypatch):
    for nom in ("PLI_GEN_DRAPEAU", "PLI_GEN_DRAPEAU_AUTRES"):
        monkeypatch.delenv(nom, raising=False)
    base = tmp_path / "MiniMax-H3"
    base.mkdir()
    lora = tmp_path / "prive-video.safetensors"
    lora.write_bytes(b"lora")
    toml = ecrire(tmp_path, TOML.replace("/x/prive-video.safetensors", str(lora)))
    c = {"defaut": "h3", "modeles": {"h3": {"chemin": str(base)}}, "lora": {},
         "dossier_fusions": str(tmp_path / "fusions"), "cache_fusions": str(tmp_path / "cache"), "fusion": {"marge_go": 0}}
    f = tmp_path / "catalogue-video.json"
    f.write_text(json.dumps(c))
    monkeypatch.setenv(catalogue_prive.ENV, toml)
    cat = fils_video.charger_catalogue(str(f))
    assert cat["lora"]["prive-video"]["prive"] is True
    m = fils_video.Moteur("", "", str(tmp_path / "s"), simuler=True, catalogue=cat)
    try:
        t = m.creer({"prompt": "x", "lora": [{"nom": "prive-video"}]})
        assert t.params["steps"] == 8 and t.params["size"] == "640x384" and t.params["seconds"] == 3.0
        t2 = m.creer({"prompt": "x", "lora": [{"nom": "prive-video"}], "steps": 4, "size": "512x512", "frames": 9})
        assert t2.params["steps"] == 4 and t2.params["size"] == "512x512" and "seconds" not in t2.params
    finally:
        m.arreter()


def test_verification_au_demarrage(tmp_path):
    cat = catalogue_image(tmp_path)
    fam = {"image": Famille(nom="image", argv=["py", "fils_image.py", "--catalogue", cat], port=1)}
    assert verifier_catalogues_prives(fam, environ={}) == {}
    noms = verifier_catalogues_prives(fam, environ=env_prive(ecrire(tmp_path, TOML)))
    assert noms == {"image": ["prive-krea", "prive-qwen"], "video": ["prive-video"]}
    double = ecrire(tmp_path, '[image.public]\nfichier = "/x/p"\nbase = "krea2"\n', "double.toml")
    with pytest.raises(ValueError):
        verifier_catalogues_prives(fam, environ=env_prive(double))


def test_cle_catalogue_prive_de_la_config_passee_aux_fils_seulement(tmp_path, monkeypatch):
    """« catalogue_prive » of PLI_GEN_CONFIG: in the child processes' environment, not the server's (checkpoints key)."""

    from pli.gen import config
    toml = ecrire(tmp_path, TOML)
    cat = catalogue_image(tmp_path)
    cfg = tmp_path / "pli-gen.json"
    cfg.write_text(json.dumps({"catalogue_prive": [toml], "familles": {
        "image": {"argv": ["{python}", "fils_image.py", "--catalogue", cat], "port": 18999, "memoire_go": 1}}}))
    monkeypatch.setenv("PLI_GEN_CONFIG", str(cfg))
    monkeypatch.delenv(catalogue_prive.ENV, raising=False)
    sup = config.superviseur_depuis_env()
    try:
        assert sup.env == {catalogue_prive.ENV: toml}
        assert catalogue_prive.ENV not in os.environ
    finally:
        sup.arreter()
    monkeypatch.setenv(catalogue_prive.ENV, toml)                      # both at the same time: refused
    with pytest.raises(ValueError):
        config.superviseur_depuis_env()
    monkeypatch.delenv(catalogue_prive.ENV)
    double = ecrire(tmp_path, '[image.public]\nfichier = "/x/p"\nbase = "krea2"\n', "double.toml")
    cfg.write_text(json.dumps({"catalogue_prive": double, "familles": {
        "image": {"argv": ["{python}", "fils_image.py", "--catalogue", cat], "port": 18999, "memoire_go": 1}}}))
    with pytest.raises(ValueError):                                     # name already public: startup fails
        config.superviseur_depuis_env()
