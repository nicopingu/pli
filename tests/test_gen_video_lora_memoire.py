"""PLI 5.6.5: video LoRA applied in memory by h3 (--lora), several at once (simulated h3 and conversion)."""
from __future__ import annotations

import os

import pytest

from pli.gen.fils_video import GO, LORA_MAX, ErreurDemande, Moteur, valider_catalogue

from test_gen_video_file import argv_h3, attendre, env, moteurs  # noqa: F401  (fixtures)


def catalogue(env, **plus):
    base = env / "MiniMax-H3"
    base.mkdir(exist_ok=True)
    loras = {}
    for nom, mode in (("turbo", "en_memoire"), ("style", "en_memoire"), ("motion", "en_memoire"), ("vieux", "sauvegardee"),
                      ("e", "en_memoire"), ("f", "en_memoire")):
        f = env / "lora" / f"{nom}.safetensors"
        f.parent.mkdir(exist_ok=True)
        f.write_bytes(b"lora" * 100)
        loras[nom] = {"chemin": str(f), "force": 1.0, "mode": mode, "taille_go": 100_000 / GO}
    loras["turbo"]["steps"] = 8
    loras["turbo"]["defauts"] = {"taille": "512x512", "secondes": 2}
    loras["style"]["defauts"] = {"taille": "864x480", "secondes": 5}
    c = {"defaut": "h3", "modeles": {"h3": {"chemin": str(base)}}, "lora": loras,
         "dossier_fusions": str(env / "sauvegardes"), "cache_fusions": str(env / "cache"), "cache_go": 250_000 / GO,
         "fusion": {"marge_go": 0}, "lora_natif": {"dossier": str(env / "natif")}}
    c.update(plus)
    return valider_catalogue(c)


def test_deux_lora_en_memoire_passes_a_h3_sans_fusion(env, moteurs):
    m = moteurs(env, catalogue(env))
    t = m.creer({"prompt": "x", "lora": [{"nom": "turbo"}, {"nom": "style", "force": 0.7}]})
    attendre(lambda: t.status == "completed")
    argv = argv_h3(m, t)
    natif = sorted(os.listdir(env / "natif"))
    assert len(natif) == 2 and all(n.endswith(".safetensors") for n in natif)
    assert all(f"--lora {f}@" in argv for f in t.fusion["fichiers"])
    assert argv.count("--lora ") == 2 and "@0.7" in argv
    assert f"-d {env / 'MiniMax-H3'} " in argv                  # base model, no merged copy
    assert not (env / "cache").exists() and not (env / "sauvegardes").exists()
    assert t.fusion["mode"] == "en_memoire" and len(t.fusion["fichiers"]) == 2
    v = m.vue(t)
    assert v["lora"] == [{"nom": "turbo", "force": 1.0}, {"nom": "style", "force": 0.7}]
    # defaults: those of the first LoRA that has them (turbo: 8 steps, 512x512, 2 s)
    assert "--steps 8" in argv and "--width 512 --height 512" in argv and "--seconds 2" in argv


def test_la_conversion_est_faite_une_fois(env, moteurs):
    m = moteurs(env, catalogue(env))
    t = m.creer({"prompt": "x", "lora": [{"nom": "motion"}]})
    attendre(lambda: t.status == "completed")
    fichier = t.fusion["fichiers"][0]
    avant = os.stat(fichier).st_mtime_ns
    u = m.creer({"prompt": "y", "lora": [{"nom": "motion", "force": 0.5}]})
    attendre(lambda: u.status == "completed")
    assert u.fusion["fichiers"] == [fichier] and os.stat(fichier).st_mtime_ns == avant
    assert "@0.5" in argv_h3(m, u)


def test_combinaison_refusee_avec_une_lora_fusionnee(env, moteurs):
    m = moteurs(env, catalogue(env))
    for lora in ([{"nom": "turbo"}, {"nom": "vieux"}], [{"nom": "vieux"}, {"nom": "turbo"}],
                 [{"nom": "turbo"}, {"nom": "turbo"}],
                 [{"nom": n} for n in ("turbo", "style", "motion", "e", "f")][:LORA_MAX + 1]):
        with pytest.raises(ErreurDemande) as e:
            m.creer({"prompt": "x", "lora": lora})
        assert e.value.code == 400, lora


def test_une_lora_fusionnee_seule_garde_la_fusion(env, moteurs, monkeypatch):
    m = moteurs(env, catalogue(env))
    t = m.creer({"prompt": "x", "lora": [{"nom": "vieux"}]})
    attendre(lambda: t.status == "completed")
    assert t.fusion["source"] == "nouvelle" and "--lora" not in argv_h3(m, t)


def test_echec_de_conversion_rapporte(env, moteurs, monkeypatch):
    monkeypatch.setenv("FAUX_NATIF_ECHEC", "1")
    m = moteurs(env, catalogue(env))
    t = m.creer({"prompt": "x", "lora": [{"nom": "turbo"}]})
    attendre(lambda: t.status == "failed")
    assert t.error_type == "lora" and "conversion" in t.error
    assert not os.listdir(env / "natif")


def test_sans_dossier_natif_503(env, moteurs):
    m = moteurs(env, catalogue(env, lora_natif={}))
    with pytest.raises(ErreurDemande) as e:
        m.creer({"prompt": "x", "lora": [{"nom": "turbo"}]})
    assert e.value.code == 503 and e.value.type == "lora"


def test_besoin_memoire_compte_les_lora(env, moteurs):
    m = moteurs(env, catalogue(env), memoire_disponible=lambda: int(12.3 * GO))
    m.creer({"prompt": "x"})                                          # 12 GB: accepted
    m2 = moteurs(env, catalogue(env), memoire_disponible=lambda: int(12.3 * GO))
    with pytest.raises(ErreurDemande) as e:                           # + 0.6 GB delta buffer: refused
        m2.creer({"prompt": "x", "lora": [{"nom": "turbo"}]})
    assert e.value.code == 503 and e.value.type == "memoire"
