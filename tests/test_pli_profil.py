"""Tests for the tools/pli-profil launcher, without a model or actual PLI launch (standard library only).

    python3 -m pytest -q tests/test_pli_profil.py        ou        python3 -m unittest tests/test_pli_profil.py

Each test gives the launcher a temporary personal folder, a machine.env pointing to that folder, a memory,
a disk space and a simulated chip. The launch test replaces the pli command with a fake script that writes its
arguments and environment.
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

DEPOT = Path(__file__).resolve().parents[1]
PROFILS = DEPOT / "templates" / "launch-profiles"
LANCEUR = DEPOT / "tools" / "pli-profil"
NOMS = ("assistant-vocal", "code", "creation", "creation-lora", "petite-machine", "tout-inclus")


class Banc(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.machine = self.home / "machine.env"
        v = {
            "PLI_DONNEES": self.home / "pli",
            "PLI_BIN": self.home / "pli/venv/bin/pli",
            "PLI_TETE_MTP": self.home / "pli/models/mtp-head-4bit.safetensors",
            "PLI_CARTE_A1": self.home / "pli/models/qwen38_27b_d2048.npz",
            "PLI_VENV_AUDIO": self.home / "venvs/mlx-audio",
            "PLI_VENV_IMAGE": self.home / "venvs/mflux",
            "PLI_MODELE_IMAGE": self.home / "modeles/qwen21",
            "PLI_H3": self.home / "h3.c/h3",
            "PLI_MODELE_VIDEO": self.home / "modeles/h3",
            "PLI_LORA_IMAGE": self.home / "lora/image.safetensors",
            "PLI_LORA_VIDEO": self.home / "lora/video.safetensors",
            "PLI_FUSION_PYTHON": self.home / "venvs/fusion/bin/python",
            "PLI_FUSION_SCRIPT": self.home / "outils/fusion_lora_h3.py",
        }
        self.v = {k: str(p) for k, p in v.items()}
        self.machine.write_text("".join(f"{k}={p}\n" for k, p in self.v.items()))
        self.env = {k: x for k, x in os.environ.items() if not k.startswith(("PLI_", "HF_"))}
        self.env.update(HOME=str(self.home), HF_HUB_CACHE=str(self.home / "hf"))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def creer(self, *cles: str) -> None:
        """Creates the files named by these variables (and for a venv, its bin/python)."""

        for cle in cles:
            p = Path(self.v[cle])
            if cle.startswith("PLI_VENV_"):
                p = p / "bin" / "python"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("")

    def tout_creer(self) -> None:
        self.creer("PLI_VENV_AUDIO", "PLI_VENV_IMAGE", "PLI_MODELE_IMAGE", "PLI_H3", "PLI_MODELE_VIDEO",
                   "PLI_LORA_IMAGE", "PLI_LORA_VIDEO", "PLI_FUSION_PYTHON", "PLI_FUSION_SCRIPT")

    def lancer(self, *args: str, ram: float = 64, disque: float = 1000, puce: str = "Apple M1 Ultra"):
        cmd = [sys.executable, str(LANCEUR), *args[:1], "--machine", str(self.machine),
               "--simuler-ram-go", str(ram), "--simuler-disque-go", str(disque), "--simuler-puce", puce, *args[1:]]
        return subprocess.run(cmd, capture_output=True, text=True, env=self.env, check=False)

    def plan(self, nom: str, *plus: str, **k) -> tuple[int, dict]:
        r = self.lancer(nom, "--plan-json", *plus, **k)
        self.assertIn(r.returncode, (0, 2), r.stderr)
        return r.returncode, json.loads(r.stdout)

    # ------------------------------------------------------------ list and --dry-run

    def test_liste(self) -> None:
        r = subprocess.run([sys.executable, str(LANCEUR), "--machine", str(self.machine)], capture_output=True,
                           text=True, env=self.env, check=False)
        self.assertEqual(r.returncode, 0, r.stderr)
        for nom in NOMS:
            self.assertIn(nom, r.stdout)

    def test_dry_run_de_chaque_profil(self) -> None:
        self.tout_creer()
        for nom in NOMS:
            with self.subTest(nom=nom):
                r = self.lancer(nom, "--dry-run")
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                self.assertIn("nothing is launched", r.stdout)
                self.assertIn(" serve ", r.stdout)
        self.assertFalse((self.home / "pli" / "profils").exists(), "--dry-run must not write anything")

    def test_profil_inconnu(self) -> None:
        r = self.lancer("inexistant", "--dry-run")
        self.assertEqual(r.returncode, 2)
        self.assertIn("unknown profile", r.stderr)

    # ------------------------------------------------------------ refus

    def test_refus_memoire(self) -> None:
        r = self.lancer("code", "--dry-run", ram=32)
        self.assertEqual(r.returncode, 2)
        self.assertIn("memory: this profile requires 48 GB", r.stderr)
        self.assertIn("petite-machine", r.stderr)

    def test_refus_memoire_tout_inclus_sur_48(self) -> None:
        self.tout_creer()
        r = self.lancer("tout-inclus", "--dry-run", ram=48)
        self.assertEqual(r.returncode, 2)
        self.assertIn("64 GB", r.stderr)

    def test_refus_disque(self) -> None:
        r = self.lancer("code", "--dry-run", disque=10)
        self.assertEqual(r.returncode, 2)
        self.assertIn("disk: ", r.stderr)

    def test_fichier_present_non_compte(self) -> None:
        _, avant = self.plan("code")
        self.creer("PLI_TETE_MTP")
        _, apres = self.plan("code")
        self.assertAlmostEqual(avant["besoin_disque_go"] - apres["besoin_disque_go"], 1.1, places=3)

    def test_refus_fichier_requis(self) -> None:
        r = self.lancer("creation", "--dry-run")              # no image model, no h3, no video model
        self.assertEqual(r.returncode, 2)
        self.assertIn("fichier requis absent", r.stderr)
        self.assertIn("introuvable", r.stderr)                # environnement mflux absent

    def test_forcer(self) -> None:
        code, plan = self.plan("code", "--forcer", ram=16)
        self.assertEqual(code, 0)
        self.assertTrue(plan["refus"])

    # ------------------------------------------------------------ settings chosen according to the machine

    def test_ultra_et_autre_puce(self) -> None:
        _, ultra = self.plan("code")
        self.assertIn("m1-ultra", ultra["commande"])
        self.assertIn("229376", ultra["commande"])
        self.assertEqual(ultra["environnement"]["MLX_MAX_MB_PER_BUFFER"], "500")
        _, autre = self.plan("code", puce="Apple M4 Max")
        self.assertNotIn("--profile", autre["commande"])
        self.assertNotIn("--context", autre["commande"])
        self.assertNotIn("MLX_MAX_MB_PER_BUFFER", autre["environnement"])

    def test_profil_machine_auto_et_choisi(self) -> None:
        _, ultra = self.plan("code", puce="Apple M2 Ultra", ram=128)
        self.assertEqual(ultra["profil_machine"], "m1-ultra")
        _, petit = self.plan("code", puce="Apple M1 Ultra", ram=48)      # Ultra under 64 GB: none (pli setup rule)
        self.assertEqual(petit["profil_machine"], "")
        dossier = self.home / "profils"
        dossier.mkdir()
        p = json.loads((PROFILS / "code.json").read_text(encoding="utf-8"))
        p["machine_profile"] = "m1-ultra"
        (dossier / "force.json").write_text(json.dumps(p), encoding="utf-8")
        p["machine_profile"] = "none"
        (dossier / "sans.json").write_text(json.dumps(p), encoding="utf-8")
        p["machine_profile"] = "inexistant"
        (dossier / "faux.json").write_text(json.dumps(p), encoding="utf-8")
        _, force = self.plan("force", "--dossier", str(dossier), puce="Apple M4 Max")
        self.assertIn("m1-ultra", force["commande"])
        self.assertIn("229376", force["commande"])                      # the condition follows the machine profile
        _, sans = self.plan("sans", "--dossier", str(dossier))
        self.assertNotIn("--profile", sans["commande"])
        r = self.lancer("faux", "--dry-run", "--dossier", str(dossier))
        self.assertEqual(r.returncode, 2)
        self.assertIn("unknown machine profile", r.stderr)

    def test_chaque_profil_designe_son_profil_machine(self) -> None:
        for chemin in PROFILS.glob("*.json"):
            self.assertIn("machine_profile", json.loads(chemin.read_text(encoding="utf-8")), chemin.name)

    def test_brouillon_hybride_si_tete_mtp(self) -> None:
        _, sans = self.plan("code")
        self.assertNotIn("PLI_HYBRID", sans["environnement"])
        self.creer("PLI_TETE_MTP")
        _, avec = self.plan("code")
        self.assertEqual(avec["environnement"]["PLI_HYBRID"], "1")
        self.assertEqual(avec["environnement"]["PLI_MTP_HEAD"], self.v["PLI_TETE_MTP"])

    def test_a1_seulement_sur_ultra(self) -> None:
        self.creer("PLI_CARTE_A1")
        _, ultra = self.plan("code")
        self.assertEqual(ultra["environnement"]["PLI_S3B_THRESHOLD"], "16384")
        _, autre = self.plan("code", puce="Apple M4 Max")
        self.assertNotIn("PLI_A1_LSA_MAP", autre["environnement"])

    def test_petite_machine(self) -> None:
        code16, _ = self.plan("petite-machine", ram=16)
        self.assertEqual(code16, 2)
        _, p24 = self.plan("petite-machine", ram=24, puce="Apple M2 Pro")
        c = p24["commande"]
        self.assertEqual(c[len(c) - 1 - c[::-1].index("--drafter") + 1], "none")
        _, p32 = self.plan("petite-machine", ram=32, puce="Apple M2 Max")
        c = p32["commande"]
        self.assertEqual(c[len(c) - 1 - c[::-1].index("--drafter") + 1], "auto")    # the last value wins

    def test_options_ajoutees_en_fin(self) -> None:
        r = self.lancer("code", "--plan-json", "--", "--port", "8081")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout)["commande"][-2:], ["--port", "8081"])

    def test_environnement_l_emporte(self) -> None:
        self.env["PLI_PORT"] = "9090"
        _, plan = self.plan("code")
        self.assertEqual(plan["commande"][-2:], ["--port", "9090"])

    # ------------------------------------------------------------ generation configurations

    def test_catalogues_et_config_valides(self) -> None:
        """The written files pass the PLI validators (fils_image, fils_video, gen.config)."""

        self.tout_creer()
        mods = {}
        for nom in ("fils_image", "fils_video"):
            spec = importlib.util.spec_from_file_location(f"_essai_{nom}", DEPOT / "src/pli/gen" / f"{nom}.py")
            mods[nom] = importlib.util.module_from_spec(spec)
            sys.modules[f"_essai_{nom}"] = mods[nom]          # dataclass reads the class module
            spec.loader.exec_module(mods[nom])
        sys.path.insert(0, str(DEPOT / "src"))
        try:
            from pli.gen.config import familles_depuis
        finally:
            sys.path.pop(0)
        for nom in ("assistant-vocal", "creation", "creation-lora", "tout-inclus"):
            with self.subTest(nom=nom):
                code, plan = self.plan(nom)
                self.assertEqual(code, 0, plan["refus"])
                gen = plan["environnement"]["PLI_GEN_CONFIG"]
                self.assertEqual(plan["environnement"]["PLI_GEN"], "1")
                familles = familles_depuis(plan["fichiers"][gen])
                for chemin, contenu in plan["fichiers"].items():
                    base = os.path.basename(chemin)
                    if base.startswith("image"):
                        mods["fils_image"].valider_catalogue(json.loads(json.dumps(contenu)))
                    elif base.startswith("video"):
                        mods["fils_video"].valider_catalogue(json.loads(json.dumps(contenu)))
                    self.assertNotIn("${", json.dumps(contenu), "unreplaced variable")
                self.assertTrue(set(familles) <= {"audio", "image", "video"})

    # ------------------------------------------------------------ lancement (faux pli)

    def test_lancement_avec_faux_pli(self) -> None:
        self.tout_creer()
        sortie = self.home / "argv.json"
        faux = Path(self.v["PLI_BIN"])
        faux.parent.mkdir(parents=True, exist_ok=True)
        faux.write_text("#!/bin/sh\n" + f'"{sys.executable}" -c \'import json,os,sys; json.dump({{"argv": sys.argv[1:], '
                        f'"env": dict(os.environ)}}, open("{sortie}", "w"))\' "$@"\n')
        faux.chmod(faux.stat().st_mode | stat.S_IEXEC)
        plus = () if platform.system() == "Darwin" else ("--forcer",)
        r = self.lancer("tout-inclus", *plus)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        vu = json.loads(sortie.read_text())
        self.assertEqual(vu["argv"][:2], ["serve", "mlx-community/Qwen3.8-27B-4bit"])
        config = Path(vu["env"]["PLI_GEN_CONFIG"])
        self.assertTrue(config.is_file())
        self.assertEqual(set(json.loads(config.read_text())["familles"]), {"audio", "image", "video"})
        self.assertTrue((self.home / "pli" / "sorties" / "videos").is_dir())


if __name__ == "__main__":
    unittest.main()
