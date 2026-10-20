"""Profils machine de PLI : réglages PLI_* par puce et par famille de modèle.

Un profil est un dossier de ``tensorfold/pli/profils/`` : ``commun.env`` (toutes familles) et ``<model_type>.env``
(une famille, par exemple ``qwen3_5`` pour le Qwen3.8 dense). Lignes ``CLE=valeur`` ; ``#`` commente.
Une variable déjà présente dans l'environnement l'emporte toujours sur le profil.
Le profil est appliqué avant tout import de noyau : plusieurs modules lisent leurs réglages à l'import.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DOSSIER = Path(__file__).resolve().parent / "profils"


def disponibles() -> list[str]:
    return sorted(p.name for p in DOSSIER.iterdir() if p.is_dir()) if DOSSIER.is_dir() else []


def _lire(fichier: Path) -> dict[str, str]:
    valeurs = {}
    for ligne in fichier.read_text().splitlines():
        ligne = ligne.strip()
        if ligne and not ligne.startswith("#") and "=" in ligne:
            cle, valeur = ligne.split("=", 1)
            valeurs[cle.strip()] = valeur.strip().strip('"')
    return valeurs


def type_de_modele(dossier: Path | None) -> str | None:
    """Le ``model_type`` du config.json d'un dossier de modèle, sans importer de famille."""

    try:
        return json.loads((Path(dossier) / "config.json").read_text()).get("model_type") if dossier else None
    except (OSError, ValueError):
        return None


def appliquer(nom: str, model_type: str | None = None, environ=os.environ) -> list[str]:
    """Pose les réglages du profil ``nom`` (et de la famille ``model_type``) ; rend les fichiers lus."""

    dossier = DOSSIER / nom
    if not dossier.is_dir():
        raise ValueError(f"profil inconnu : {nom} (profils disponibles : {', '.join(disponibles()) or 'aucun'})")
    lus = []
    for fichier in [dossier / "commun.env"] + ([dossier / f"{model_type}.env"] if model_type else []):
        if fichier.is_file():
            for cle, valeur in _lire(fichier).items():
                environ.setdefault(cle, valeur)
            lus.append(f"{nom}/{fichier.name}")
    return lus
