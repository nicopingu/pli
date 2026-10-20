"""PLI : journal d'événements, une ligne JSON par action, avec la conversation (PLI 3.6.0).

Allumé par ``PLI_EVENEMENTS=<fichier>`` (pli.env) ; éteint, chaque appel rend la main aussitôt. Lu par le
tableau de bord de supervision (vue Inférence, actions de PLI par session). Ne touche ni au calcul ni à l'ordre
des demandes : le texte produit est le même. Une erreur d'écriture est avalée, jamais la demande.

Rien du contenu des demandes n'est écrit : une conversation n'est qu'une empreinte (10 caractères) des
messages jusqu'au premier message de l'utilisateur compris, que les tours suivants gardent. Les actions qui ne
connaissent que des jetons (point de reprise, déversement, relecture du disque) sont rattachées à la
conversation récente dont l'invite commence par ces jetons ; plusieurs : « partagé » (bloc système commun).

Champs : t (heure Unix), m (modèle), a (action), req (demande), conv (conversation), puis ceux de l'action.
Le fichier passe à ``<fichier>.1`` au-delà de PLI_EVENEMENTS_MAX_MO (50 Mo par défaut).
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
import sys
import threading
import time
from typing import Any

_CHEMIN = os.environ.get("PLI_EVENEMENTS", "").strip()
try:
    _MAX_OCTETS = max(1, int(os.environ.get("PLI_EVENEMENTS_MAX_MO", "50"))) * 1024 * 1024
except ValueError:
    _MAX_OCTETS = 50 * 1024 * 1024
_VERROU = threading.Lock()
_RECENTES: "OrderedDict[str, tuple[str, list[int]]]" = OrderedDict()   # demande -> (conversation, invite)
_MAX_RECENTES = 64
_MODELE: str | None = None


def actif() -> bool:
    return bool(_CHEMIN)


def _modele() -> str:
    """Nom servi (--name), sinon dossier du modèle, lus une fois sur la ligne de commande de « tensorfold serve »."""

    global _MODELE
    if _MODELE is None:
        args, nom = sys.argv, ""
        try:
            if "--name" in args:
                nom = args[args.index("--name") + 1]
            elif "serve" in args:
                nom = os.path.basename(str(args[args.index("serve") + 1]).rstrip("/"))
        except (IndexError, ValueError):
            nom = ""
        _MODELE = nom
    return _MODELE


def _texte(contenu: Any) -> str:
    if isinstance(contenu, str):
        return contenu
    if isinstance(contenu, list):
        return "".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in contenu)
    return "" if contenu is None else str(contenu)


def conversation(messages: Any) -> str:
    """Empreinte des messages jusqu'au premier message de l'utilisateur compris ("" si illisible)."""

    try:
        h = hashlib.sha1()
        for m in messages or []:
            role = str(m.get("role", "")) if isinstance(m, dict) else ""
            h.update(role.encode("utf-8"))
            h.update(b"\0")
            h.update(_texte(m.get("content") if isinstance(m, dict) else m).encode("utf-8", "replace"))
            h.update(b"\1")
            if role == "user":
                break
        return h.hexdigest()[:10]
    except Exception:  # noqa: BLE001 - une empreinte manquée ne coûte que le regroupement
        return ""


def lier(job: Any, conv: str) -> None:
    """Retient la conversation et l'invite d'une demande (les 64 dernières), pour rattacher les actions par jetons."""

    if not _CHEMIN:
        return
    try:
        job.pli_conv = conv
        with _VERROU:
            _RECENTES[job.job_id] = (conv, list(job.prompt_ids))
            _RECENTES.move_to_end(job.job_id)
            while len(_RECENTES) > _MAX_RECENTES:
                _RECENTES.popitem(last=False)
    except Exception:  # noqa: BLE001
        pass


def conv_des_jetons(jetons: Any) -> str:
    """Conversation dont l'invite commence par ``jetons`` ou en est le début ; « partagé » si plusieurs."""

    if not _CHEMIN:
        return ""
    try:
        t = list(jetons)
        n = len(t)
        if not n:
            return ""
        with _VERROU:
            recentes = list(_RECENTES.values())
        trouvees: list[str] = []
        for conv, ids in reversed(recentes):
            m = min(n, len(ids))
            if conv and m and ids[:m] == t[:m] and m >= n // 2 and conv not in trouvees:
                trouvees.append(conv)
        if not trouvees:
            return ""
        return trouvees[0] if len(trouvees) == 1 else "partagé"
    except Exception:  # noqa: BLE001
        return ""


def ev(action: str, job: Any = None, conv: str | None = None, **champs: Any) -> None:
    """Écrit une action (une ligne). Les champs à None sont omis ; aucune exception ne sort d'ici."""

    if not _CHEMIN:
        return
    try:
        d: dict[str, Any] = {"t": round(time.time(), 3), "m": _modele(), "a": action}
        if job is not None:
            d["req"] = str(getattr(job, "job_id", ""))
            if conv is None:
                conv = getattr(job, "pli_conv", "") or ""
        if conv:
            d["conv"] = conv
        for cle, valeur in champs.items():
            if valeur is None:
                continue
            d[cle] = round(valeur, 3) if isinstance(valeur, float) else valeur
        ligne = json.dumps(d, ensure_ascii=False, separators=(",", ":")) + "\n"
        with _VERROU:
            try:
                if os.path.getsize(_CHEMIN) > _MAX_OCTETS:
                    os.replace(_CHEMIN, _CHEMIN + ".1")
            except OSError:
                pass
            with open(_CHEMIN, "a", encoding="utf-8") as f:
                f.write(ligne)
    except Exception:  # noqa: BLE001 - le journal ne fait jamais échouer une demande
        pass
