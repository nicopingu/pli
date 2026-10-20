"""PLI (lot de patch 20261016-220013-4533, repris de Rapid-MLX 0.16.1 #4512) : garde Host et Origin.

Un serveur PLI écoute en 127.0.0.1 sans clé d'API. Deux attaques venues d'un navigateur l'atteignent :
- la reliure DNS (DNS rebinding) : une page fait pointer son propre nom vers 127.0.0.1 et lit les réponses ;
  la requête arrive alors avec un Host qui n'est pas une adresse locale ;
- la requête de passage (drive-by) : une page envoie un POST « simple » vers http://127.0.0.1:port ; elle ne lit pas
  la réponse, mais le calcul part ; le navigateur joint un en-tête Origin.

La garde refuse (403) un Host absent de la liste admise quand le serveur écoute sur une adresse locale, et toute requête
portant un Origin dont l'hôte n'est pas admis, quelle que soit l'adresse d'écoute. Les clients d'API (Claude Code, curl,
Python, un relais local) n'envoient pas d'Origin ; le relais réécrit Host en 127.0.0.1:port.

Éteinte par défaut, comme tout lot PLI (pli.env absent : comportement de TensorFold 0.6.6). PLI_GARDE_HOTE=1 l'allume ;
PLI_HOTES_ADMIS="nom1,nom2" ajoute des noms admis (Host et Origin), en plus des noms locaux et de l'adresse d'écoute.
"""

from __future__ import annotations

import ipaddress
import os
from urllib.parse import urlsplit

_LOCAUX = ("localhost", "127.0.0.1", "::1")


def _hote(valeur: str) -> str:
    """Le nom d'hôte d'un en-tête Host (« nom:port », « [::1]:port »), en minuscules, sans port ni crochets."""

    valeur = valeur.strip().lower()
    if valeur.startswith("["):
        return valeur[1:].split("]", 1)[0]
    if valeur.count(":") == 1:
        return valeur.split(":", 1)[0]
    return valeur  # sans port, ou IPv6 nue


def _local(nom: str) -> bool:
    if nom in _LOCAUX or nom.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(nom).is_loopback
    except ValueError:
        return False


class HostGuard:
    def __init__(self, ecoute: str, admis: tuple[str, ...] = ()) -> None:
        self.ecoute = _hote(str(ecoute or ""))
        self.admis = frozenset(n for n in (_hote(a) for a in admis) if n)
        # Host n'est vérifié que sur une écoute locale : sur 0.0.0.0 ou une adresse du réseau, les clients appellent
        # le serveur par des noms qu'on ne connaît pas d'avance, et la reliure DNS ne vise que les adresses locales.
        self.verifie_host = _local(self.ecoute)

    def _admis(self, nom: str) -> bool:
        return bool(nom) and (_local(nom) or nom == self.ecoute or nom in self.admis)

    def refuse(self, headers) -> str | None:
        """Le motif du refus, ou None si la requête passe."""

        if self.verifie_host:
            host = headers.get("Host")
            if host is not None and not self._admis(_hote(host)):
                return f"Host {host!r} non admis"
        origin = headers.get("Origin")
        if origin is not None:
            nom = (urlsplit(origin.strip()).hostname or "") if origin.strip().lower() != "null" else ""
            if not self._admis(nom):
                return f"Origin {origin!r} non admis"
        return None


def configure(args, environ=os.environ) -> HostGuard | None:
    if environ.get("PLI_GARDE_HOTE", "") not in ("1", "oui", "on"):
        return None
    admis = tuple(n.strip() for n in environ.get("PLI_HOTES_ADMIS", "").split(",") if n.strip())
    garde = HostGuard(getattr(args, "host", "127.0.0.1"), admis)
    print(f"[tensorfold] PLI garde Host/Origin allumée (Host vérifié : {'oui' if garde.verifie_host else 'non'}"
          f"{', admis en plus : ' + ', '.join(sorted(garde.admis)) if garde.admis else ''})", flush=True)
    return garde
