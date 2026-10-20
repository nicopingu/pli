"""PLI (lot de patch 20261016-220013-4533, idée de TensorFold 1.0.12 « Recall ») : brouillons recopiés du texte déjà lu par le serveur.

Le serveur garde en mémoire les jetons des demandes finies (prompt et réponse). Quand la recopie ordinaire (le contexte de la
demande, SuffixLookupProposer) ne trouve rien, le rappel cherche les 16 derniers jetons de la demande dans ce registre et propose
la suite qui les y suivait. Le modèle vérifie chaque jeton recopié : le texte produit est le même, rappel allumé ou éteint (V2).

Version 2 (17/10/2026, après le banc pli-rappel27a) :
- une demande n'entre au registre qu'à sa fin (TensorFold : « its reply when the reply ends ») ; la v1 versait le contexte à
  chaque tour, ce qui écrasait dans l'index l'occurrence ancienne par celle de la demande en cours, si bien que la recopie
  annoncée au calcul de la profondeur du brouillon (un seul jeton de brouillon) disparaissait au tour suivant (« head 2 ») ;
- même réponse pour deux appels sur le même contexte (calcul de profondeur puis tour) ;
- règle de TensorFold : une recopie n'est proposée que si elle doit faire avancer plus que les brouillons récents du flux
  (moyenne glissante des jetons acceptés par tour de brouillon, contre celle des recopies du registre, par tranche de
  longueur de concordance 16-31, 32-63, 64+) ; une recopie qui n'avance rien fait taire le registre 4 tours pour ce flux ;
  des sondages de plus en plus rares tant qu'ils ratent.

Éteint par défaut. PLI_RAPPEL=1 l'allume ; PLI_RAPPEL_JETONS (défaut 1 000 000) borne le registre, les plus anciens partent d'abord.
Mémoire seulement (rien sur disque) : le registre repart vide à chaque démarrage du serveur.
"""

from __future__ import annotations

import os
import threading
import weakref
from array import array
from typing import Any, Sequence

import numpy as np

CLE = 16                         # jetons de la clé de recherche (TensorFold : 16 au moins)
PAS_DEDOUBLE = 64                # pas de la recherche du début neuf d'un prompt déjà lu
_P = np.uint64(0x100000001B3)    # multiplicateur du hachage polynomial (modulo 2**64)
ALPHA = 0.25                     # poids d'un tour dans les moyennes glissantes
DEPART_RAPPEL = 6.0              # jetons attendus d'une recopie du registre avant toute mesure (optimiste : il faut essayer)


def _hachages(ids: np.ndarray) -> np.ndarray:
    """Hachage de chaque fenêtre de CLE jetons de ``ids`` (len(ids) - CLE + 1 valeurs, fenêtre i = ids[i : i + CLE])."""

    n = len(ids) - CLE + 1
    h = np.zeros(n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(CLE):
            h = h * _P + ids[j:j + n].astype(np.uint64)
    return h


def _hachage(cle: Sequence[int]) -> int:
    return int(_hachages(np.asarray(cle, dtype=np.int64))[0])


def _tranche(longueur: int) -> int:
    return 0 if longueur < 32 else (1 if longueur < 64 else 2)


class Registre:
    """Jetons des demandes finies et l'index de leurs fenêtres de CLE jetons (dernière position de fin)."""

    def __init__(self, max_ids: int) -> None:
        self.max_ids = max(4 * CLE, int(max_ids))
        self.ids = array("i")
        self.debut = 0                  # position globale de ids[0]
        self.index: dict[int, int] = {}
        self.verrou = threading.Lock()
        self.attendu = [DEPART_RAPPEL] * 3   # jetons acceptés par recopie, par tranche de concordance
        self.offres = self.proposes = self.acceptes = 0

    @property
    def fin(self) -> int:
        return self.debut + len(self.ids)

    def jeton(self, pos: int) -> int:
        return self.ids[pos - self.debut]

    def _elaguer(self) -> None:
        if len(self.ids) <= self.max_ids:
            return
        coupe = len(self.ids) - self.max_ids // 2
        self.ids = self.ids[coupe:]
        self.debut += coupe
        self.index = {h: p for h, p in self.index.items() if p - CLE + 1 >= self.debut}

    def ajouter(self, seq: Sequence[int]) -> None:
        """Ajoute une demande finie (un segment à part : aucune fenêtre à cheval sur le segment d'avant)."""

        if len(seq) < CLE + 1:
            return
        with self.verrou:
            de = self.fin
            self.ids.extend(int(t) for t in seq)
            h = _hachages(np.asarray(seq, dtype=np.int64))
            self.index.update(zip(h.tolist(), range(de + CLE - 1, de + CLE - 1 + len(h))))
            self._elaguer()

    def debut_neuf(self, seq: Sequence[int]) -> int:
        """Indice de ``seq`` à partir duquel elle n'est plus déjà dans le registre (par pas de PAS_DEDOUBLE)."""

        n = len(seq)
        bornes = list(range(CLE - 1, n, PAS_DEDOUBLE))
        if not bornes:
            return 0
        tab = np.asarray(seq, dtype=np.int64)
        with self.verrou:
            for i, fin in enumerate(bornes):
                if _hachage(tab[fin - CLE + 1:fin + 1]) not in self.index:
                    return 0 if i == 0 else max(0, bornes[i - 1] + 1 - (CLE - 1))
        return max(0, bornes[-1] + 1 - (CLE - 1))

    def verser(self, seq: Sequence[int]) -> None:
        """Une demande finie : seule sa partie neuve entre au registre (un prompt déjà lu n'est pas reversé)."""

        de = self.debut_neuf(seq)
        self.ajouter(seq[de:])

    def chercher(self, contexte: Sequence[int], max_draft: int) -> tuple[list[int], int]:
        """(suite proposée, jetons concordants) pour la fin de ``contexte``."""

        if len(contexte) < CLE or max_draft <= 0:
            return [], 0
        queue = [int(t) for t in contexte[-CLE:]]
        h = _hachage(queue)
        with self.verrou:
            pos = self.index.get(h)
            if pos is None or pos < self.debut + CLE - 1 or pos + 1 >= self.fin:
                return [], 0
            if [self.jeton(p) for p in range(pos - CLE + 1, pos + 1)] != queue:
                return [], 0            # collision de hachage
            long = CLE
            limite = min(64, len(contexte), pos - self.debut + 1)
            while long < limite and self.jeton(pos - long) == int(contexte[-1 - long]):
                long += 1
            suite = [self.jeton(p) for p in range(pos + 1, min(self.fin, pos + 1 + max_draft))]
        return suite, long


_REGISTRE: Registre | None = None
_VERROU = threading.Lock()


def registre() -> Registre | None:
    global _REGISTRE
    if os.environ.get("PLI_RAPPEL", "") not in ("1", "oui", "on"):
        return None
    with _VERROU:
        if _REGISTRE is None:
            _REGISTRE = Registre(int(os.environ.get("PLI_RAPPEL_JETONS", "1000000") or 1000000))
            print(f"[tensorfold] PLI rappel allumé (v2) : registre en mémoire, {_REGISTRE.max_ids} jetons au plus", flush=True)
        return _REGISTRE


def _verser_a_la_fin(reg: Registre, porte: dict) -> None:
    ctx, n = porte.get("ctx"), int(porte.get("n", 0))
    if ctx is not None and n > CLE:
        try:
            reg.verser(ctx[:n])
        except Exception:  # noqa: BLE001 - la fin d'une demande ne doit jamais casser le serveur
            pass


class ProposeurRappel:
    """Enveloppe d'un proposeur de recopie : la recopie du contexte d'abord, le registre quand elle ne trouve rien."""

    TAIRE = 4                           # tours sans le registre après une recopie qui n'a rien fait avancer

    def __init__(self, base: Any, reg: Registre) -> None:
        self.base = base
        self.reg = reg
        self.porte = {"ctx": None, "n": 0}   # le contexte de la demande (liste grandissante), versé au registre à la fin
        weakref.finalize(self, _verser_a_la_fin, reg, self.porte)
        self.source = ""                # "base" ou "rappel" : qui a fait la dernière proposition
        self.last_match = 0
        self.last_confident = False
        self.tete = 0.0                 # jetons acceptés par tour de brouillon du modèle (moyenne glissante)
        self.tranche = 0
        self._taire = 0
        self._refus = 0                 # appels où la règle a écarté une recopie possible
        self._sonde_tous = 8            # un sondage tous les N refus, doublé à chaque sondage raté
        self._sondage = False
        self._dernier: tuple[int, int, list[int], str, int, bool] | None = None

    def __getattr__(self, nom: str) -> Any:     # télémétrie et attributs propres au proposeur de base
        base = self.__dict__.get("base")
        if base is None:
            raise AttributeError(nom)
        return getattr(base, nom)

    def _garder(self, contexte: Sequence[int]) -> None:
        ancien, n = self.porte["ctx"], self.porte["n"]
        if ancien is not None and ancien is not contexte:
            suite = (len(contexte) >= n and n > 0 and int(contexte[0]) == int(ancien[0])
                     and int(contexte[n - 1]) == int(ancien[n - 1]))
            if not suite:                                   # contexte remplacé : l'ancien est une demande finie
                _verser_a_la_fin(self.reg, dict(self.porte))
        self.porte["ctx"] = contexte
        self.porte["n"] = len(contexte)

    def propose(self, context: Sequence[int], max_draft: int) -> list[int]:
        self._garder(context)
        cle = (len(context), int(max_draft))
        d = self._dernier
        if d is not None and (d[0], d[1]) == cle:   # même contexte (profondeur du brouillon, puis tour) : même réponse
            self.source, self.last_match, self.last_confident = d[3], d[4], d[5]
            return list(d[2])
        prop = self.base.propose(context, max_draft)
        if prop:
            res = (list(prop), "base", int(getattr(self.base, "last_match", 0) or 0),
                   bool(getattr(self.base, "last_confident", False)))
        else:
            res = ([], "", int(getattr(self.base, "last_match", 0) or 0), False)
            if self._taire > 0:
                self._taire -= 1
            else:
                suite, long = self.reg.chercher(context, max_draft)
                if len(suite) >= 2:
                    t = _tranche(long)
                    attendu = min(float(len(suite)), self.reg.attendu[t])
                    self._sondage = False
                    if attendu <= self.tete:
                        self._refus += 1
                        if self._refus % self._sonde_tous:
                            suite = []
                        else:
                            self._sondage = True
                    if suite:
                        self.tranche = t
                        res = (suite, "rappel", long, long >= 24)
                        self.reg.offres += 1
                        self.reg.proposes += len(suite)
        self.source, self.last_match, self.last_confident = res[1], res[2], res[3]
        self._dernier = (cle[0], cle[1], list(res[0]), res[1], res[2], res[3])
        return list(res[0])

    def observe(self, proposed: int, accepted: int) -> None:
        if self.source == "rappel":
            reg = self.reg
            reg.acceptes += int(accepted)
            reg.attendu[self.tranche] += ALPHA * (float(accepted) - reg.attendu[self.tranche])
            if accepted == 0:
                self._taire = self.TAIRE
            if self._sondage:
                self._sonde_tous = 8 if accepted > self.tete else min(256, 2 * self._sonde_tous)
                self._sondage = False
            if reg.offres % 50 == 0:
                print(f"[tensorfold] PLI rappel : {reg.offres} offres, {reg.proposes} jetons proposés, {reg.acceptes} "
                      f"acceptés, attendu {', '.join(f'{a:.1f}' for a in reg.attendu)}, registre {len(reg.ids)} jetons",
                      flush=True)
            return
        obs = getattr(self.base, "observe", None)
        if obs is not None:
            obs(proposed, accepted)

    def observe_tete(self, accepted: int) -> None:
        """Un tour de brouillon du modèle (lane_family) : jetons acceptés."""

        self.tete += ALPHA * (float(accepted) - self.tete)


def envelopper(base: Any) -> Any:
    """Le proposeur de recopie, enveloppé du rappel si PLI_RAPPEL=1."""

    reg = registre()
    return base if reg is None or base is None else ProposeurRappel(base, reg)
