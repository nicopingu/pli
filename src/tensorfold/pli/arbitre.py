"""PLI : arbitre de coût des tours (PLI_ARBITRE=1). Ne change que la largeur des fenêtres, jamais les jetons.

TensorFold choisit la largeur d'une fenêtre avec des coûts mesurés au chargement, contexte vide. Or chaque ligne d'une
fenêtre relit le contexte dans les couches à attention complète : à long contexte, une ligne coûte bien plus que prévu,
et les copies larges (16 à 128 lignes) font perdre du temps. L'arbitre apprend en ligne le coût réel d'un tour :

    ms = base(lignes) + A + K x lignes x contexte / 100 000

base : coûts du chargement ; A : part fixe du tour (brouillon, tirage) ; K : coût d'une ligne par 100 000 jetons de contexte.
A et K sont ajustés par moindres carrés avec oubli sur les tours observés. Il sert à deux choix :
- la largeur d'une copie : celle qui donne le plus de jetons attendus par milliseconde, avec le taux de réussite des
  copies de la demande en cours ; pas de copie si un tour d'une ligne fait mieux ;
- le nombre de brouillons d'un tour (têtes qui donnent leurs probabilités, comme DFlash2) : mêmes coûts.
"""

from __future__ import annotations

import os

ACTIF = os.environ.get("PLI_ARBITRE", "0") == "1"
JOURNAL = os.environ.get("PLI_ARBITRE_JOURNAL", "")


class Arbitre:
    oubli = 0.985            # poids d'un tour passé à chaque nouveau tour (environ 65 tours de mémoire)
    copie_oubli = 0.9        # idem pour le taux de réussite des copies d'une demande
    copie_depart = (15.0, 1.0)   # réussites et échecs supposés au départ : 0,94, le taux de TensorFold
    marge = 1.05             # une copie ou un brouillon doit faire au moins 5 % de mieux qu'un tour d'une ligne

    def __init__(self, base: dict[int, float]) -> None:
        self.base = {int(w): float(ms) for w, ms in base.items()}
        self.a, self.k = 8.0, 0.0
        self.n = self.sx = self.sy = self.sxx = self.sxy = 0.0
        self.copies: dict[str, list[float]] = {}
        self.tours = 0

    def observe(self, rows: int, contexte: int, ms: float) -> None:
        """Un tour d'une demande seule : ``rows`` lignes à ``contexte`` jetons, ``ms`` au total."""

        b = self.base.get(int(rows))
        if b is None or ms <= 0:
            return
        x, y, w = rows * contexte / 1e5, ms - b, self.oubli
        self.n, self.sx, self.sy = self.n * w + 1.0, self.sx * w + x, self.sy * w + y
        self.sxx, self.sxy = self.sxx * w + x * x, self.sxy * w + x * y
        self.tours += 1
        det = self.n * self.sxx - self.sx * self.sx
        if self.tours >= 12 and det > 1e-3 * self.n * self.sxx:        # assez de tours, et des largeurs assez variées
            self.k = max(0.0, (self.n * self.sxy - self.sx * self.sy) / det)
        self.a = max(0.0, (self.sy - self.k * self.sx) / self.n)
        if JOURNAL and self.tours % 50 == 0:
            with open(JOURNAL, "a") as h:
                h.write(f"{self.tours} A={self.a:.1f} K={self.k:.2f}\n")

    def cout(self, rows: int, contexte: int) -> float | None:
        b = self.base.get(int(rows))
        return None if b is None else b + self.a + self.k * rows * contexte / 1e5

    def table(self, contexte: float, most: int) -> dict[int, float]:
        """Coûts par nombre de lignes, sans la part fixe (elle se passe à part, comme le surcoût de TensorFold)."""

        return {w: b + self.k * w * contexte / 1e5 for w, b in self.base.items() if w <= most}

    def copie_vue(self, stream_id: str, proposes: int, acceptes: int) -> None:
        s = self.copies.setdefault(stream_id, list(self.copie_depart))
        s[0] = s[0] * self.copie_oubli + acceptes
        s[1] = s[1] * self.copie_oubli + (1.0 if acceptes < proposes else 0.0)

    def largeur_copie(self, stream_id: str, contexte: int, plafond: int) -> int:
        """Jetons qu'une copie peut proposer (0 : pas de copie), au plus ``plafond``."""

        s = self.copies.get(stream_id, self.copie_depart)
        r = s[0] / (s[0] + s[1])
        seul = self.cout(1, contexte)
        if seul is None:
            return plafond
        meilleur, largeur = self.marge / seul, 0
        attendu = reste = 1.0
        for w in range(1, int(plafond) + 1):
            reste *= r
            attendu += reste
            c = self.cout(w + 1, contexte)
            if c is not None and attendu / c > meilleur:
                meilleur, largeur = attendu / c, w
        return largeur

    def oublie(self, stream_id: str) -> None:
        self.copies.pop(stream_id, None)


__all__ = ["ACTIF", "Arbitre"]
