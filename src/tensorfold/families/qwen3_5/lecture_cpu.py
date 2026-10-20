"""PLI (lecture répartie, lot S3) : pendant la lecture d'un morceau, le CPU calcule le MLP des dernières lignes.

Découpe par lignes : le GPU garde les premières lignes du morceau, le CPU fait gate, up, SiLU et down des dernières avec le
noyau ``qmm_cpu`` (poids 4 bits lus dans le tampon du GPU, sans copie). Aucune somme n'est recollée entre organes : les lignes
sont mises bout à bout. Les lignes du CPU sont calculées en 32 bits : leurs valeurs diffèrent un peu de celles du GPU, donc le
texte peut changer (classe V3 tant qu'un chemin exact ne le remplace pas).

Éteint par défaut. ``PLI_LECTURE_CPU`` : part des lignes donnée au CPU (0 à 0,9). ``PLI_LECTURE_CPU_MIN`` : nombre de lignes
du morceau en dessous duquel rien n'est partagé (256). ``PLI_LECTURE_CPU_PRIORITE`` : 0 haute, 1 moyenne (défaut), 2 de fond.
``PLI_LECTURE_CPU_MODE`` : « decale » (défaut) ou « couche ».

Mode « couche » : dans chaque couche, le CPU fait ses lignes pendant que le GPU fait le MLP des siennes, puis on attend.
Mode « decale » : le morceau est traité comme deux suites de lignes (celles du GPU, puis celles du CPU). Le CPU fait le MLP de
ses lignes pour la couche k pendant que le GPU fait l'attention et le MLP de ses propres lignes pour la couche k + 1 (le modèle
est causal : les lignes du GPU ne dépendent pas de celles du CPU). Le GPU n'attend le CPU qu'au moment de faire l'attention
des lignes du CPU.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any

MESURES = {"couches": 0, "attention_ms": 0.0, "gpu_ms": 0.0, "cpu_ms": 0.0, "attente_ms": 0.0, "relais_ms": 0.0,
           "lignes_gpu": 0, "lignes_cpu": 0}


def part() -> float:
    try:
        p = float(os.environ.get("PLI_LECTURE_CPU", "0") or 0)
    except ValueError:
        return 0.0
    return p if 0.0 < p <= 0.9 else 0.0


def _entier(nom: str, defaut: int) -> int:
    try:
        return int(os.environ.get(nom, str(defaut)) or defaut)
    except ValueError:
        return defaut


def trace(lignes: int, lc: int, t0: float) -> None:
    """PLI_LECTURE_CPU_TRACE=1 : une ligne par morceau au journal (temps cumulés depuis le chargement, en ms)."""

    if os.environ.get("PLI_LECTURE_CPU_TRACE", "0") != "1":
        return
    import sys

    m = MESURES
    print("[pli] lecture répartie : morceau de %d lignes dont %d au CPU, %.0f ms ; cumul couches %d, GPU %.0f, attention des "
          "lignes du CPU %.0f, CPU %.0f, attente du CPU %.0f, relais %.0f" % (lignes, lc, (time.perf_counter() - t0) * 1000,
          m["couches"], m["gpu_ms"], m["attention_ms"], m["cpu_ms"], m["attente_ms"], m["relais_ms"]), file=sys.stderr, flush=True)


def mode() -> str:
    return "couche" if os.environ.get("PLI_LECTURE_CPU_MODE", "decale") == "couche" else "decale"


class LectureCPU:
    """Les trois matrices du MLP de chaque couche, vues par le noyau CPU ; une instance par modèle."""

    def __init__(self) -> None:
        self._poids: dict[int, Any] = {}
        self._noyau: Any = None
        self.minimum = max(16, _entier("PLI_LECTURE_CPU_MIN", 256))

    def _charger(self) -> Any:
        if self._noyau is None:
            from tensorfold.kernels.pli_cpu import qmm_cpu
            qmm_cpu.charger()
            qmm_cpu.priorite(_entier("PLI_LECTURE_CPU_PRIORITE", 1))
            self._noyau = qmm_cpu
        return self._noyau

    def poids(self, layer: Any) -> Any:
        """(gate, up, down) lisibles sans copie, ou None si la couche n'a pas un MLP dense en 4 bits par groupes de 64."""

        cle = id(layer)
        if cle in self._poids:
            return self._poids[cle]
        import mlx.core as mx

        trio: Any = None
        mlp = getattr(layer, "mlp", None)
        noms = ("gate_proj", "up_proj", "down_proj")
        if mlp is not None and all(hasattr(mlp, n) for n in noms):
            try:
                q = self._charger()
                faits = []
                for n in noms:
                    m = getattr(mlp, n)
                    if (getattr(m, "bits", 0) != 4 or getattr(m, "group_size", 0) != 64 or getattr(m, "biases", None) is None
                            or "bias" in m or getattr(m, "mode", "affine") != "affine"):
                        raise ValueError("format non pris")
                    sb = {mx.bfloat16: 0, mx.float16: 1, mx.float32: 2}[m.scales.dtype]
                    if m.biases.dtype != m.scales.dtype:
                        raise ValueError("types des échelles")
                    mx.eval(m.weight, m.scales, m.biases)
                    faits.append(q.Poids(m.weight, m.scales, m.biases, sb=sb))
                trio = tuple(faits)
            except Exception:
                trio = None
        self._poids[cle] = trio
        return trio

    def couche(self, layer: Any, x: Any, mask: Any, cache: Any, p: float) -> Any:
        """Une couche du passage de MLX, le MLP des dernières lignes fait par le CPU pendant que le GPU fait les autres."""

        import mlx.core as mx
        import numpy as np

        from tensorfold.engine.family_common import cache_arrays

        trio = self.poids(layer)
        lignes = int(x.shape[1])
        lc = (int(lignes * p) // 8) * 8
        if trio is None or x.shape[0] != 1 or lignes < self.minimum or lc < 8 or lc >= lignes:
            return layer(x, mask=mask, cache=cache)
        q = self._noyau
        t0 = time.perf_counter()
        if layer.is_linear:
            r = layer.linear_attn(layer.input_layernorm(x), mask, cache)
        else:
            r = layer.self_attn(layer.input_layernorm(x), mask, cache)
        h = x + r
        xn = layer.post_attention_layernorm(h)
        mx.eval(xn, h, *cache_arrays([cache]))
        t1 = time.perf_counter()
        xc = np.array(xn[0, lignes - lc:].astype(mx.float32))
        boite: dict[str, Any] = {}

        def travail() -> None:
            a = time.perf_counter()
            try:
                g = q.produit("blas", trio[0], xc)
                u = q.produit("blas", trio[1], xc)
                q.swiglu(g, u)
                boite["y"] = q.produit("blas", trio[2], g)
            except Exception as err:                       # le fil ne doit jamais emporter la lecture
                boite["erreur"] = err
            boite["ms"] = (time.perf_counter() - a) * 1000

        fil = threading.Thread(target=travail, name="pli-lecture-cpu")
        t2 = time.perf_counter()
        fil.start()
        yg = layer.mlp(xn[:, :lignes - lc])
        mx.eval(yg)
        t3 = time.perf_counter()
        fil.join()
        t4 = time.perf_counter()
        if "y" not in boite:                               # repli : le GPU fait aussi les lignes du CPU
            yc = layer.mlp(xn[:, lignes - lc:])
        else:
            yc = mx.array(boite["y"]).astype(yg.dtype)[None]
        out = h + mx.concatenate([yg, yc], axis=1)
        M = MESURES
        M["couches"] += 1; M["attention_ms"] += (t1 - t0) * 1000; M["relais_ms"] += (t2 - t1) * 1000
        M["gpu_ms"] += (t3 - t2) * 1000; M["cpu_ms"] += boite.get("ms", 0.0); M["attente_ms"] += (t4 - t3) * 1000
        M["lignes_gpu"] += lignes - lc; M["lignes_cpu"] += lc
        return out

    def lignes_cpu(self, couches: Any, h: Any, p: float) -> int:
        """Nombre de lignes du morceau données au CPU en mode décalé ; 0 = ne pas partager ce morceau."""

        lignes = int(h.shape[1])
        lc = (int(lignes * p) // 8) * 8
        if h.shape[0] != 1 or lignes < self.minimum or lc < 8 or lc >= lignes:
            return 0
        return lc if all(self.poids(layer) is not None for layer in couches) else 0

    def passage_decale(self, couches: Any, caches: Any, h: Any, masque_fa: Any, masque_ssm: Any, lc: int, pas: int,
                       releves: Any = None) -> Any:
        """Générateur : toutes les couches en mode décalé ; rend la main toutes les ``pas`` couches (0 = jamais).

        Valeur de fin : les états de sortie du morceau entier, avant la norme finale."""

        import mlx.core as mx
        import numpy as np

        from tensorfold.engine.family_common import cache_arrays

        q = self._noyau
        debut = time.perf_counter()
        g = int(h.shape[1]) - lc
        hg, hc = h[:, :g], h[:, g:]
        attente: Any = None                                 # (fil, boîte, h après attention, entrée du MLP, couche)
        M = MESURES

        def finir(a: Any) -> Any:
            fil, boite, h1, xn, couche = a
            t = time.perf_counter()
            fil.join()
            M["attente_ms"] += (time.perf_counter() - t) * 1000
            M["cpu_ms"] += boite.get("ms", 0.0)
            y = boite.get("y")
            yc = couche.mlp(xn) if y is None else mx.array(y).astype(h1.dtype)[None]     # repli : le GPU refait ces lignes
            return h1 + yc

        def attention(couche: Any, x: Any, c: Any) -> Any:
            if couche.is_linear:
                return x + couche.linear_attn(couche.input_layernorm(x), masque_ssm(x, c), c)
            return x + couche.self_attn(couche.input_layernorm(x), masque_fa(x, c), c)

        total = len(couches)
        for k, (couche, c) in enumerate(zip(couches, caches)):
            trio = self.poids(couche)
            t0 = time.perf_counter()
            h1g = attention(couche, hg, c)
            hg = h1g + couche.mlp(couche.post_attention_layernorm(h1g))
            mx.eval(hg, *cache_arrays([c]))
            t1 = time.perf_counter()
            if attente is not None:
                hc = finir(attente)
            t2 = time.perf_counter()
            h1c = attention(couche, hc, c)
            xn = couche.post_attention_layernorm(h1c)
            mx.eval(xn, h1c, *cache_arrays([c]))
            xc = np.array(xn[0].astype(mx.float32))
            t3 = time.perf_counter()
            boite: dict[str, Any] = {}

            def travail(trio: Any = trio, xc: Any = xc, boite: dict[str, Any] = boite) -> None:
                a = time.perf_counter()
                try:
                    gg = q.produit("blas", trio[0], xc)
                    uu = q.produit("blas", trio[1], xc)
                    q.swiglu(gg, uu)
                    boite["y"] = q.produit("blas", trio[2], gg)
                except Exception as err:
                    boite["erreur"] = err
                boite["ms"] = (time.perf_counter() - a) * 1000

            fil = threading.Thread(target=travail, name="pli-lecture-cpu")
            fil.start()
            attente = (fil, boite, h1c, xn, couche)
            M["couches"] += 1; M["gpu_ms"] += (t1 - t0) * 1000; M["attention_ms"] += (t3 - t2) * 1000
            M["lignes_gpu"] += g; M["lignes_cpu"] += lc
            fait = k + 1
            if pas > 0 and fait % pas == 0 and fait < total:
                garde = list(releves) if releves else None      # relevés du brouillon : voir prefill_steps
                yield
                if garde is not None:
                    releves[:] = garde
        hc = finir(attente)
        sortie = mx.concatenate([hg, hc], axis=1)
        mx.eval(sortie)
        trace(g + lc, lc, debut)
        return sortie
