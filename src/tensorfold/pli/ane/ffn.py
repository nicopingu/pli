"""PLI SP1 : FFN dense du 27B partagé par canaux entre le GPU et le Neural Engine pendant la lecture du prompt (V3).

Repris de Splash 1.3.2-m1 (runtime/ops/AneFfn.cpp, Apache-2.0, voir natif/LICENSE-splash). Le GPU calcule les canaux
intermédiaires de tête (chemin PLI_LECTURE_DEPLIE_MLP=t de deplie.py, restreint à ces canaux), le Neural Engine ceux de
queue en W8A8 (programme de client.py). Les deux parts de down sont additionnées en fp32. Le texte change (V3).

Voie a : une synchronisation par le CPU à chaque couche. Ordre d'une couche i (jeu de surfaces s = i % 2) :
  1. GPU : entrées tournées et quantifiées en int8 (et poids de la couche i si le jeu s ne les a pas), évaluées ;
  2. CPU : copie dans les surfaces du jeu s, évaluation du Neural Engine lancée sur un fil (ctypes rend le verrou de Python) ;
  3. GPU : poids de la couche suivante requantifiés (d'abord) puis part GPU de la couche i ; le CPU copie les poids
     suivants dans l'autre jeu pendant que le GPU et le Neural Engine calculent ;
  4. attente du Neural Engine, copie de sa sortie, jonction sur le GPU.

  PLI_ANE_FFN = 0 (défaut, éteint) | 1
  PLI_ANE_FFN_UNITES = 12 : unités de 512 canaux au Neural Engine (sonde pli-sp1-sonde2 : 12 -> T = 0,66 L à 2 048 lignes)
  PLI_ANE_FFN_LIGNES = 512 : morceaux d'au moins ce nombre de lignes (au plus 2 048) ; en dessous, chemin GPU inchangé
  PLI_ANE_BLOCS = 1 (défaut, 3.3.0) | 2 : unités coupées en deux programmes épinglés sur deux blocs (PLI_ANE_BLOCS_IDS = 1,3) ;
    sonde pli-sp1-2blocs-c (16/10/2026) : 18 unités sur deux blocs -> T = 0,76 de la 3.3.0 à 2 048 lignes
Ne prend que les MLP déjà posés par deplie._installer_mlp (PLI_LECTURE_DEPLIE=f16 et PLI_LECTURE_DEPLIE_MLP=t), poids rangés
(P2) ou non. Écriture et petits morceaux : rien ne change. Disjoncteur : à la première erreur, la couche est refaite sur le GPU
seul et le partage s'arrête pour le processus.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

ACTIF = os.environ.get("PLI_ANE_FFN", "0") == "1"
UNITES = int(os.environ.get("PLI_ANE_FFN_UNITES", "12"))
LIGNES = max(512, int(os.environ.get("PLI_ANE_FFN_LIGNES", "512")))
LIGNES_MAX = 2048
BLOCS = int(os.environ.get("PLI_ANE_BLOCS", "1"))
if BLOCS > 1:
    # deux programmes épinglés sur deux blocs : seules les puces Ultra ont deux Neural Engine
    import subprocess as _sp

    _puce = _sp.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout
    if "Ultra" not in _puce:
        print(f"[tensorfold] PLI_ANE_BLOCS={BLOCS} demande une puce Ultra ({_puce.strip() or 'puce inconnue'}) : 1 bloc", flush=True)
        BLOCS = 1
BLOCS_IDS = [int(v) for v in os.environ.get("PLI_ANE_BLOCS_IDS", "1,3").split(",")]
PAS = 128
compte = {"couches": 0, "ane": 0, "gpu_seul": 0, "requant": 0, "ane_ms": 0.0, "attente_ms": 0.0, "disjoncte": ""}
_etat: Any = None
_classes: dict = {}
_noyaux: dict = {}


def _dire(*a: Any) -> None:
    print("PLI ane_ffn :", *a, file=sys.stderr, flush=True)


# ---- noyaux Metal ----

_LECTURE_RG = ("  const size_t p = row / 32, t = (row / 8) % 4, fm = row % 8;\n"
               "  const uint w32 = W[2 * (((p * G + g) * 32 + fm * 4 + fnh) * 4 + t) + hi];\n"
               "  const uint sbw = SC[((p * G + g) * 8 + fm) * 4 + t];\n"
               "  const float scf = as_type<float>(sbw << 16), bif = as_type<float>(sbw & 0xFFFF0000u);\n")
_LECTURE = ("  const uint w32 = W[row * (K / 8) + 8 * g + 2 * fnh + hi];\n"
            "  const float scf = float(SC[row * G + g]), bif = float(BI[row * G + g]);\n")
# valeur d'un poids : même expression que deplie._noyau (échelle et décalage arrondis en bf16, scale * d + bias en bf16)
_VALEUR = ("  const bfloat sb_ = bfloat(scf), bb_ = bfloat(bif);\n"
           "  const uchar d = uchar((w32 >> (4 * (k & 7))) & 0xFu);\n"
           "  const bfloat v_ = sb_ * d + bb_;\n")


def _poids(col: str) -> str:
    """Code qui lit le poids (row, col) dans `v` (float) ; K, G, rg fixés par l'appelant."""
    return (f"  {{ const size_t c_ = {col}; const size_t g = c_ / 64, k = c_ % 64, fnh = k / 16, hi = (k & 15) >= 8 ? 1 : 0;\n")


def noyau_requant(K: int, C0: int, NC: int, BLOC: int, rg: bool) -> Any:
    """Requantification d'une matrice 4 bits pour le Neural Engine, une ligne par groupe de 256 fils : colonnes [C0, C0+NC)
    dépliées, rotation de Hadamard par blocs de BLOC (o = sg[o] / sqrt(BLOC) * somme H[o][i] v[i], transformée rapide sur un
    groupe SIMD), puis int8 par ligne (pic / 127, arrondi au pair, borné à 127) comme client.int8_lignes. Sorties : Q int8
    (lignes, NC) et E fp16 (lignes) = échelle * 128."""

    import mlx.core as mx

    cle = ("rq", K, C0, NC, BLOC, rg)
    if cle in _noyaux:
        return _noyaux[cle]
    J = BLOC // 32
    etages_reg = {128: 2, 512: 4}[BLOC]
    src = (f"  constexpr int K = {K}, G = {K // 64}, C0 = {C0}, NC = {NC}, BLOC = {BLOC}, J = {J};\n"
           "  threadgroup float vals[NC];\n  threadgroup float pics[8];\n"
           "  const uint row = threadgroup_position_in_grid.x;\n"
           "  const uint l = thread_index_in_simdgroup, sgi = simdgroup_index_in_threadgroup;\n"
           "  const uint tid = thread_position_in_threadgroup.x;\n"
           f"  const float norme = {1.0 / (BLOC ** 0.5)!r}f;\n"
           "  float pic_local = 0.0f;\n"
           "  for (int b = int(sgi); b < NC / BLOC; b += 8) {\n"
           "    float r[J];\n"
           "    for (int j = 0; j < J; j++) {\n"
           + _poids("C0 + b * BLOC + j * 32 + l")
           + (_LECTURE_RG if rg else _LECTURE).replace("\n", "\n  ")
           + _VALEUR.replace("\n", "\n  ") +
           "    r[j] = float(v_); }\n    }\n"
           # étages dans le registre : pas 32, 64, (128, 256)
           f"    for (int m = 0; m < {etages_reg}; m++) {{\n"
           "      const int s = 1 << m;\n"
           "      for (int j = 0; j < J; j++) if ((j & s) == 0) { const float a = r[j], c = r[j + s]; r[j] = a + c; r[j + s] = a - c; }\n"
           "    }\n"
           # étages entre fils : pas 1 à 16
           "    for (uint h = 1; h < 32; h <<= 1) {\n"
           "      for (int j = 0; j < J; j++) { const float a = r[j], c = simd_shuffle_xor(a, h); r[j] = (l & h) ? (c - a) : (a + c); }\n"
           "    }\n"
           "    for (int j = 0; j < J; j++) {\n"
           "      const int o = j * 32 + int(l);\n"
           "      const float y = r[j] * (SG[o] * norme);\n"
           "      vals[b * BLOC + o] = y; pic_local = max(pic_local, fabs(y));\n"
           "    }\n  }\n"
           "  pic_local = simd_max(pic_local);\n"
           "  if (l == 0) pics[sgi] = pic_local;\n"
           "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
           "  float pic = pics[0];\n  for (int i = 1; i < 8; i++) pic = max(pic, pics[i]);\n"
           "  pic = max(pic, 6.103515625e-05f);\n"
           "  const float e = pic / 127.0f;\n"
           "  for (int c = int(tid); c < NC; c += 256) {\n"
           "    const float q = clamp(rint(vals[c] / e), -127.0f, 127.0f);\n"
           "    Q[size_t(row) * NC + c] = char(q);\n  }\n"
           "  if (tid == 0) E[row] = half(e * 128.0f);\n")
    _noyaux[cle] = mx.fast.metal_kernel(name="pli_anerq_" + hashlib.sha256(src.encode()).hexdigest()[:12],
                                        input_names=["W", "SC", "BI", "SG"], output_names=["Q", "E"], source=src, header="")
    return _noyaux[cle]


def noyau_colonnes(T: str, K: int, KO: int, rg: bool) -> Any:
    """Dépliage des colonnes [0, KO) de chaque ligne d'une matrice 4 bits de K colonnes (rangée ou non), mêmes valeurs que
    deplie._noyau : un fil par (ligne, groupe de 64, quart de groupe)."""

    import mlx.core as mx

    cle = ("col", T, K, KO, rg)
    if cle in _noyaux:
        return _noyaux[cle]
    src = (f"  constexpr int K = {K}, G = {K // 64}, GO = {KO // 64}, KO = {KO};\n  typedef {T} T;\n"
           "  const size_t i = thread_position_in_grid.x;\n"
           "  const size_t fnh = i & 3, g = (i >> 2) % GO, row = i / (4 * GO);\n"
           "  device T* o = OUT + row * KO + g * 64 + fnh * 16;\n"
           "  for (int hi = 0; hi < 2; hi++) {\n"
           + (_LECTURE_RG if rg else _LECTURE).replace("\n", "\n  ") +
           "  const bfloat sb_ = bfloat(scf), bb_ = bfloat(bif);\n"
           "    for (int k = 0; k < 8; k++) {\n"
           "      const uchar d = uchar((w32 >> (4 * k)) & 0xFu);\n"
           "      const bfloat v_ = sb_ * d + bb_;\n      o[hi * 8 + k] = T(float(v_));\n    }\n  }\n")
    _noyaux[cle] = mx.fast.metal_kernel(name="pli_anecol_" + hashlib.sha256(src.encode()).hexdigest()[:12],
                                        input_names=["W", "SC", "BI"], output_names=["OUT"], source=src, header="")
    return _noyaux[cle]


def _rg(m: Any) -> bool:
    import mlx.core as mx

    return m["scales"].dtype == mx.uint32


def requant(m: Any, n0: int, ane: int) -> Any:
    """Poids de la part Neural Engine du MLP m : (qg, eg, qu, eu, qd, ed), int8 et fp16 (échelle * 128)."""

    import mlx.core as mx

    E = _etat
    out = []
    for p in (m.gate_proj, m.up_proj):
        w, s, b = p["weight"], p["scales"], p["biases"]
        K = int(w.shape[1]) * 8
        k = noyau_requant(K, 0, K, 128, _rg(p))
        out += k(inputs=[w[n0:n0 + ane], s[n0:n0 + ane], b[n0:n0 + ane], E.sg128], grid=(ane * 256, 1, 1), threadgroup=(256, 1, 1),
                 output_shapes=[(ane, K), (ane,)], output_dtypes=[mx.int8, mx.float16])
    d = m.down_proj
    w, s, b = d["weight"], d["scales"], d["biases"]
    N, K = int(w.shape[0]), int(w.shape[1]) * 8
    k = noyau_requant(K, n0, ane, 512, _rg(d))
    out += k(inputs=[w, s, b, E.sg512], grid=(N * 256, 1, 1), threadgroup=(256, 1, 1),
             output_shapes=[(N, ane), (N,)], output_dtypes=[mx.int8, mx.float16])
    return out


def entrees(x2: Any) -> tuple[Any, Any]:
    """Entrées du Neural Engine : x tourné par blocs de 128, int8 par jeton, rangé par canal (H, lignes) ; tx = échelle * 128."""

    import mlx.core as mx

    E = _etat
    rows, H = int(x2.shape[0]), int(x2.shape[1])
    xr = (x2.astype(mx.float32).reshape(rows, H // 128, 128) @ E.r128t).reshape(rows, H)
    pic = mx.maximum(mx.abs(xr).max(axis=1), 2.0 ** -14)
    sx = pic / 127.0
    q = mx.clip(mx.round(xr / sx[:, None]), -127, 127).astype(mx.int8)
    return mx.contiguous(q.T), (sx * 128.0).astype(mx.float16)


def part_gpu(m: Any, x2: Any, n0: int) -> Any:
    """Part GPU, transposée : yT (H, lignes) en bf16 de la somme sur les canaux [0, n0). Mêmes opérations que deplie._mlp_t."""

    import mlx.core as mx

    from tensorfold.pli import deplie

    if deplie._swc is None:
        from mlx_lm.models.activations import swiglu

        deplie._swc = mx.compile(lambda a, b: swiglu(a.astype(mx.bfloat16), b.astype(mx.bfloat16)), shapeless=True)
    g, u, d = m.gate_proj, m.up_proj, m.down_proj

    def tete(p: Any) -> Any:
        return deplie._deplier_wsb(p["weight"][:n0], p["scales"][:n0], p["biases"][:n0], p.bits, p.group_size,
                                   getattr(p, "mode", "affine"), mx.float16)

    xt = x2.astype(mx.float16).T
    ht = deplie._swc(tete(g) @ xt, tete(u) @ xt)
    w, s, b = d["weight"], d["scales"], d["biases"]
    N, K = int(w.shape[0]), int(w.shape[1]) * 8
    wd = noyau_colonnes("bfloat", K, n0, _rg(d))(inputs=[w, s, b], grid=(N * (n0 // 64) * 4, 1, 1), threadgroup=(256, 1, 1),
                                                 output_shapes=[(N, n0)], output_dtypes=[mx.bfloat16])[0]
    return wd @ ht


class _Etat:
    """Programmes, surfaces et liaisons du partage. PLI_ANE_BLOCS = 1 : un programme, bloc choisi par le service (3.3.0).
    PLI_ANE_BLOCS = 2 : les unités sont coupées en deux parts égales, chacune un programme épinglé sur son bloc
    (PLI_ANE_BLOCS_IDS, défaut 1,3 ; sonde pli-sp1-2blocs-c : 18 unités, T = 0,76 de la 3.3.0), évaluées ensemble sur deux fils.
    Les parts lisent les mêmes surfaces d'entrée ; leurs sorties s'ajoutent à la part GPU en fp32."""

    def __init__(self, client: Any, H: int, N: int):
        import mlx.core as mx
        import numpy as np

        self.c = client
        self.nparts = BLOCS
        if UNITES % BLOCS:
            raise ValueError(f"PLI_ANE_FFN_UNITES={UNITES} n'est pas divisible par PLI_ANE_BLOCS={BLOCS}")
        self.f = client.Forme(H, N, UNITES // BLOCS)      # forme d'une part (le décalage des canaux n'y entre pas)
        self.H, self.N, self.ane = H, N, self.f.ane       # ane : canaux d'une part
        self.n0 = N - BLOCS * self.ane                    # canaux [0, n0) sur le GPU
        self.debuts = [self.n0 + b * self.ane for b in range(BLOCS)]
        sg = client.signes()
        self.sg128 = mx.array(sg[:128])
        self.sg512 = mx.array(sg[:512])
        self.r128t = mx.array(np.ascontiguousarray(client.rotation(128, sg).T))
        # jeux[s][b] : jeu de surfaces s (couche paire ou impaire), part b ; les parts d'un jeu partagent x et tx
        self.jeux = []
        for _ in range(2):
            parts = [client.Memoire(self.f, LIGNES_MAX) for _ in range(BLOCS)]
            for m in parts[1:]:
                m.x, m.tx = parts[0].x, parts[0].tx
            self.jeux.append(parts)
        self.tag = [-1, -1]
        self.fonctions = list(range(-(-LIGNES // PAS) * PAS, LIGNES_MAX + 1, PAS))
        mil = client.programme_mil(self.f, self.jeux[0][0], self.fonctions)
        cache = subprocess.run(["getconf", "DARWIN_USER_CACHE_DIR"], capture_output=True, text=True).stdout.strip() + "pli-ane"
        os.makedirs(cache, exist_ok=True)
        t0 = time.perf_counter()
        self.progs = []
        for b in range(BLOCS):
            client.bloc_suivant(BLOCS_IDS[b] if BLOCS > 1 else 0)
            try:
                self.progs.append(client.Programme(mil, client.blob_rotation(self.ane, sg), cache))
            finally:
                client.bloc_suivant(0)
        self.chargement_s = time.perf_counter() - t0
        self.liens: list[list[dict]] = []
        for parts in self.jeux:
            par_part = []
            for b, j in enumerate(parts):
                prog = self.progs[b]
                surf = {nom: s for nom, s, _, _ in j.entrees(self.f)}
                d = {}
                for r in self.fonctions:
                    p = prog.procedure(f"ffn{r}")
                    d[r] = prog.lier(p, [surf[n] for n in prog.entrees(p)], j.sortie)
                par_part.append(d)
            self.liens.append(par_part)
        self.fil = ThreadPoolExecutor(max_workers=BLOCS, thread_name_prefix="pli-ane")
        self.actif = True

    def octets(self) -> int:
        """Octets des surfaces (x et tx comptés une fois par jeu)."""
        tout = 0
        for parts in self.jeux:
            tout += parts[0].octets()
            for m in parts[1:]:
                tout += m.octets() - sum(x.size for x in m.x) - m.tx.size
        return tout

    def requant(self, m: Any) -> list:
        return [requant(m, d, self.ane) for d in self.debuts]

    def ecrire_poids(self, s: int, poids: list) -> None:
        import numpy as np

        S = self.c.SEGMENT
        for b, pb in enumerate(poids):
            qg, eg, qu, eu, qd, ed = (np.asarray(a) for a in pb)
            j = self.jeux[s][b]
            for k in range(self.f.seg_entree):
                j.wg[k].ecrire(qg[:, k * S:(k + 1) * S])
                j.wu[k].ecrire(qu[:, k * S:(k + 1) * S])
            j.sg.ecrire(eg[:, None])
            j.su.ecrire(eu[:, None])
            debut = 0
            for i, w in enumerate(self.f.seg_down):
                j.wd[i].ecrire(qd[:, debut:debut + w])
                debut += w
            j.sd.ecrire(ed[:, None])

    def couche(self, m: Any, x: Any, rows: int) -> Any:
        import mlx.core as mx
        import numpy as np

        i, H, n0, S = m._pli_ane_i, self.H, self.n0, self.c.SEGMENT
        s = i % 2
        fn = -(-rows // PAS) * PAS
        x2 = x.reshape(rows, H)
        qT, tx = entrees(x2)
        a_evaluer = [qT, tx]
        poids = None
        if self.tag[s] != i:
            poids = self.requant(m)
            a_evaluer += [a for pb in poids for a in pb]
        mx.eval(a_evaluer)
        parts = self.jeux[s]
        if poids is not None:
            self.tag[s] = -1
            self.ecrire_poids(s, poids)
            self.tag[s] = i
            compte["requant"] += 1
        qn = np.asarray(qT)
        j0 = parts[0]
        for k in range(self.f.seg_entree):
            j0.x[k].tableau()[:, :rows] = qn[k * S:(k + 1) * S]
        j0.tx.tableau()[0, :rows] = np.asarray(tx)
        futs = [self.fil.submit(self.progs[b].evaluer, self.liens[s][b][fn]) for b in range(self.nparts)]
        # poids de la couche suivante d'abord sur le GPU, puis la part GPU de cette couche
        suivant = m._pli_ane_suivant
        s2 = suivant._pli_ane_i % 2
        p2 = self.requant(suivant) if (s2 != s and self.tag[s2] != suivant._pli_ane_i) else None
        if p2 is not None:
            mx.async_eval([a for pb in p2 for a in pb])
        yg = part_gpu(m, x2, n0)
        mx.async_eval(yg)
        if p2 is not None:
            mx.eval([a for pb in p2 for a in pb])
            self.tag[s2] = -1
            self.ecrire_poids(s2, p2)
            self.tag[s2] = suivant._pli_ane_i
            compte["requant"] += 1
        t0 = time.perf_counter()
        compte["ane_ms"] += max(f.result(timeout=10.0) for f in futs)
        compte["attente_ms"] += (time.perf_counter() - t0) * 1e3
        f32 = mx.float32
        txf = tx.astype(f32)
        yT = yg.astype(f32)
        for j in parts:   # dans l'ordre des parts : ordre de somme fixe
            ys = mx.array(np.ascontiguousarray(j.sortie.tableau()[:H + 1, :rows]))
            yT = yT + ys[:H].astype(f32) * (ys[H].astype(f32) * txf)[None, :]
        compte["ane"] += 1
        return mx.contiguous(yT.astype(mx.bfloat16).T).reshape(*x.shape[:-1], H)

    def disjoncter(self, e: BaseException) -> None:
        self.actif = False
        compte["disjoncte"] = f"{type(e).__name__}: {e}"
        _dire(f"DISJONCTEUR : {compte['disjoncte']} ; GPU seul pour la suite du processus")


def _mlp_ane(self: Any, x: Any) -> Any:
    import mlx.core as mx

    base = type(self)._pli_ane_base
    E = _etat
    rows = x.size // int(x.shape[-1]) if x.ndim >= 2 else 0
    if E is None or not E.actif or rows < LIGNES or rows > LIGNES_MAX or x.dtype != mx.bfloat16:
        if E is not None and rows >= LIGNES:
            compte["gpu_seul"] += 1
        return base.__call__(self, x)
    try:
        return E.couche(self, x, rows)
    except Exception as e:  # disjoncteur : couche refaite sur le GPU seul
        E.disjoncter(e)
        return base.__call__(self, x)


def installer(couches: Any) -> int:
    """Pose le partage sur les MLP transposés (deplie._mlp_t) des couches ; rend le nombre de couches prises (0 = GPU seul)."""

    global _etat
    if not ACTIF or couches is None or _etat is not None:
        return 0
    mlps = [getattr(c, "mlp", None) for c in couches]
    mlps = [m for m in mlps if m is not None and getattr(type(m), "_pli_mlp_t", False)]
    if not mlps:
        _dire("aucun MLP transposé (PLI_LECTURE_DEPLIE=f16 et PLI_LECTURE_DEPLIE_MLP=t requis) : GPU seul")
        return 0
    try:
        from tensorfold.pli.ane import client

        raison = client.indisponible()
        if raison:
            _dire(f"Neural Engine indisponible ({raison}) : GPU seul")
            return 0
        w = mlps[0].gate_proj["weight"]
        N, H = int(w.shape[0]), int(w.shape[1]) * 8
        if any(int(m.gate_proj["weight"].shape[0]) != N or int(m.down_proj["weight"].shape[0]) != H for m in mlps):
            _dire("formes de MLP différentes d'une couche à l'autre : GPU seul")
            return 0
        _etat = _Etat(client, H, N)
    except Exception as e:
        _dire(f"mise en place impossible ({type(e).__name__}: {e}) : GPU seul")
        _etat = None
        return 0
    for i, m in enumerate(mlps):
        base = type(m)
        if base not in _classes:
            _classes[base] = type(base.__name__ + "A", (base,), {"__call__": _mlp_ane, "_pli_ane_base": base})
        object.__setattr__(m, "_pli_ane_i", i)
        m.__class__ = _classes[base]
    for i, m in enumerate(mlps):
        object.__setattr__(m, "_pli_ane_suivant", mlps[(i + 1) % len(mlps)])
    compte["couches"] = len(mlps)
    E = _etat
    blocs = f", {E.nparts} blocs ({','.join(map(str, BLOCS_IDS[:E.nparts]))})" if E.nparts > 1 else ""
    _dire(f"{len(mlps)} MLP partagés, {UNITES} unités ({E.nparts * E.ane} canaux sur {E.N}, part {E.nparts * E.ane / E.N:.3f}{blocs}), "
          f"morceaux de {LIGNES} à {LIGNES_MAX} lignes, {len(E.fonctions)} fonctions chargées en {E.chargement_s:.1f} s, "
          f"surfaces {E.octets() / 2 ** 20:.0f} Mio")
    return len(mlps)
