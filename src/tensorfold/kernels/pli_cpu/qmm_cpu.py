"""PLI : chargement et appel du noyau CPU qmm_cpu.c (produit y = x . W^T sur les poids 4 bits de MLX, sans copie).

Rien n'est branché dans le moteur : ce module sert aux bancs. La bibliothèque est compilée à côté du fichier C au premier appel.
"""
import ctypes, os, subprocess
import numpy as np

_ICI = os.path.dirname(os.path.abspath(__file__))
_LIB = None


def charger(dossier=None):
    """Compile qmm_cpu.c si besoin et rend la bibliothèque."""
    global _LIB
    if _LIB is not None:
        return _LIB
    d = dossier or _ICI
    import hashlib
    src = os.path.join(d, "qmm_cpu.c")
    # le nom de la bibliothèque porte l'empreinte du source : un source changé est toujours recompilé (les dates ne suffisent pas,
    # une copie garde la date d'origine : arrêt du banc pli-e15b le 10/10)
    lib = os.path.join(d, "libpliqmm-%s.dylib" % hashlib.sha256(open(src, "rb").read()).hexdigest()[:12])
    if not os.path.exists(lib):
        subprocess.run(["clang", "-O3", "-mcpu=apple-m1", "-shared", "-fblocks", "-ffp-contract=off", "-Wno-deprecated-declarations", src, "-framework", "Accelerate", "-o", lib],
                       check=True)
    L = ctypes.CDLL(lib)
    p, n = ctypes.c_void_p, ctypes.c_long
    for nom in ("pli_qmm_f32", "pli_qmm_f16"):
        getattr(L, nom).argtypes = [p, p, p, ctypes.c_int, n, n, p, n, p, n, n, ctypes.c_int]
        getattr(L, nom).restype = ctypes.c_int
    L.pli_qmm_blas.argtypes = [p, p, p, ctypes.c_int, n, n, p, n, p, n, n, n]
    L.pli_qmm_blas.restype = ctypes.c_int
    L.pli_qmm_priorite.argtypes = [ctypes.c_int]
    L.pli_qmm_ent.argtypes = [p, p, p, ctypes.c_int, n, n, p, p, p, n, p, n, n, ctypes.c_int]
    L.pli_qmm_ent.restype = ctypes.c_int
    L.pli_qmm_priorite.restype = None
    L.pli_qmm_e15.argtypes = [p, p, p, ctypes.c_int, n, n, p, n, p, n, n, ctypes.c_int, ctypes.c_int]
    L.pli_qmm_e15.restype = ctypes.c_int
    L.pli_swiglu.argtypes = [p, p, n]
    L.pli_swiglu.restype = ctypes.c_int
    _LIB = L
    return L


def vue(tab):
    """Vue numpy en octets sur le tampon d'un tableau (MLX ou numpy), sans copie ; None si le tampon ne se laisse pas lire."""
    if isinstance(tab, np.ndarray):
        return np.ascontiguousarray(tab).view(np.uint8).reshape(-1)
    try:
        return np.frombuffer(memoryview(tab), dtype=np.uint8)
    except Exception:
        try:
            return np.asarray(memoryview(tab).cast("B"))
        except Exception:
            return None


class Poids:
    """Une matrice 4 bits de MLX : weight (uint32 [N, K/8]), scales et biases ([N, K/64], BF16, float16 ou float32)."""

    def __init__(self, w, s, b, sb=0):
        self.N, self.K = int(w.shape[0]), int(w.shape[1]) * 8
        self.sb = sb
        self._garde = [w, s, b]
        self.vw, self.vs, self.vb = vue(w), vue(s), vue(b)
        if self.vw is None or self.vs is None or self.vb is None:
            raise ValueError("tampon illisible sans copie")
        octets = {0: 2, 1: 2, 2: 4}[sb]
        if self.vw.size != self.N * self.K // 2 or self.vs.size != self.N * (self.K // 64) * octets:
            raise ValueError("tailles inattendues : %d %d" % (self.vw.size, self.vs.size))

    def adresses(self):
        return self.vw.ctypes.data, self.vs.ctypes.data, self.vb.ctypes.data


def produit(variante, poids, x, n0=0, n1=None, fils=0, bloc=4096):
    """y [L, n1 - n0] en float32. variante : « f32 », « f16 » (x en float16) ou « blas » (x en float32)."""
    L = charger()
    n1 = poids.N if n1 is None else n1
    x = np.ascontiguousarray(x, dtype=np.float16 if variante == "f16" else np.float32)
    lignes = x.shape[0]
    y = np.empty((lignes, n1 - n0), dtype=np.float32)
    aw, a_s, ab = poids.adresses()
    if variante == "blas":
        r = L.pli_qmm_blas(aw, a_s, ab, poids.sb, poids.N, poids.K, x.ctypes.data, lignes, y.ctypes.data, n0, n1, bloc)
    else:
        f = L.pli_qmm_f16 if variante == "f16" else L.pli_qmm_f32
        r = f(aw, a_s, ab, poids.sb, poids.N, poids.K, x.ctypes.data, lignes, y.ctypes.data, n0, n1, fils)
    if r:
        raise RuntimeError("noyau CPU : code %d" % r)
    return y


def priorite(niveau):
    """Priorité des fils du noyau : 0 = haute (défaut), 1 = moyenne, 2 = de fond."""
    charger().pli_qmm_priorite(int(niveau))


def swiglu(g, u):
    """Sur place dans g (float32) : silu(g) * u."""
    charger().pli_swiglu(g.ctypes.data, u.ctypes.data, g.size)
    return g


def entiers(x, bits=16):
    """Activations x [L, K] en entiers, une échelle par ligne : (xh, xl, ech). 16 bits : valeur = 256 * xh + xl, moitiés centrées
    (-128..127). 8 bits : valeur = xh, xl = None."""
    x = np.asarray(x, dtype=np.float64); lim = 127 if bits == 8 else 32639   # 127 * 256 + 127 : la moitié haute tient dans -128..127
    ech = np.abs(x).max(axis=1, keepdims=True) / lim; ech[ech == 0] = 1.0; v = np.round(x / ech).astype(np.int64)
    if bits == 8:
        return np.ascontiguousarray(v, dtype=np.int8), None, np.ascontiguousarray(ech[:, 0], dtype=np.float32)
    xl = ((v + 128) & 255) - 128; xh = (v - xl) >> 8
    return np.ascontiguousarray(xh, dtype=np.int8), np.ascontiguousarray(xl, dtype=np.int8), np.ascontiguousarray(ech[:, 0], dtype=np.float32)


def produit_entiers(poids, xh, xl, ech, n0=0, n1=None, fils=0):
    """Produit « entiers d'abord » : y [L, n1 - n0] en float32 (voir pli_qmm_ent dans qmm_cpu.c pour l'ordre exact du calcul)."""
    L = charger(); n1 = poids.N if n1 is None else n1; lignes = xh.shape[0]
    y = np.empty((lignes, n1 - n0), dtype=np.float32); aw, a_s, ab = poids.adresses()
    r = L.pli_qmm_ent(aw, a_s, ab, poids.sb, poids.N, poids.K, xh.ctypes.data, None if xl is None else xl.ctypes.data, ech.ctypes.data,
                      lignes, y.ctypes.data, n0, n1, fils)
    if r:
        raise RuntimeError("noyau CPU entiers : code %d" % r)
    return y


def produit_entiers_numpy(q, s, b, xh, xl, ech):
    """Le même calcul en numpy, même ordre, pour vérifier les bits : q [N, K] entiers 0..15, s et b [N, K/64] float32."""
    N, K = q.shape; G = K // 64; Lg = xh.shape[0]; acc = np.zeros((Lg, N), dtype=np.float32); f = np.float32
    for g in range(G):
        qg = q[:, 64 * g:64 * g + 64].astype(np.int64); h = xh[:, 64 * g:64 * g + 64].astype(np.int64)
        ph = (h @ qg.T).astype(f); sxh = h.sum(1).astype(f)[:, None]
        if xl is None:
            pl = np.zeros_like(ph); sxl = np.zeros_like(sxh)
        else:
            lo = xl[:, 64 * g:64 * g + 64].astype(np.int64); pl = (lo @ qg.T).astype(f); sxl = lo.sum(1).astype(f)[:, None]
        sg = s[:, g].astype(f)[None, :]; bg = b[:, g].astype(f)[None, :]; kh = f(1 if xl is None else 256)
        u = (sg * kh) * ph + sg * pl; v = (bg * kh) * sxh + bg * sxl
        acc = acc + (u + v)
    return acc * ech.astype(f)[:, None]


ENT_MAX = 16383      # comme simd_qmm.ENT_MAX : la somme d'un groupe de 64 produits reste exacte en flottant 32 bits


def morceaux(n, k):
    """Nombre de morceaux du noyau de TensorFold (simd_qmm.splits) : fixe l'ordre des additions."""
    return 32 if n <= 64 else (16 if n <= 6144 else 8)


def entiers15(x):
    """x [L, K] flottant -> (xi int16, exposants int32 [L]) : xi = arrondi(x * 2^-e), e le plus petit exposant tel que |xi| <= ENT_MAX.
    Échelle en puissance de 2 : la multiplication est exacte, seul l'arrondi (au plus proche, pair en cas d'égalité) joue."""
    x = np.asarray(x, dtype=np.float32); m = np.abs(x).max(axis=1)
    e = np.where(m > 0, np.ceil(np.log2(np.maximum(m, 1e-38) / ENT_MAX)), 0).astype(np.int32)
    xi = np.rint(np.ldexp(x, -e[:, None]))
    while True:                                   # l'arrondi peut dépasser d'une unité : on monte l'exposant de ces lignes
        trop = np.abs(xi).max(axis=1) > ENT_MAX
        if not trop.any():
            break
        e[trop] += 1; xi[trop] = np.rint(np.ldexp(x[trop], -e[trop][:, None]))
    return np.ascontiguousarray(xi, dtype=np.int16), e


def produit_e15(poids, xi, n0=0, n1=None, fils=0):
    """Produit en entiers d'abord, ordre du noyau GPU de TensorFold : y [L, n1 - n0] en float32, sans l'échelle de x."""
    L = charger(); n1 = poids.N if n1 is None else n1; xi = np.ascontiguousarray(xi, dtype=np.int16); lignes = xi.shape[0]
    y = np.empty((lignes, n1 - n0), dtype=np.float32); aw, a_s, ab = poids.adresses()
    r = L.pli_qmm_e15(aw, a_s, ab, poids.sb, poids.N, poids.K, xi.ctypes.data, lignes, y.ctypes.data, n0, n1, morceaux(poids.N, poids.K), fils)
    if r:
        raise RuntimeError("noyau CPU e15 : code %d" % r)
    return y
