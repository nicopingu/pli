"""tensorfold.pli.ane.client : client du Neural Engine pour PLI (15/10/2026).

Enveloppe ctypes de libpli_ane.dylib (client privé repris de Splash 1.3.2-m1, Apache-2.0) et générateur du programme MIL
du FFN partagé par canaux, repris de Splash runtime/ops/AneFfn.cpp (AneFfn::function), réécrit en Python.

Le Neural Engine calcule les canaux intermédiaires de queue d'une couche FFN SwiGLU en W8A8 :
  entrées x (int8, rangées par canal, rotation de Hadamard par blocs de 128) et leurs échelles par jeton tx ;
  poids wg, wu (int8 par ligne, mêmes rotations) et échelles sg, su ; wd (int8, rotation des entrées intermédiaires par
  blocs de 512) et sd. Sortie : ds (lignes de down avant l'échelle des jetons) puis une ligne hscale.
  Le résultat vaut y[t][c] = ds[c][t] * hscale[t] * tx[t] (jonction en fp32).
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
BIBLIO = os.path.join(ICI, "natif", "libpli_ane.dylib")
if not os.path.exists(BIBLIO):
    # première utilisation sur cette machine : compilée par construire.sh (clang++ des Command Line Tools)
    import subprocess

    fait = subprocess.run(["bash", os.path.join(ICI, "construire.sh")], capture_output=True, text=True)
    if fait.returncode != 0 or not os.path.exists(BIBLIO):
        raise OSError("libpli_ane.dylib absente et non compilée (xcode-select --install, puis "
                      f"bash {os.path.join(ICI, 'construire.sh')}) : {(fait.stderr or fait.stdout)[-300:]}")
_lib = ctypes.CDLL(BIBLIO)

V, P, U32, U64, D, S, I = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32, ctypes.c_uint64,
                           ctypes.c_double, ctypes.c_char_p, ctypes.c_int)
for nom, res, args in [
    ("pli_ane_error", S, []),
    ("pli_ane_unavailable", I, [ctypes.c_char_p, ctypes.c_size_t]),
    ("pli_ane_surface", V, [U32, U32, I]),
    ("pli_ane_surface_free", None, [V]),
    ("pli_ane_surface_base", V, [V]),
    ("pli_ane_surface_stride", U32, [V]),
    ("pli_ane_surface_size", U64, [V]),
    ("pli_ane_surface_type", I, [V, U32, U32, ctypes.c_char_p, ctypes.c_size_t]),
    ("pli_ane_surface_strides", I, [V, U32, ctypes.c_char_p, ctypes.c_size_t]),
    ("pli_ane_blob", ctypes.c_size_t, [V, ctypes.c_size_t, V, ctypes.c_size_t]),
    ("pli_ane_program", V, [S, V, ctypes.c_size_t, S, D]),
    ("pli_ane_program_free", None, [V]),
    ("pli_ane_procedure", I, [V, S]),
    ("pli_ane_inputs", I, [V, I, ctypes.c_char_p, ctypes.c_size_t]),
    ("pli_ane_bind", V, [V, I, P, I, V]),
    ("pli_ane_bind_free", None, [V]),
    ("pli_ane_eval", D, [V, V, D]),
    ("pli_ane_next_block", None, [I]),
]:
    f = getattr(_lib, nom)
    f.restype, f.argtypes = res, args


def bloc_suivant(bloc: int) -> None:
    """PLI (16/10/2026) : bloc du Neural Engine (1 à 3 ; 0 : le service choisit) imposé aux programmes construits ensuite."""
    _lib.pli_ane_next_block(int(bloc))


class AneError(RuntimeError):
    pass


def _erreur(quoi: str) -> AneError:
    return AneError(f"{quoi} : {_lib.pli_ane_error().decode(errors='replace')}")


def _texte(f, *args) -> str:
    n = f(*args, None, 0)
    buf = ctypes.create_string_buffer(max(n, 0) + 1)
    f(*args, buf, len(buf))
    return buf.value.decode()


def indisponible() -> str:
    """Chaîne vide si le Neural Engine est joignable, sinon la raison."""
    return _texte(_lib.pli_ane_unavailable)


class Surface:
    """IOSurface de `rows` lignes de `width` éléments (int8 ou fp16), chaque ligne alignée sur 64 octets."""

    def __init__(self, rows: int, width: int, int8: bool):
        self.h = _lib.pli_ane_surface(rows, width, int(int8))
        if not self.h:
            raise _erreur("surface")
        self.rows, self.width, self.int8 = rows, width, int8
        self.stride = _lib.pli_ane_surface_stride(self.h)
        self.size = _lib.pli_ane_surface_size(self.h)
        self.base = _lib.pli_ane_surface_base(self.h)

    def tableau(self) -> np.ndarray:
        """Vue numpy [rows, stride / taille] sans copie (mémoire unifiée)."""
        brut = np.ctypeslib.as_array((ctypes.c_uint8 * self.size).from_address(self.base))
        vue = brut[: self.rows * self.stride].reshape(self.rows, self.stride)
        return vue.view(np.int8) if self.int8 else vue.view(np.float16)

    def ecrire(self, valeurs: np.ndarray) -> None:
        t = self.tableau()
        t[: valeurs.shape[0], : valeurs.shape[1]] = valeurs

    def type(self, rows: int, width: int) -> str:
        return _texte(_lib.pli_ane_surface_type, self.h, rows, width)

    def strides(self, rows: int) -> str:
        return _texte(_lib.pli_ane_surface_strides, self.h, rows)

    def mil(self) -> str:
        return "int8" if self.int8 else "fp16"

    def __del__(self):
        if getattr(self, "h", None):
            _lib.pli_ane_surface_free(self.h)
            self.h = None


def blob(valeurs_f16: np.ndarray) -> bytes:
    v = np.ascontiguousarray(valeurs_f16, dtype=np.float16)
    n = _lib.pli_ane_blob(v.ctypes.data, v.size, None, 0)
    out = (ctypes.c_uint8 * n)()
    _lib.pli_ane_blob(v.ctypes.data, v.size, ctypes.addressof(out), n)
    return bytes(out)


class Programme:
    def __init__(self, mil: str, poids: bytes, cache: str, limite_s: float = 300.0):
        self._poids = (ctypes.c_uint8 * len(poids)).from_buffer_copy(poids)
        self.h = _lib.pli_ane_program(mil.encode(), ctypes.addressof(self._poids), len(poids), cache.encode(), limite_s)
        if not self.h:
            raise _erreur("programme")

    def procedure(self, fonction: str) -> int:
        p = _lib.pli_ane_procedure(self.h, fonction.encode())
        if p < 0:
            raise _erreur("procédure " + fonction)
        return p

    def entrees(self, procedure: int) -> list[str]:
        return _texte(_lib.pli_ane_inputs, self.h, procedure).split("\n")

    def lier(self, procedure: int, entrees: list[Surface], sortie: Surface):
        tab = (ctypes.c_void_p * len(entrees))(*[s.h for s in entrees])
        b = _lib.pli_ane_bind(self.h, procedure, tab, len(entrees), sortie.h)
        if not b:
            raise _erreur("liaison")
        return b

    def evaluer(self, liaison, delai_s: float = 10.0) -> float:
        ms = _lib.pli_ane_eval(self.h, liaison, delai_s)
        if ms < 0:
            raise _erreur("évaluation")
        return ms


# ---- programme MIL du FFN partagé (repris de AneFfn::function) ----

SEGMENT = 2560          # kSegment
BLOC_ENTREE = 128       # ANE_FFN_INPUT_BLOCK
BLOC_INTER = 512        # ANE_FFN_INTERMEDIATE_BLOCK
UNITE_CANAUX = 512      # kChannelUnit
UNITE_INT8 = 128.0      # ANE_FFN_INT8_UNIT
PIC_INT8 = 127.0        # ANE_FFN_INT8_PEAK
PLANCHER_INTER = 2.0 ** -9  # kIntermediateFloor
PLANCHER_PIC = 2.0 ** -14   # ANE_FFN_PEAK_FLOOR


def fp16(v: float) -> str:
    """Constante fp16 écrite exactement, comme le %a du C de Splash (0x1p-7, 0x1.fcp+6). Pas de snprintf par ctypes :
    sur arm64 d'Apple, ctypes passe mal un double à une fonction à arguments variables (tout sortait 0x0p+0)."""
    h = float(v).hex()                      # '0x1.fc00000000000p+6'
    mant, exp = h.split("p")
    if "." in mant:
        mant = mant.rstrip("0").rstrip(".")
    return f"fp16({mant}p{exp})"


def dims(rows: int, width: int) -> str:
    return f"[1, 1, {rows}, {width}]"


def tensor(t: str, rows: int, width: int) -> str:
    return f"tensor<{t}, {dims(rows, width)}>"


def segments(n: int) -> list[int]:
    return [min(SEGMENT, n - b) for b in range(0, n, SEGMENT)]


class Forme:
    def __init__(self, hidden: int, intermediate: int, unites: int):
        if hidden % SEGMENT:
            raise ValueError("hidden doit être un multiple de 2560")
        self.hidden, self.intermediate = hidden, intermediate
        self.ane = unites * UNITE_CANAUX
        self.gpu = intermediate - self.ane
        if self.ane <= 0 or self.gpu <= 0:
            raise ValueError("partage hors bornes")
        self.seg_entree = hidden // SEGMENT
        self.seg_down = segments(self.ane)

    @property
    def part(self) -> float:
        return self.ane / self.intermediate


class Memoire:
    """Les surfaces d'un jeu (une couche) pour des morceaux de `lignes_max` lignes au plus."""

    def __init__(self, f: Forme, lignes_max: int):
        self.x = [Surface(SEGMENT, lignes_max, True) for _ in range(f.seg_entree)]
        self.tx = Surface(1, lignes_max, False)
        self.wg = [Surface(f.ane, SEGMENT, True) for _ in range(f.seg_entree)]
        self.wu = [Surface(f.ane, SEGMENT, True) for _ in range(f.seg_entree)]
        self.sg = Surface(f.ane, 1, False)
        self.su = Surface(f.ane, 1, False)
        self.wd = [Surface(f.hidden, w, True) for w in f.seg_down]
        self.sd = Surface(f.hidden, 1, False)
        self.sortie = Surface(f.hidden + 1, lignes_max, False)

    def entrees(self, f: Forme):
        """(nom, surface, lignes, largeur) ; largeur 0 = lignes du morceau. Ordre de programInputs."""
        e = [(f"x{k}", s, SEGMENT, 0) for k, s in enumerate(self.x)]
        e.append(("tx", self.tx, 1, 0))
        for k in range(f.seg_entree):
            e.append((f"wg{k}", self.wg[k], f.ane, SEGMENT))
            e.append((f"wu{k}", self.wu[k], f.ane, SEGMENT))
        e.append(("sg", self.sg, f.ane, 1))
        e.append(("su", self.su, f.ane, 1))
        for i, w in enumerate(f.seg_down):
            e.append((f"wd{i}", self.wd[i], f.hidden, w))
        e.append(("sd", self.sd, f.hidden, 1))
        return e

    def octets(self) -> int:
        tout = self.x + [self.tx] + self.wg + self.wu + [self.sg, self.su] + self.wd + [self.sd, self.sortie]
        return sum(s.size for s in tout)


def fonction(f: Forme, entrees, sortie: Surface, rows: int) -> str:
    unit = fp16(1.0 / UNITE_INT8)
    params: list[str] = []
    body: list[str] = []
    line = body.append

    def f16(nom, h, w, expr):
        line(f"{tensor('fp16', h, w)} {nom} = {expr}")

    for nom, s, r, w in entrees:
        largeur = w if w else rows
        params.append(f"{s.type(r, largeur)} {nom}")
        line(f"{tensor(s.mil(), r, largeur)} {nom}_t = tensor_buffer_to_tensor<ios17>(input = {nom})")

    def matmul(nom, poids, valeurs, h, w):
        f16(poids + "_d", h, w, f"dequantize(input = {poids}_t, scale = {unit})")
        line(f"{tensor('fp16', h, rows)} {nom} = matmul(transpose_x = bool(false), transpose_y = bool(false), "
             f"x = {poids}_d, y = {valeurs})")

    def somme(prefixe, termes, h):
        total = prefixe + "0"
        for t in range(1, termes):
            suivant = f"{prefixe}_sum{t}"
            f16(suivant, h, rows, f"add(x = {total}, y = {prefixe}{t})")
            total = suivant
        return total

    c, ns = f.ane, f.seg_entree
    for k in range(ns):
        f16(f"x{k}_d", SEGMENT, rows, f"dequantize(input = x{k}_t, scale = {unit})")
    for p in ("g", "u"):
        for k in range(ns):
            matmul(f"{p}m{k}", f"w{p}{k}", f"x{k}_d", c, SEGMENT)
        f16(f"{p}s", c, rows, f"mul(x = {somme(p + 'm', ns, c)}, y = s{p}_t)")
    f16("gt", c, rows, "mul(x = gs, y = tx_t)")
    f16("gh", c, rows, f"mul(x = gt, y = {fp16(0.5)})")
    f16("th", c, rows, "tanh(x = gh)")
    f16("tp", c, rows, f"add(x = th, y = {fp16(1.0)})")
    f16("silu", c, rows, "mul(x = gh, y = tp)")
    f16("h", c, rows, "mul(x = silu, y = us)")
    line(f"tensor<fp16, [1, {c}, 1, {rows}]> h4 = reshape(x = h, shape = tensor<int32, [4]>([1, {c}, 1, {rows}]))")
    line(f"tensor<fp16, [{c}, {BLOC_INTER}, 1, 1]> rotation = const()[name = string(\"rotation\"), "
         f"val = tensor<fp16, [{c}, {BLOC_INTER}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights.bin\"), "
         f"offset = uint64(64)))]")
    line(f"tensor<fp16, [1, {c}, 1, {rows}]> hr4 = conv(dilations = tensor<int32, [2]>([1, 1]), "
         f"groups = int32({c // BLOC_INTER}), pad = tensor<int32, [4]>([0, 0, 0, 0]), pad_type = string(\"valid\"), "
         f"strides = tensor<int32, [2]>([1, 1]), weight = rotation, x = h4)")
    f16("hr", c, rows, f"reshape(x = hr4, shape = tensor<int32, [4]>({dims(c, rows)}))")
    f16("habs", c, rows, "abs(x = hr)")
    f16("peak", 1, rows, "reduce_max(x = habs, axes = tensor<int32, [1]>([2]), keep_dims = bool(true))")
    f16("floor", 1, rows, f"maximum(x = peak, y = {fp16(PLANCHER_INTER)})")
    f16("inverse", 1, rows, f"real_div(x = {fp16(PIC_INT8)}, y = floor)")
    f16("hs", c, rows, "mul(x = hr, y = inverse)")
    line(f"{tensor('int8', c, rows)} hq = quantize(input = hs, scale = fp16(1), output_dtype = string(\"int8\"))")
    f16("hd", c, rows, f"dequantize(input = hq, scale = {unit})")
    f16("hscale", 1, rows, f"mul(x = floor, y = {fp16(UNITE_INT8 / PIC_INT8)})")
    debut = 0
    for i, w in enumerate(f.seg_down):
        f16(f"hd{i}", w, rows, f"slice_by_size(x = hd, begin = tensor<int32, [4]>([0, 0, {debut}, 0]), "
                              f"size = tensor<int32, [4]>({dims(w, rows)}))")
        matmul(f"dm{i}", f"wd{i}", f"hd{i}", f.hidden, w)
        debut += w
    f16("ds", f.hidden, rows, f"mul(x = {somme('dm', len(f.seg_down), f.hidden)}, y = sd_t)")
    f16("yt", f.hidden + 1, rows, "concat(axis = int32(2), interleave = bool(false), values = (ds, hscale))")
    line(f"{sortie.type(f.hidden + 1, rows)} y = tensor_to_tensor_buffer<ios17>(input = yt, "
         f"interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>({sortie.strides(f.hidden + 1)}))")
    corps = "".join(f"        {t};\n" for t in body)
    return f"    func ffn{rows}<ios18>({', '.join(params)}) {{\n{corps}    }} -> (y);\n"


def programme_mil(f: Forme, m: Memoire, lignes: list[int]) -> str:
    e = m.entrees(f)
    return "program(1.3)\n{\n" + "".join(fonction(f, e, m.sortie, r) for r in lignes) + "}\n"


# ---- rotations et quantification (côté CPU pour la sonde ; côté GPU dans le moteur) ----

def signes(n: int = BLOC_INTER, graine: int = 20261015) -> np.ndarray:
    return np.where(np.random.RandomState(graine).randint(0, 2, n) == 1, -1.0, 1.0).astype(np.float32)


def hadamard(n: int) -> np.ndarray:
    i = np.arange(n)
    pc = np.vectorize(lambda v: bin(v).count("1") & 1)(i[:, None] & i[None, :])
    return np.where(pc == 1, -1.0, 1.0).astype(np.float32)


def rotation(n: int, sg: np.ndarray) -> np.ndarray:
    """R = D H / sqrt(n), D = diag(premiers n signes) : sortie o = sum_i R[o][i] v[i]."""
    return (sg[:n, None] * hadamard(n)) / np.sqrt(n)


def blob_rotation(canaux: int, sg: np.ndarray) -> bytes:
    r = rotation(BLOC_INTER, sg)  # [o][i]
    v = np.tile(r, (canaux // BLOC_INTER, 1)).reshape(canaux, BLOC_INTER, 1, 1)
    return blob(v.astype(np.float16))


def tourner_entrees(m: np.ndarray, bloc: int, r: np.ndarray) -> np.ndarray:
    """Applique R par blocs de `bloc` sur le dernier axe : v' = R v (lignes de m = vecteurs)."""
    s = m.shape
    return (m.reshape(-1, s[-1] // bloc, bloc) @ r.T).reshape(s)


def int8_lignes(w: np.ndarray):
    """Codes int8 par ligne et échelle de la ligne (pic / 127)."""
    pic = np.maximum(np.abs(w).max(axis=1), PLANCHER_PIC)
    e = pic / PIC_INT8
    q = np.clip(np.rint(w / e[:, None]), -127, 127).astype(np.int8)
    return q, e.astype(np.float32)
