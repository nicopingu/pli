"""PLI (lot E4) : le MLP en entiers d'abord, du début à la fin, avec les mêmes bits sur le GPU et sur le CPU.

Chaîne, pour chaque ligne (les lignes sont indépendantes : le résultat ne dépend pas du nombre de lignes de l'appel) :
  1. passage aux entiers : xi = arrondi(x * 2^-e), e pris sur les bits du plus grand |x| de la ligne (xi tient en 15 bits signés) ;
  2. gate et up : produits en entiers (simd_qmm.qmm_ent sur le GPU, pli_qmm_e15 sur le CPU), sortie en 32 bits ;
  3. activation : gate repassé en entiers (gi, exposant k), puis lecture d'une table commune T[k][gi] = silu(gi * 2^k) * 2^-k ;
     a = T * up (une multiplication) ;
  4. a repassé en entiers, produit down en entiers, puis remise à l'échelle par une puissance de 2 (exacte).
Seules opérations flottantes hors noyaux : maximum, multiplication par une puissance de 2 (exacte), x + 0,5 puis partie entière
(l'arrondi), une multiplication, et la lecture de la table. Chacune rend le même résultat sur tout organe conforme à IEEE 754.
Rien n'est branché dans le moteur : ce module sert aux essais de justesse."""
import numpy as np

ENT_MAX = 16383
K_MIN, K_MAX = -30, 4            # exposants tabulés ; en dessous : silu(v) = v / 2 ; au-dessus : silu(v) = max(v, 0)
E_LIM = 100
_table = None


def table():
    """Table commune [K_MAX - K_MIN + 3, 2 * ENT_MAX + 1] en 32 bits : ligne 0 = moitié, lignes 1.. = k de K_MIN à K_MAX, dernière = max(gi, 0)."""
    global _table
    if _table is None:
        gi = np.arange(-ENT_MAX, ENT_MAX + 1, dtype=np.float64)
        lignes = [gi / 2]
        for k in range(K_MIN, K_MAX + 1):
            v = np.ldexp(gi, k)
            with np.errstate(over="ignore"):
                lignes.append(v / (1.0 + np.exp(-v)) / (2.0 ** k))
        lignes.append(np.maximum(gi, 0.0))
        _table = np.ascontiguousarray(np.stack(lignes), dtype=np.float32)
    return _table


# ---------- CPU (numpy + noyau pli_qmm_e15) ----------
def entiers_np(x):
    """x [L, K] float32 -> (xi int16, e int32 [L]) : xi = partie entière(x * 2^-e + 0,5), bornée à +- ENT_MAX."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    m = np.abs(x).max(axis=1)
    e = (((m.view(np.uint32) >> 23) & 0xFF).astype(np.int32) - 127) - 13
    e = np.clip(e, -E_LIM, E_LIM)
    ech = (((127 - e).astype(np.uint32)) << 23).view(np.float32)
    v = np.floor(x * ech[:, None] + np.float32(0.5))
    return np.clip(v, -ENT_MAX, ENT_MAX).astype(np.int16), e


def pow2_np(e):
    return ((127 + np.clip(e, -126, 127)).astype(np.uint32) << 23).view(np.float32)


def mlp_np(x, poids_gate, poids_up, poids_down):
    from tensorfold.kernels.pli_cpu import qmm_cpu
    T = table()
    xi, e = entiers_np(x)
    g = qmm_cpu.produit_e15(poids_gate, xi)
    u = qmm_cpu.produit_e15(poids_up, xi)
    gi, eg = entiers_np(g)
    k = eg + e
    lig = np.clip(k - K_MIN + 1, 0, K_MAX - K_MIN + 2)
    a = T[lig[:, None], gi.astype(np.int32) + ENT_MAX] * u
    ai, ea = entiers_np(a)
    y = qmm_cpu.produit_e15(poids_down, ai)
    return y * pow2_np(ea + k + e)[:, None]


# ---------- GPU (MLX + noyau de TensorFold à entrée entière) ----------
def entiers_mx(x):
    import mlx.core as mx
    x = x.astype(mx.float32)
    m = mx.max(mx.abs(x), axis=1)
    e = ((m.view(mx.uint32) >> 23) & 0xFF).astype(mx.int32) - 127 - 13
    e = mx.clip(e, -E_LIM, E_LIM)
    ech = ((127 - e).astype(mx.uint32) << 23).view(mx.float32)
    v = mx.floor(x * ech[:, None] + mx.array(0.5, dtype=mx.float32))
    return mx.clip(v, -ENT_MAX, ENT_MAX).astype(mx.int16), e


def mlp_mx(x, gate, up, down, T=None):
    """gate, up, down : triplets (weight, scales, biases) du format 4 bits de MLX. Rend y [L, sorties] en 32 bits."""
    import mlx.core as mx
    from tensorfold.kernels.qwen.dense.v1 import simd_qmm
    T = mx.array(table()) if T is None else T
    xi, e = entiers_mx(x)
    g = simd_qmm.qmm_ent(xi, *gate)
    u = simd_qmm.qmm_ent(xi, *up)
    gi, eg = entiers_mx(g)
    k = eg + e
    lig = mx.clip(k - K_MIN + 1, 0, K_MAX - K_MIN + 2)
    plat = lig[:, None] * (2 * ENT_MAX + 1) + gi.astype(mx.int32) + ENT_MAX
    a = mx.take(T.reshape(-1), plat) * u
    ai, ea = entiers_mx(a)
    y = simd_qmm.qmm_ent(ai, *down)
    ef = mx.clip(ea + k + e, -126, 127)
    return y * ((127 + ef).astype(mx.uint32) << 23).view(mx.float32)[:, None]


# ---------- GPU, version à noyaux (banc pli-e15c : les petites opérations de MLX coûtaient 0,4 ms par passage aux entiers) ----------
# Deux noyaux Metal font le même calcul que entiers_mx et que l'activation par table, en un lancement chacun :
#   pli_ent_q   : passage aux entiers d'un tableau [L, K] (une ligne par groupe de fils) ;
#   pli_ent_act : gate repassé en entiers, table, multiplication par up, résultat repassé en entiers.
# Mêmes opérations, dans le même ordre, que la version numpy (mlp_np) : mêmes bits.
TG = 256
_ENTETE = r"""
inline int pli_exposant(float m) { return clamp(int((as_type<uint>(m) >> 23) & 0xFFu) - 140, -100, 100); }
inline float pli_echelle(int e) { return as_type<float>(uint(127 - e) << 23); }
inline short pli_arrondi(float v) { return short(clamp(floor(v + 0.5f), -16383.0f, 16383.0f)); }
"""
_SRC_Q = r"""
  constexpr uint TG = 256;
  const uint r = threadgroup_position_in_grid.x;
  const uint t = uint(simdgroup_index_in_threadgroup) * 32u + uint(thread_index_in_simdgroup);
  const uint K = uint(X_shape[1]);
  threadgroup float grands[TG];
  float m = 0.0f;
  for (uint j = t; j < K; j += TG) m = max(m, fabs(float(X[size_t(r) * K + j])));
  grands[t] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float mm = 0.0f;
  for (uint i = 0; i < TG; i++) mm = max(mm, grands[i]);
  const int e = pli_exposant(mm);
  const float ech = pli_echelle(e);
  for (uint j = t; j < K; j += TG) XI[size_t(r) * K + j] = pli_arrondi(float(X[size_t(r) * K + j]) * ech);
  if (t == 0) E[r] = e;
"""
_SRC_ACT = r"""
  constexpr uint TG = 256;
  const uint r = threadgroup_position_in_grid.x;
  const uint t = uint(simdgroup_index_in_threadgroup) * 32u + uint(thread_index_in_simdgroup);
  const uint N = uint(G_shape[1]);
  const int kmin = int(P[0]), lignes = int(P[1]), large = int(P[2]);
  threadgroup float grands[TG];
  float m = 0.0f;
  for (uint j = t; j < N; j += TG) m = max(m, fabs(G[size_t(r) * N + j]));
  grands[t] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float mm = 0.0f;
  for (uint i = 0; i < TG; i++) mm = max(mm, grands[i]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int eg = pli_exposant(mm);
  const float echg = pli_echelle(eg);
  const int k = eg + EX[r];
  const size_t base = size_t(clamp(k - kmin + 1, 0, lignes - 1)) * size_t(large) + size_t(large / 2);
  m = 0.0f;
  for (uint j = t; j < N; j += TG) {
    const float a = T[base + int(pli_arrondi(G[size_t(r) * N + j] * echg))] * U[size_t(r) * N + j];
    m = max(m, fabs(a));
  }
  grands[t] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  mm = 0.0f;
  for (uint i = 0; i < TG; i++) mm = max(mm, grands[i]);
  const int ea = pli_exposant(mm);
  const float echa = pli_echelle(ea);
  for (uint j = t; j < N; j += TG) {
    const float a = T[base + int(pli_arrondi(G[size_t(r) * N + j] * echg))] * U[size_t(r) * N + j];
    AI[size_t(r) * N + j] = pli_arrondi(a * echa);
  }
  if (t == 0) EF[r] = ea + k + EX[r];
"""
_noyaux: dict = {}


def _noyau(nom):
    import mlx.core as mx
    if nom not in _noyaux:
        if nom == "q":
            _noyaux[nom] = mx.fast.metal_kernel(name="pli_ent_q", input_names=["X"], output_names=["XI", "E"], source=_SRC_Q, header=_ENTETE)
        else:
            _noyaux[nom] = mx.fast.metal_kernel(name="pli_ent_act", input_names=["G", "U", "EX", "T", "P"], output_names=["AI", "EF"],
                                                source=_SRC_ACT, header=_ENTETE)
    return _noyaux[nom]


def entiers_noyau(x):
    """Comme entiers_mx, en un lancement. x [L, K] en bf16 ou en 32 bits."""
    import mlx.core as mx
    L, K = int(x.shape[0]), int(x.shape[1])
    xi, e = _noyau("q")(inputs=[x], grid=(TG * L, 1, 1), threadgroup=(TG, 1, 1), output_shapes=[(L, K), (L,)], output_dtypes=[mx.int16, mx.int32])
    return xi, e


_params = None


def mlp_noyaux(x, gate, up, down, T=None):
    """Comme mlp_mx, avec les deux noyaux : 5 lancements (entiers, gate, up, activation, down) et une remise à l'échelle."""
    import mlx.core as mx
    from tensorfold.kernels.qwen.dense.v1 import simd_qmm
    global _params
    T = mx.array(table()) if T is None else T
    if _params is None:
        _params = mx.array([K_MIN, K_MAX - K_MIN + 3, 2 * ENT_MAX + 1], dtype=mx.int32)
    L = int(x.shape[0])
    xi, e = entiers_noyau(x)
    g = simd_qmm.qmm_ent(xi, *gate)
    u = simd_qmm.qmm_ent(xi, *up)
    N = int(g.shape[1])
    ai, ef = _noyau("act")(inputs=[g, u, e, T.reshape(-1), _params], grid=(TG * L, 1, 1), threadgroup=(TG, 1, 1),
                           output_shapes=[(L, N), (L,)], output_dtypes=[mx.int16, mx.int32])
    y = simd_qmm.qmm_ent(ai, *down)
    return y * ((127 + mx.clip(ef, -126, 127)).astype(mx.uint32) << 23).view(mx.float32)[:, None]
