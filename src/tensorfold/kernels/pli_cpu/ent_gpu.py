"""PLI, lot E (entiers d'abord), côté GPU : le même calcul que pli_qmm_ent (qmm_cpu.c), écrit en Metal par mx.fast.metal_kernel.

Version simple (un fil par sortie, non optimisée) : elle sert à vérifier que le GPU et le CPU rendent les mêmes bits.
Sommes d'entiers par paquet de 64 colonnes (exactes), puis l'étape flottante dans l'ordre fixé par qmm_cpu.c :
    acc = acc + ( ((s * kh) * Ph + s * Pl) + ((b * kh) * SXh + b * SXl) ) ;  y = acc * ech
Rien n'est branché dans le moteur.
"""
import numpy as np
import mlx.core as mx

_SOURCE = """
  uint n = thread_position_in_grid.x;
  uint l = thread_position_in_grid.y;
  float acc = 0.0f;
  for (int g = 0; g < G; g++) {
    int ph = 0, pl = 0;
    for (int m = 0; m < 8; m++) {
      uint mot = W[n * uint(K / 8) + uint(g * 8 + m)];
      for (int j = 0; j < 8; j++) {
        int q = int((mot >> uint(4 * j)) & 15u);
        uint c = l * uint(K) + uint(g * 64 + m * 8 + j);
        ph += q * XH[c];
        pl += q * XL[c];
      }
    }
    float s = SC[n * uint(G) + uint(g)], b = BI[n * uint(G) + uint(g)];
    float a1 = (s * KH) * float(ph);
    float a2 = s * float(pl);
    float u = a1 + a2;
    float c1 = (b * KH) * SXH[uint(g) * uint(LP) + l];
    float c2 = b * SXL[uint(g) * uint(LP) + l];
    float v = c1 + c2;
    float t = u + v;
    acc = acc + t;
  }
  OUT[l * uint(N) + n] = acc * ECH[l];
"""
_noyaux = {}


def produit_entiers_gpu(w, s32, b32, xh, xl, ech):
    """w : mx uint32 [N, K/8] ; s32, b32 : mx float32 [N, K/64] ; xh, xl, ech : comme qmm_cpu.entiers. Rend un mx float32 [L, N]."""
    N, K = int(w.shape[0]), int(w.shape[1]) * 8; G = K // 64; L = int(xh.shape[0]); LP = (L + 3) & ~3
    kh = 1.0 if xl is None else 256.0
    h = xh.astype(np.int32); lo = np.zeros_like(h) if xl is None else xl.astype(np.int32)
    sxh = np.zeros((G, LP), np.float32); sxl = np.zeros((G, LP), np.float32)
    sxh[:, :L] = h.reshape(L, G, 64).sum(-1).T; sxl[:, :L] = lo.reshape(L, G, 64).sum(-1).T
    cle = (N, K, LP, kh)
    if cle not in _noyaux:
        consts = "".join("  constexpr int %s = %d;\n" % c for c in (("N", N), ("K", K), ("G", G), ("LP", LP))) + "  constexpr float KH = %s;\n" % ("256.0f" if kh == 256.0 else "1.0f")
        _noyaux[cle] = mx.fast.metal_kernel(name="pli_ent_%d_%d_%d_%d" % (N, K, LP, int(kh)), input_names=["W", "SC", "BI", "XH", "XL", "SXH", "SXL", "ECH"],
                                            output_names=["OUT"], source=consts + _SOURCE)
    return _noyaux[cle](inputs=[w, s32, b32, mx.array(h), mx.array(lo), mx.array(sxh), mx.array(sxl), mx.array(ech.astype(np.float32))],
                        grid=(N, L, 1), threadgroup=(32, 1, 1), output_shapes=[(L, N)], output_dtypes=[mx.float32])[0]
