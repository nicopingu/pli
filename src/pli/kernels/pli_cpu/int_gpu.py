"""PLI, item E (integers first), GPU side: the same computation as pli_qmm_ent (qmm_cpu.c), written in Metal with mx.fast.metal_kernel.

Simple version (one thread per output, not optimized): it serves to check that the GPU and the CPU give the same bits.
Integer sums per pack of 64 columns (exact), then the floating-point step in the order set by qmm_cpu.c:
    acc = acc + ( ((s * kh) * Ph + s * Pl) + ((b * kh) * SXh + b * SXl) );  y = acc * scale
Nothing is wired into the engine.
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
      uint word = W[n * uint(K / 8) + uint(g * 8 + m)];
      for (int j = 0; j < 8; j++) {
        int q = int((word >> uint(4 * j)) & 15u);
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
  OUT[l * uint(N) + n] = acc * SCALE[l];
"""
_kernels = {}


def int_product_gpu(w, s32, b32, xh, xl, scale):
    """w: mx uint32 [N, K/8]; s32, b32: mx float32 [N, K/64]; xh, xl, scale: as qmm_cpu.to_ints. Returns an mx float32 [L, N]."""
    N, K = int(w.shape[0]), int(w.shape[1]) * 8; G = K // 64; L = int(xh.shape[0]); LP = (L + 3) & ~3
    kh = 1.0 if xl is None else 256.0
    h = xh.astype(np.int32); lo = np.zeros_like(h) if xl is None else xl.astype(np.int32)
    sxh = np.zeros((G, LP), np.float32); sxl = np.zeros((G, LP), np.float32)
    sxh[:, :L] = h.reshape(L, G, 64).sum(-1).T; sxl[:, :L] = lo.reshape(L, G, 64).sum(-1).T
    key = (N, K, LP, kh)
    if key not in _kernels:
        consts = "".join("  constexpr int %s = %d;\n" % c for c in (("N", N), ("K", K), ("G", G), ("LP", LP))) + "  constexpr float KH = %s;\n" % ("256.0f" if kh == 256.0 else "1.0f")
        _kernels[key] = mx.fast.metal_kernel(name="pli_int_%d_%d_%d_%d" % (N, K, LP, int(kh)), input_names=["W", "SC", "BI", "XH", "XL", "SXH", "SXL", "SCALE"],
                                            output_names=["OUT"], source=consts + _SOURCE)
    return _kernels[key](inputs=[w, s32, b32, mx.array(h), mx.array(lo), mx.array(sxh), mx.array(sxl), mx.array(scale.astype(np.float32))],
                        grid=(N, L, 1), threadgroup=(32, 1, 1), output_shapes=[(L, N)], output_dtypes=[mx.float32])[0]
