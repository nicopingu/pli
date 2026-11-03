// requant_cpu.c (PLI, 21/10/2026): requantization on the CPU of the weights of the FFN's Neural Engine part.
// Goal: take off the GPU the pli_anerq kernel (5.59 ms per layer and per chunk) and the numpy copy of
// _State.write_weights. The int8 codes and the scales are written directly into the Neural Engine's surfaces, one
// layer ahead, while the GPU and the Neural Engine compute the current layer.
//
// Same arithmetic as ffn.requant_kernel (checked bit for bit against it by bench/rq_check.py):
//   value of a weight: mode & 1 = 0: bf16(bf16(sb * d) + bb) (two roundings); mode & 1 = 1: bf16(fma(sb, d, bb));
//     bench/rq_check.py, 21/10: mode 1 reproduces the Metal compiler bit for bit (100 % of the codes and scales);
//   mode & 2: scalar path forced (trials); otherwise 4 rows at a time in vectors (four_rows), same bits;
//   Hadamard in blocks of BLOCK: stages of steps 32, 64 (, 128, 256), then 1 to 16, butterfly (low + high, low - high);
//   y = v * (SG[o] * norm); int8 per row: e = max(peak, 2^-14) / 127, q = rint(y / e) bounded to 127; E = half(e * 128).
// Compilation: clang -O3 -ffp-contract=off (no contraction into FMA outside mode 1, no approximate computation).
#ifdef __APPLE__
#include <dispatch/dispatch.h>
#endif
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define PLI_REQUANT_NC_MAX 17408

static inline uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static inline float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
// Round to nearest even to bf16; finite values only (a model's weights, scales and biases).
static inline float bf16_r(float f) { const uint32_t u = f2u(f); return u2f((u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u); }
static inline float bf16_f(uint32_t b) { return u2f(b << 16); }
static inline float f16_f(uint16_t h) { _Float16 x; memcpy(&x, &h, 2); return (float)x; }

// sc_kind: 0 bf16 scales and biases per row, 1 fp16 per row, 2 relaid out P2 (uint32: bf16 scale low, bias high)
static inline void read_sb(const void *SC, const void *BI, int sc_kind, size_t G, size_t row, size_t g, float *scf, float *bif) {
  if (sc_kind == 2) {
    const size_t p = row / 32, t = (row / 8) % 4, fm = row % 8;
    const uint32_t sbw = ((const uint32_t *)SC)[((p * G + g) * 8 + fm) * 4 + t];
    *scf = bf16_f(sbw & 0xFFFFu);
    *bif = bf16_f(sbw >> 16);
  } else if (sc_kind == 0) {
    *scf = bf16_f(((const uint16_t *)SC)[row * G + g]);
    *bif = bf16_f(((const uint16_t *)BI)[row * G + g]);
  } else {
    *scf = f16_f(((const uint16_t *)SC)[row * G + g]);
    *bif = f16_f(((const uint16_t *)BI)[row * G + g]);
  }
}

// Word of 8 4-bit weights of row `row`, columns c .. c + 7 (c multiple of 8); relaid out P2 or row by row.
static inline uint32_t read_w(const uint32_t *W, int rg, size_t K, size_t G, size_t row, size_t c) {
  const size_t g = c / 64, k = c % 64, fnh = k / 16, hi = (k & 15) >= 8 ? 1 : 0;
  if (rg) {
    const size_t p = row / 32, t = (row / 8) % 4, fm = row % 8;
    return W[2 * (((p * G + g) * 32 + fm * 4 + fnh) * 4 + t) + hi];
  }
  return W[row * (K / 8) + 8 * g + 2 * fnh + hi];
}

typedef struct {
  const uint32_t *W;
  const void *SC, *BI;
  int sc_kind, rg;
  size_t K;
  int row0, C0, NC, BLOCK;
  const float *SG;
  int seg_n;
  const int *seg_w;
  int8_t *const *seg_ptr;
  const size_t *seg_stride;
  uint16_t *E;
  size_t e_stride;
  int mode;
} args_t;

static inline void butterflies(float *v, int B, int s) {
  for (int base = 0; base < B; base += 2 * s)
    for (int o = base; o < base + s; o++) {
      const float x = v[o], y = v[o + s];
      v[o] = x + y;
      v[o + s] = x - y;
    }
}

static void one_row(const args_t *a, int r, float *vals) {
  const size_t row = (size_t)a->row0 + (size_t)r, G = a->K / 64;
  const int B = a->BLOCK;
  const float norm = (float)(1.0 / sqrt((double)B));
  float peak = 0.0f;
  for (int b = 0; b < a->NC / B; b++) {
    float *v = vals + (size_t)b * B;
    for (int g0 = 0; g0 < B; g0 += 64) {
      const size_t c0 = (size_t)a->C0 + (size_t)(b * B + g0);  // C0 and BLOCK multiples of 64: one scale group
      float scf, bif;
      read_sb(a->SC, a->BI, a->sc_kind, G, row, c0 / 64, &scf, &bif);
      const float sb = bf16_r(scf), bb = bf16_r(bif);
      float d[64];
      for (int k8 = 0; k8 < 64; k8 += 8) {
        const uint32_t w32 = read_w(a->W, a->rg, a->K, G, row, c0 + (size_t)k8);
        for (int k = 0; k < 8; k++) d[k8 + k] = (float)((w32 >> (4 * k)) & 0xFu);
      }
      float *o = v + g0;
      if (a->mode & 1)
        for (int k = 0; k < 64; k++) o[k] = bf16_r(fmaf(sb, d[k], bb));
      else
        for (int k = 0; k < 64; k++) o[k] = bf16_r(bf16_r(sb * d[k]) + bb);
    }
    for (int s = 32; s < B; s <<= 1) butterflies(v, B, s);  // the kernel's "register" stages
    for (int s = 1; s < 32; s <<= 1) butterflies(v, B, s);  // "across threads" stages (simd_shuffle_xor)
    for (int o = 0; o < B; o++) {
      v[o] = v[o] * (a->SG[o] * norm);
      peak = fmaxf(peak, fabsf(v[o]));
    }
  }
  peak = fmaxf(peak, 6.103515625e-05f);
  const float e = peak / 127.0f;
  int start = 0;
  for (int s = 0; s < a->seg_n; s++) {
    int8_t *dst = a->seg_ptr[s] + (size_t)r * a->seg_stride[s];
    const float *src = vals + start;
    for (int c = 0; c < a->seg_w[s]; c++) {
      float q = rintf(src[c] / e);
      q = q < -127.0f ? -127.0f : (q > 127.0f ? 127.0f : q);
      dst[c] = (int8_t)q;
    }
    start += a->seg_w[s];
  }
  const _Float16 eh = (_Float16)(e * 128.0f);
  memcpy(&a->E[(size_t)r * a->e_stride], &eh, 2);
}

// ---- 4 rows at a time, one row per vector lane (clang vectors: NEON on arm64) ----
// Same operations per element as one_row and in the same order (Hadamard stages grouped in registers without
// changing their order): same bits. A vector = the same column of 4 rows, so no mixing between lanes.
typedef float f4 __attribute__((ext_vector_type(4)));
typedef uint32_t u4 __attribute__((ext_vector_type(4)));
typedef int32_t i4 __attribute__((ext_vector_type(4)));
typedef int8_t c4 __attribute__((ext_vector_type(4)));

static inline u4 f4_u4(f4 x) { u4 u; memcpy(&u, &x, 16); return u; }
static inline f4 u4_f4(u4 u) { f4 x; memcpy(&x, &u, 16); return x; }
static inline f4 bf16_r4(f4 x) { const u4 u = f4_u4(x); return u4_f4((u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u); }
static inline f4 splat(float s) { return (f4)(s); }
static inline f4 fma4(f4 a, f4 b, f4 c) {
#if __has_builtin(__builtin_elementwise_fma)
  return __builtin_elementwise_fma(a, b, c);
#else
  return (f4){fmaf(a[0], b[0], c[0]), fmaf(a[1], b[1], c[1]), fmaf(a[2], b[2], c[2]), fmaf(a[3], b[3], c[3])};
#endif
}
static inline f4 rint4(f4 a) {
#if __has_builtin(__builtin_elementwise_roundeven)
  return __builtin_elementwise_roundeven(a);
#else
  return (f4){rintf(a[0]), rintf(a[1]), rintf(a[2]), rintf(a[3])};
#endif
}
static inline f4 max4(f4 a, f4 b) {
#if __has_builtin(__builtin_elementwise_max)
  return __builtin_elementwise_max(a, b);
#else
  return (f4){fmaxf(a[0], b[0]), fmaxf(a[1], b[1]), fmaxf(a[2], b[2]), fmaxf(a[3], b[3])};
#endif
}
static inline f4 min4(f4 a, f4 b) {
#if __has_builtin(__builtin_elementwise_min)
  return __builtin_elementwise_min(a, b);
#else
  return (f4){fminf(a[0], b[0]), fminf(a[1], b[1]), fminf(a[2], b[2]), fminf(a[3], b[3])};
#endif
}
static inline f4 abs4(f4 a) {
#if __has_builtin(__builtin_elementwise_abs)
  return __builtin_elementwise_abs(a);
#else
  return (f4){fabsf(a[0]), fabsf(a[1]), fabsf(a[2]), fabsf(a[3])};
#endif
}

static void dequant4(const args_t *a, const size_t row[4], int b, f4 *v) {
  const size_t G = a->K / 64;
  const int B = a->BLOCK;
  for (int g0 = 0; g0 < B; g0 += 64) {
    const size_t c0 = (size_t)a->C0 + (size_t)(b * B + g0);
    f4 sb, bb;
    for (int l = 0; l < 4; l++) {
      float scf, bif;
      read_sb(a->SC, a->BI, a->sc_kind, G, row[l], c0 / 64, &scf, &bif);
      sb[l] = bf16_r(scf);
      bb[l] = bf16_r(bif);
    }
    for (int k8 = 0; k8 < 64; k8 += 8) {
      u4 w;
      for (int l = 0; l < 4; l++) w[l] = read_w(a->W, a->rg, a->K, G, row[l], c0 + (size_t)k8);
      for (int k = 0; k < 8; k++) {
        const f4 d = __builtin_convertvector((w >> (u4)(4u * (uint32_t)k)) & 0xFu, f4);
        v[g0 + k8 + k] = (a->mode & 1) ? bf16_r4(fma4(sb, d, bb)) : bf16_r4(bf16_r4(sb * d) + bb);
      }
    }
  }
}

// The kernel's "register" stages (columns o + 32 j, steps of j 1, 2, 4, 8), J = BLOCK / 32 <= 16 vectors in registers.
static inline __attribute__((always_inline)) void register_stages_J(f4 *v, const int J) {
  for (int o = 0; o < 32; o++) {
    f4 r[16];
    for (int j = 0; j < J; j++) r[j] = v[o + 32 * j];
    for (int s = 1; s < J; s <<= 1)
      for (int j = 0; j < J; j++)
        if (!(j & s)) {
          const f4 x = r[j], y = r[j + s];
          r[j] = x + y;
          r[j + s] = x - y;
        }
    for (int j = 0; j < J; j++) v[o + 32 * j] = r[j];
  }
}
static void register_stages(f4 *v, int B) {
  if (B == 128) register_stages_J(v, 4);
  else if (B == 512) register_stages_J(v, 16);
  else register_stages_J(v, B / 32);
}

// "Across threads" stages of steps 1, 2, 4, 8, by groups of 16 consecutive columns in registers.
static void thread_stages(f4 *v, int B) {
  for (int base = 0; base < B; base += 16) {
    f4 c[16];
    for (int i = 0; i < 16; i++) c[i] = v[base + i];
    for (int h = 1; h < 16; h <<= 1)
      for (int i = 0; i < 16; i++)
        if (!(i & h)) {
          const f4 x = c[i], y = c[i + h];
          c[i] = x + y;
          c[i + h] = x - y;
        }
    for (int i = 0; i < 16; i++) v[base + i] = c[i];
  }
}

// Last stage (step 16), then y = v * (SG[o] * norm) and the peak per row.
static f4 stage16_scale(f4 *v, int B, const float *SG, float norm, f4 peak) {
  for (int base = 0; base < B; base += 32)
    for (int i = 0; i < 16; i++) {
      const f4 x = v[base + i], y = v[base + i + 16];
      const float m0 = SG[base + i] * norm, m1 = SG[base + i + 16] * norm;
      const f4 p = (x + y) * splat(m0), q = (x - y) * splat(m1);
      v[base + i] = p;
      v[base + i + 16] = q;
      peak = max4(peak, max4(abs4(p), abs4(q)));
    }
  return peak;
}

static void four_rows(const args_t *a, int r, f4 *V) {
  size_t row[4];
  for (int l = 0; l < 4; l++) row[l] = (size_t)a->row0 + (size_t)(r + l);
  const int B = a->BLOCK;
  const float norm = (float)(1.0 / sqrt((double)B));
  f4 peak = splat(0.0f);
  for (int b = 0; b < a->NC / B; b++) {
    f4 *v = V + (size_t)b * B;
    dequant4(a, row, b, v);
    register_stages(v, B);
    thread_stages(v, B);
    peak = stage16_scale(v, B, a->SG, norm, peak);
  }
  peak = max4(peak, splat(6.103515625e-05f));
  const f4 e = peak / splat(127.0f);
  int start = 0;
  for (int s = 0; s < a->seg_n; s++) {
    const int w = a->seg_w[s];
    int8_t *dst[4];
    for (int l = 0; l < 4; l++) dst[l] = a->seg_ptr[s] + (size_t)(r + l) * a->seg_stride[s];
    const f4 *src = V + start;
    int c = 0;
    for (; c + 4 <= w; c += 4) {
      i4 q[4];
      for (int t = 0; t < 4; t++)
        q[t] = __builtin_convertvector(min4(max4(rint4(src[c + t] / e), splat(-127.0f)), splat(127.0f)), i4);
      for (int l = 0; l < 4; l++) {  // transposition: column -> row
        const c4 packed = {(int8_t)q[0][l], (int8_t)q[1][l], (int8_t)q[2][l], (int8_t)q[3][l]};
        memcpy(dst[l] + c, &packed, 4);
      }
    }
    for (; c < w; c++)
      for (int l = 0; l < 4; l++) {
        float q = rintf(src[c][l] / e[l]);
        q = q < -127.0f ? -127.0f : (q > 127.0f ? 127.0f : q);
        dst[l][c] = (int8_t)q;
      }
    start += w;
  }
  for (int l = 0; l < 4; l++) {
    const _Float16 eh = (_Float16)(e[l] * 128.0f);
    memcpy(&a->E[(size_t)(r + l) * a->e_stride], &eh, 2);
  }
}

// Rows [start, end): by 4 on the vector path (except mode & 2: scalar forced, for trials), the rest in scalar.
// V: NC work vectors (16-byte aligned), owned by the caller.
static void do_rows(const args_t *a, int start, int end, f4 *V) {
  int r = start;
  if (!(a->mode & 2) && a->BLOCK <= 512)
    for (; r + 4 <= end; r += 4) four_rows(a, r, V);
  for (; r < end; r++) one_row(a, r, (float *)V);
}

int pli_rq_version(void) { return 2; }

// Rows [row0, row0 + nrows) of a 4-bit matrix of K columns, columns [C0, C0 + NC). Output of row r (relative):
// columns cut into seg_n segments of widths seg_w (sum = NC), to seg_ptr[s] + r * seg_stride[s] (bytes);
// scale to E + r * e_stride (fp16 elements). Rows spread into `threads` contiguous parts (GCD).
// Returns 0; -1 parameters out of bounds; -2 segments whose sum is not NC.
int pli_rq_cpu(const uint32_t *W, const void *SC, const void *BI, int sc_kind, int rg, size_t K, int row0, int nrows,
               int C0, int NC, int BLOCK, const float *SG, int seg_n, const int *seg_w, int8_t *const *seg_ptr,
               const size_t *seg_stride, uint16_t *E, size_t e_stride, int mode, int threads) {
  if (BLOCK < 64 || (BLOCK & (BLOCK - 1)) || NC <= 0 || NC > PLI_REQUANT_NC_MAX || NC % BLOCK || C0 % 64 || nrows <= 0 ||
      K % 64 || mode < 0 || mode > 3 || sc_kind < 0 || sc_kind > 2 || seg_n <= 0)
    return -1;
  int total = 0;
  for (int s = 0; s < seg_n; s++) total += seg_w[s];
  if (total != NC) return -2;
  if (threads < 1) threads = 1;
  const int per = ((nrows + threads - 1) / threads + 3) & ~3;  // row parts multiples of 4
  threads = (nrows + per - 1) / per;
  f4 *work = (f4 *)malloc((size_t)threads * (size_t)NC * sizeof(f4));
  if (!work) return -3;
  const args_t a = {.W = W, .SC = SC, .BI = BI, .sc_kind = sc_kind, .rg = rg, .K = K, .row0 = row0, .C0 = C0, .NC = NC,
                    .BLOCK = BLOCK, .SG = SG, .seg_n = seg_n, .seg_w = seg_w, .seg_ptr = seg_ptr, .seg_stride = seg_stride,
                    .E = E, .e_stride = e_stride, .mode = mode};
#ifdef __APPLE__
  const args_t *pa = &a;
  dispatch_apply((size_t)threads, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t f) {
    const int start = (int)f * per, end = start + per < nrows ? start + per : nrows;
    do_rows(pa, start, end, work + f * (size_t)NC);
  });
#else  // trials outside macOS: the parts one after the other
  for (int f = 0; f < threads; f++) {
    const int start = f * per, end = start + per < nrows ? start + per : nrows;
    do_rows(&a, start, end, work + (size_t)f * (size_t)NC);
  }
#endif
  free(work);
  return 0;
}
