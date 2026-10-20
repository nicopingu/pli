/* qmm_cpu.c : PLI, noyau CPU du produit y = x . W^T, W lu dans le format 4 bits de MLX, sans copie des poids.
 *
 * Format lu (mlx/ops.cpp, v0.32.3) : weight = mots de 32 bits, 8 valeurs de 4 bits par mot, poids faible d'abord ;
 * scales et biases = une valeur par ligne de W et par groupe de 64 colonnes ; poids = valeur * echelle + decalage.
 * W : N lignes (sorties), K colonnes (entrees), K multiple de 64.
 * Trois variantes, meme interface : x [L, K], y [L, n1 - n0] en 32 bits, lignes de W n0 a n1.
 *   pli_qmm_f32  : NEON 32 bits, poids deplies au vol dans les registres.
 *   pli_qmm_f16  : NEON 16 bits (x en float16), somme par groupe en 16 bits puis cumul en 32 bits.
 *   pli_qmm_blas : poids deplies par blocs de lignes dans un tampon, puis sgemm d'Accelerate.
 * sb : type des echelles et decalages (0 = BF16, 1 = float16, 2 = float32).
 * fils : nombre de parts de travail (0 = parts de 64 lignes de W, reparties par le systeme).
 * Compilation : clang -O3 -mcpu=apple-m1 -shared -fblocks qmm_cpu.c -framework Accelerate -o libpliqmm.dylib
 */
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <dispatch/dispatch.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define BL 16 /* lignes de x traitees ensemble */

static inline float lire_sb(const void *p, long i, int sb) {
    if (sb == 2) return ((const float *)p)[i];
    if (sb == 1) return (float)((const float16_t *)p)[i];
    uint32_t u = (uint32_t)((const uint16_t *)p)[i] << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

/* 32 octets = 64 valeurs de 4 bits -> 4 registres de 16 octets, dans l'ordre des colonnes */
static inline void deballer(const uint8_t *p, uint8x16_t q[4]) {
    const uint8x16_t m = vdupq_n_u8(15);
    uint8x16_t a = vld1q_u8(p), c = vld1q_u8(p + 16);
    uint8x16_t ae = vandq_u8(a, m), ao = vshrq_n_u8(a, 4), ce = vandq_u8(c, m), co = vshrq_n_u8(c, 4);
    q[0] = vzip1q_u8(ae, ao);
    q[1] = vzip2q_u8(ae, ao);
    q[2] = vzip1q_u8(ce, co);
    q[3] = vzip2q_u8(ce, co);
}

static inline void groupe_f32(const uint8_t *p, float s, float b, float32x4_t wv[16]) {
    uint8x16_t q[4];
    deballer(p, q);
    const float32x4_t vs = vdupq_n_f32(s), vb = vdupq_n_f32(b);
    for (int i = 0; i < 4; i++) {
        uint16x8_t lo = vmovl_u8(vget_low_u8(q[i])), hi = vmovl_high_u8(q[i]);
        wv[4 * i + 0] = vfmaq_f32(vb, vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo))), vs);
        wv[4 * i + 1] = vfmaq_f32(vb, vcvtq_f32_u32(vmovl_high_u16(lo)), vs);
        wv[4 * i + 2] = vfmaq_f32(vb, vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi))), vs);
        wv[4 * i + 3] = vfmaq_f32(vb, vcvtq_f32_u32(vmovl_high_u16(hi)), vs);
    }
}

static inline void groupe_f16(const uint8_t *p, float s, float b, float16x8_t wv[8]) {
    uint8x16_t q[4];
    deballer(p, q);
    const float16x8_t vs = vdupq_n_f16((float16_t)s), vb = vdupq_n_f16((float16_t)b);
    for (int i = 0; i < 4; i++) {
        wv[2 * i + 0] = vfmaq_f16(vb, vcvtq_f16_u16(vmovl_u8(vget_low_u8(q[i]))), vs);
        wv[2 * i + 1] = vfmaq_f16(vb, vcvtq_f16_u16(vmovl_high_u8(q[i])), vs);
    }
}

/* Priorite des fils du noyau : 0 = haute (defaut), 1 = moyenne, 2 = de fond (coeurs economes, ne gene pas le fil qui pilote le GPU). */
static int pli_qos = 0;
void pli_qmm_priorite(int niveau) { pli_qos = niveau; }
static dispatch_queue_t file_de_travail(void) {
    return dispatch_get_global_queue(pli_qos == 2 ? QOS_CLASS_BACKGROUND : pli_qos == 1 ? QOS_CLASS_UTILITY : QOS_CLASS_USER_INTERACTIVE, 0);
}

static void parts(long n0, long n1, int fils, void (^corps)(long a, long b)) {
    long total = n1 - n0;
    if (total <= 0) return;
    long pas = fils > 0 ? (total + fils - 1) / fils : 64;
    if (pas < 1) pas = 1;
    size_t nb = (size_t)((total + pas - 1) / pas);
    if (nb == 1) {
        corps(n0, n1);
        return;
    }
    dispatch_apply(nb, file_de_travail(), ^(size_t i) {
        long a = n0 + (long)i * pas, b = a + pas;
        corps(a, b > n1 ? n1 : b);
    });
}

int pli_qmm_f32(const uint32_t *w, const void *s, const void *b, int sb, long N, long K, const float *x, long L,
                float *y, long n0, long n1, int fils) {
    if (K % 64 || n0 < 0 || n1 > N || n0 > n1 || L < 1) return -1;
    const long G = K / 64, M = n1 - n0;
    parts(n0, n1, fils, ^(long a, long bb) {
        float32x4_t wv[16], acc[BL][4];
        for (long l0 = 0; l0 < L; l0 += BL) {
            const long nl = L - l0 < BL ? L - l0 : BL;
            for (long n = a; n < bb; n++) {
                const uint8_t *pw = (const uint8_t *)(w + n * (K / 8));
                for (long l = 0; l < nl; l++)
                    for (int j = 0; j < 4; j++) acc[l][j] = vdupq_n_f32(0.0f);
                for (long g = 0; g < G; g++) {
                    groupe_f32(pw + 32 * g, lire_sb(s, n * G + g, sb), lire_sb(b, n * G + g, sb), wv);
                    for (long l = 0; l < nl; l++) {
                        const float *px = x + (l0 + l) * K + 64 * g;
                        float32x4_t a0 = acc[l][0], a1 = acc[l][1], a2 = acc[l][2], a3 = acc[l][3];
                        for (int v = 0; v < 16; v += 4) {
                            a0 = vfmaq_f32(a0, wv[v + 0], vld1q_f32(px + 4 * v));
                            a1 = vfmaq_f32(a1, wv[v + 1], vld1q_f32(px + 4 * v + 4));
                            a2 = vfmaq_f32(a2, wv[v + 2], vld1q_f32(px + 4 * v + 8));
                            a3 = vfmaq_f32(a3, wv[v + 3], vld1q_f32(px + 4 * v + 12));
                        }
                        acc[l][0] = a0, acc[l][1] = a1, acc[l][2] = a2, acc[l][3] = a3;
                    }
                }
                for (long l = 0; l < nl; l++)
                    y[(l0 + l) * M + (n - n0)] =
                        vaddvq_f32(vaddq_f32(vaddq_f32(acc[l][0], acc[l][1]), vaddq_f32(acc[l][2], acc[l][3])));
            }
        }
    });
    return 0;
}

int pli_qmm_f16(const uint32_t *w, const void *s, const void *b, int sb, long N, long K, const float16_t *x, long L,
                float *y, long n0, long n1, int fils) {
    if (K % 64 || n0 < 0 || n1 > N || n0 > n1 || L < 1) return -1;
    const long G = K / 64, M = n1 - n0;
    parts(n0, n1, fils, ^(long a, long bb) {
        float16x8_t wv[8];
        float32x4_t acc[BL];
        for (long l0 = 0; l0 < L; l0 += BL) {
            const long nl = L - l0 < BL ? L - l0 : BL;
            for (long n = a; n < bb; n++) {
                const uint8_t *pw = (const uint8_t *)(w + n * (K / 8));
                for (long l = 0; l < nl; l++) acc[l] = vdupq_n_f32(0.0f);
                for (long g = 0; g < G; g++) {
                    groupe_f16(pw + 32 * g, lire_sb(s, n * G + g, sb), lire_sb(b, n * G + g, sb), wv);
                    for (long l = 0; l < nl; l++) {
                        const float16_t *px = x + (l0 + l) * K + 64 * g;
                        float16x8_t a0 = vmulq_f16(wv[0], vld1q_f16(px)), a1 = vmulq_f16(wv[1], vld1q_f16(px + 8));
                        a0 = vfmaq_f16(a0, wv[2], vld1q_f16(px + 16));
                        a1 = vfmaq_f16(a1, wv[3], vld1q_f16(px + 24));
                        a0 = vfmaq_f16(a0, wv[4], vld1q_f16(px + 32));
                        a1 = vfmaq_f16(a1, wv[5], vld1q_f16(px + 40));
                        a0 = vfmaq_f16(a0, wv[6], vld1q_f16(px + 48));
                        a1 = vfmaq_f16(a1, wv[7], vld1q_f16(px + 56));
                        a0 = vaddq_f16(a0, a1);
                        acc[l] = vaddq_f32(acc[l], vaddq_f32(vcvt_f32_f16(vget_low_f16(a0)), vcvt_high_f32_f16(a0)));
                    }
                }
                for (long l = 0; l < nl; l++) y[(l0 + l) * M + (n - n0)] = vaddvq_f32(acc[l]);
            }
        }
    });
    return 0;
}

/* rr : lignes de W depliees par bloc (tampon de rr * K * 4 octets, rendu a la fin) */
int pli_qmm_blas(const uint32_t *w, const void *s, const void *b, int sb, long N, long K, const float *x, long L,
                 float *y, long n0, long n1, long rr) {
    if (K % 64 || n0 < 0 || n1 > N || n0 > n1 || L < 1 || rr < 1) return -1;
    const long G = K / 64, M = n1 - n0;
    float *buf = NULL;
    if (posix_memalign((void **)&buf, 16384, (size_t)rr * (size_t)K * sizeof(float))) return -2;
    for (long r0 = n0; r0 < n1; r0 += rr) {
        const long nr = n1 - r0 < rr ? n1 - r0 : rr;
        parts(r0, r0 + nr, 0, ^(long a, long bb) {
            float32x4_t wv[16];
            for (long n = a; n < bb; n++) {
                const uint8_t *pw = (const uint8_t *)(w + n * (K / 8));
                float *d = buf + (n - r0) * K;
                for (long g = 0; g < G; g++) {
                    groupe_f32(pw + 32 * g, lire_sb(s, n * G + g, sb), lire_sb(b, n * G + g, sb), wv);
                    for (int v = 0; v < 16; v++) vst1q_f32(d + 64 * g + 4 * v, wv[v]);
                }
            }
        });
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)L, (int)nr, (int)K, 1.0f, x, (int)K, buf, (int)K, 0.0f,
                    y + (r0 - n0), (int)M);
    }
    free(buf);
    return 0;
}

/* Lot E (entiers d'abord). Activations en entiers : xh et xl sont les deux moities centrees (-128..127) d'un entier de 16 bits
 * (valeur = 256 * xh + xl) ; xl == NULL : activations de 8 bits (valeur = xh). ech : une echelle par ligne de x.
 * Par paquet de 64 colonnes, les sommes Ph = q.xh et Pl = q.xl sont des entiers exacts (instruction SDOT), comme les sommes
 * SXh et SXl des activations. L'etape flottante est la meme partout, dans cet ordre, sans multiplication-addition fusionnee,
 * en flottants 32 bits, avec kh = 256 (16 bits) ou 1 (8 bits) :
 *     acc = acc + ( ((s * kh) * Ph + s * Pl) + ((b * kh) * SXh + b * SXl) )      paquet apres paquet
 *     y = acc * ech
 * Tout organe qui calcule les memes entiers et suit cet ordre rend les memes bits (verifie contre numpy : produit_entiers_numpy). */
int pli_qmm_ent(const uint32_t *w, const void *s, const void *b, int sb, long N, long K, const int8_t *xh, const int8_t *xl,
                const float *ech, long L, float *y, long n0, long n1, int fils) {
    if (K % 64 || n0 < 0 || n1 > N || n0 > n1 || L < 1) return -1;
    static const int8_t zeros[64] = {0};
    const long G = K / 64, M = n1 - n0, Lp = (L + 3) & ~3L;
    const float kh = xl ? 256.0f : 1.0f;
    float *sxh = calloc((size_t)Lp * G, sizeof(float)), *sxl = calloc((size_t)Lp * G, sizeof(float));   /* [g][l] */
    if (!sxh || !sxl) { free(sxh); free(sxl); return -2; }
    for (long l = 0; l < L; l++)
        for (long g = 0; g < G; g++) {
            int32_t a = 0, c = 0;
            for (int j = 0; j < 64; j++) { a += xh[l * K + 64 * g + j]; if (xl) c += xl[l * K + 64 * g + j]; }
            sxh[g * Lp + l] = (float)a; sxl[g * Lp + l] = (float)c;
        }
    parts(n0, n1, fils, ^(long a, long bb) {
        float32x4_t acc[BL / 4];
        float sortie[4];
        for (long l0 = 0; l0 < L; l0 += BL) {
            const long nl = L - l0 < BL ? L - l0 : BL, nq = (nl + 3) / 4;
            for (long n = a; n < bb; n++) {
                const uint8_t *pw = (const uint8_t *)(w + n * (K / 8));
                for (long v = 0; v < nq; v++) acc[v] = vdupq_n_f32(0.0f);
                for (long g = 0; g < G; g++) {
                    uint8x16_t q[4];
                    deballer(pw + 32 * g, q);
                    const int8x16_t q0 = vreinterpretq_s8_u8(q[0]), q1 = vreinterpretq_s8_u8(q[1]), q2 = vreinterpretq_s8_u8(q[2]),
                                    q3 = vreinterpretq_s8_u8(q[3]);
                    const float sv = lire_sb(s, n * G + g, sb), bv = lire_sb(b, n * G + g, sb);
                    const float32x4_t vs = vdupq_n_f32(sv), vsk = vdupq_n_f32(sv * kh), vb = vdupq_n_f32(bv), vbk = vdupq_n_f32(bv * kh);
                    for (long v = 0; v < nq; v++) {
                        int32x4_t h[4], lo[4];
                        for (int e = 0; e < 4; e++) {
                            const long l = 4 * v + e;
                            const int8_t *ph = l < nl ? xh + (l0 + l) * K + 64 * g : zeros;
                            int32x4_t t = vdotq_s32(vdupq_n_s32(0), q0, vld1q_s8(ph));
                            t = vdotq_s32(t, q1, vld1q_s8(ph + 16));
                            t = vdotq_s32(t, q2, vld1q_s8(ph + 32));
                            h[e] = vdotq_s32(t, q3, vld1q_s8(ph + 48));
                            lo[e] = vdupq_n_s32(0);
                            if (xl) {
                                const int8_t *pl = l < nl ? xl + (l0 + l) * K + 64 * g : zeros;
                                t = vdotq_s32(vdupq_n_s32(0), q0, vld1q_s8(pl));
                                t = vdotq_s32(t, q1, vld1q_s8(pl + 16));
                                t = vdotq_s32(t, q2, vld1q_s8(pl + 32));
                                lo[e] = vdotq_s32(t, q3, vld1q_s8(pl + 48));
                            }
                        }
                        const float32x4_t Ph = vcvtq_f32_s32(vpaddq_s32(vpaddq_s32(h[0], h[1]), vpaddq_s32(h[2], h[3])));
                        const float32x4_t Pl = vcvtq_f32_s32(vpaddq_s32(vpaddq_s32(lo[0], lo[1]), vpaddq_s32(lo[2], lo[3])));
                        const float32x4_t u = vaddq_f32(vmulq_f32(vsk, Ph), vmulq_f32(vs, Pl));
                        const float32x4_t c = vaddq_f32(vmulq_f32(vbk, vld1q_f32(sxh + g * Lp + l0 + 4 * v)),
                                                        vmulq_f32(vb, vld1q_f32(sxl + g * Lp + l0 + 4 * v)));
                        acc[v] = vaddq_f32(acc[v], vaddq_f32(u, c));
                    }
                }
                for (long v = 0; v < nq; v++) {
                    vst1q_f32(sortie, acc[v]);
                    for (int e = 0; e < 4 && 4 * v + e < nl; e++) y[(l0 + 4 * v + e) * M + (n - n0)] = sortie[e] * ech[l0 + 4 * v + e];
                }
            }
        }
    });
    free(sxh); free(sxl);
    return 0;
}

/* SwiGLU sur place : g = g / (1 + exp(-g)) * u, n valeurs, par parts de 16 384 (vvexpf d'Accelerate). */
int pli_swiglu(float *g, const float *u, long n) {
    const long PT = 16384;
    long nb = (n + PT - 1) / PT;
    dispatch_apply((size_t)nb, file_de_travail(), ^(size_t i) {
      long a = (long)i * PT, b = a + PT > n ? n : a + PT;
      float t[1024];
      for (long j = a; j < b; j += 1024) {
          int c = (int)(b - j > 1024 ? 1024 : b - j);
          for (int k = 0; k < c; k++) t[k] = -g[j + k];
          vvexpf(t, t, &c);
          for (int k = 0; k < c; k++) g[j + k] = g[j + k] / (1.0f + t[k]) * u[j + k];
      }
    });
    return 0;
}

/* Lot E4 (entiers d'abord, memes bits que le noyau GPU simd_qmm « _ent » de TensorFold). x : entiers de 16 bits, |x| <= 16383.
 * Par groupe g de 64 colonnes : P = somme des q * x (entier exact, sous 2^24), SX = somme des x (entier exact).
 * S morceaux : le morceau c prend les groupes c, c + S, c + 2 S, ... dans l'ordre, avec
 *     acc_c = fma(b_g, SX_g, fma(s_g, P_g, acc_c))
 * puis arbre : pour w = 1, 2, 4, ... : v[k] = v[k + w] + v[k] pour k = 0, 2 w, 4 w, ...  Sortie : v[0], en 32 bits, sans echelle.
 * S est celui de simd_qmm.splits : 32 si N <= 64, 16 si N <= 6144, sinon 8. */
int pli_qmm_e15(const uint32_t *w, const void *s, const void *b, int sb, long N, long K, const int16_t *x, long L, float *y,
                long n0, long n1, int S, int fils) {
    if (K % 64 || n0 < 0 || n1 > N || n0 > n1 || L < 1 || S < 1 || S > 32 || (S & (S - 1))) return -1;
    const long G = K / 64, M = n1 - n0;
    float *sx = malloc((size_t)L * G * sizeof(float));
    if (!sx) return -2;
    for (long l = 0; l < L; l++)
        for (long g = 0; g < G; g++) {
            int32_t a = 0;
            for (int j = 0; j < 64; j++) a += x[l * K + 64 * g + j];
            sx[l * G + g] = (float)a;
        }
    parts(n0, n1, fils, ^(long a, long bb) {
        float *sc = malloc((size_t)G * 2 * sizeof(float)), *bi = sc + G;
        int16_t *qd = malloc((size_t)K * sizeof(int16_t));
        for (long n = a; n < bb; n++) {
            const uint8_t *pw = (const uint8_t *)(w + n * (K / 8));
            for (long g = 0; g < G; g++) {
                uint8x16_t q[4];
                deballer(pw + 32 * g, q);
                for (int e = 0; e < 4; e++) {
                    vst1q_s16(qd + 64 * g + 16 * e, vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(q[e]))));
                    vst1q_s16(qd + 64 * g + 16 * e + 8, vreinterpretq_s16_u16(vmovl_high_u8(q[e])));
                }
                sc[g] = lire_sb(s, n * G + g, sb);
                bi[g] = lire_sb(b, n * G + g, sb);
            }
            for (long l = 0; l < L; l++) {
                const int16_t *px = x + l * K;
                const float *psx = sx + l * G;
                float v[32];
                for (int c = 0; c < S; c++) {
                    float acc = 0.0f;
                    for (long g = c; g < G; g += S) {
                        int32x4_t t = vdupq_n_s32(0);
                        for (int j = 0; j < 64; j += 8) {
                            const int16x8_t xv = vld1q_s16(px + 64 * g + j), qv = vld1q_s16(qd + 64 * g + j);
                            t = vmlal_s16(t, vget_low_s16(xv), vget_low_s16(qv));
                            t = vmlal_high_s16(t, xv, qv);
                        }
                        acc = fmaf(bi[g], psx[g], fmaf(sc[g], (float)vaddvq_s32(t), acc));
                    }
                    v[c] = acc;
                }
                for (int ww = 1; ww < S; ww *= 2)
                    for (int k = 0; k + ww < S; k += 2 * ww) v[k] = v[k + ww] + v[k];
                y[l * M + (n - n0)] = v[0];
            }
        }
        free(sc); free(qd);
    });
    free(sx);
    return 0;
}
