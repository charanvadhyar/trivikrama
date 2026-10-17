/*
 * Trivikrama kernels.
 * Copyright 2026 Vadhyar Sai Charan
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trivikrama.h"

#include <math.h>
#include <string.h>

static int8_t q8(float v, float s)
{
    long q = lrintf(v / s);
    if (q > 127) q = 127;
    if (q < -127) q = -127;
    return (int8_t)q;
}

void trv_quantize_act_q8g(const float *x, uint32_t T, uint32_t K, uint32_t ldx,
                          uint32_t group, int8_t *xq, uint32_t ldq,
                          float *xscale, uint32_t lds)
{
    uint32_t t, g0, k;
    for (t = 0; t < T; ++t) {
        const float *row = x + (size_t)t * ldx;
        int8_t *qrow = xq + (size_t)t * ldq;
        float *srow = xscale + (size_t)t * lds;
        for (g0 = 0; g0 < K; g0 += group) {
            float maxabs = 0.0f, s;
            for (k = g0; k < g0 + group; ++k) {
                float a = fabsf(row[k]);
                if (a > maxabs) maxabs = a;
            }
            s = (maxabs > 0.0f) ? maxabs / 127.0f : 1.0f;
            srow[g0 / group] = s;
            for (k = g0; k < g0 + group; ++k) qrow[k] = q8(row[k], s);
        }
    }
}

void trv_quantize_act_q8(const float *x, uint32_t T, uint32_t K, uint32_t ldx,
                         int8_t *xq, float *xscale)
{
    trv_quantize_act_q8g(x, T, K, ldx, K, xq, K, xscale, 1);
}

void trv_layernorm_quantize(const float *x, uint32_t T, uint32_t D, uint32_t ldx,
                            const float *gamma, const float *beta, float eps,
                            uint32_t group, int8_t *xq, float *xscale)
{
    uint32_t t, i, g0, ng = D / group;
    for (t = 0; t < T; ++t) {
        const float *row = x + (size_t)t * ldx;
        float mean = 0.0f, var = 0.0f, inv;
        for (i = 0; i < D; ++i) mean += row[i];
        mean /= (float)D;
        for (i = 0; i < D; ++i) {
            float d = row[i] - mean;
            var += d * d;
        }
        var /= (float)D;
        inv = 1.0f / sqrtf(var + eps);
        /* two passes per group: find the scale, then quantize. The normalized
         * value is recomputed rather than stored, so no float buffer is used. */
        for (g0 = 0; g0 < D; g0 += group) {
            float maxabs = 0.0f, s;
            for (i = g0; i < g0 + group; ++i) {
                float v = (row[i] - mean) * inv * gamma[i];
                if (beta) v += beta[i];
                if (fabsf(v) > maxabs) maxabs = fabsf(v);
            }
            s = (maxabs > 0.0f) ? maxabs / 127.0f : 1.0f;
            xscale[(size_t)t * ng + g0 / group] = s;
            for (i = g0; i < g0 + group; ++i) {
                float v = (row[i] - mean) * inv * gamma[i];
                if (beta) v += beta[i];
                xq[(size_t)t * D + i] = q8(v, s);
            }
        }
    }
}

void trv_layernorm(float *x, uint32_t T, uint32_t D, uint32_t ldx,
                   const float *gamma, const float *beta, float eps)
{
    uint32_t t, i;
    for (t = 0; t < T; ++t) {
        float *row = x + (size_t)t * ldx;
        float mean = 0.0f, var = 0.0f, inv;
        for (i = 0; i < D; ++i) mean += row[i];
        mean /= (float)D;
        for (i = 0; i < D; ++i) {
            float d = row[i] - mean;
            var += d * d;
        }
        var /= (float)D;
        inv = 1.0f / sqrtf(var + eps);
        for (i = 0; i < D; ++i) {
            float v = (row[i] - mean) * inv * gamma[i];
            row[i] = beta ? v + beta[i] : v;
        }
    }
}

static float gelu1(float v)
{
    return 0.5f * v * (1.0f + erff(v * 0.70710678118654752f));
}

void trv_gelu(float *x, uint32_t n)
{
    uint32_t i;
    for (i = 0; i < n; ++i) x[i] = gelu1(x[i]);
}

void trv_geglu(const float *in, uint32_t T, uint32_t I, uint32_t ldin,
               float *out, uint32_t ldout)
{
    uint32_t t, i;
    for (t = 0; t < T; ++t) {
        const float *a = in + (size_t)t * ldin;
        float *o = out + (size_t)t * ldout;
        for (i = 0; i < I; ++i) {
            float gate = a[I + i];
            o[i] = gelu1(a[i]) * gate;
        }
    }
}

void trv_rope_pos(float *x, uint32_t T, uint32_t n_heads, uint32_t head_dim,
                  uint32_t ldx, const uint32_t *pos, float base)
{
    uint32_t half = head_dim / 2u, i, t, h;
    for (i = 0; i < half; ++i) {
        float inv_freq = powf(base, -2.0f * (float)i / (float)head_dim);
        for (t = 0; t < T; ++t) {
            float ang = (float)pos[t] * inv_freq;
            float c = cosf(ang), s = sinf(ang);
            for (h = 0; h < n_heads; ++h) {
                float *xh = x + (size_t)t * ldx + (size_t)h * head_dim;
                float a = xh[i], b = xh[i + half];
                xh[i] = a * c - b * s;
                xh[i + half] = b * c + a * s;
            }
        }
    }
}

void trv_rope(float *x, uint32_t T, uint32_t n_heads, uint32_t head_dim,
              uint32_t ldx, uint32_t pos0, float base)
{
    uint32_t half = head_dim / 2u, i, t, h;
    for (i = 0; i < half; ++i) {
        float inv_freq = powf(base, -2.0f * (float)i / (float)head_dim);
        for (t = 0; t < T; ++t) {
            float ang = (float)(pos0 + t) * inv_freq;
            float c = cosf(ang), s = sinf(ang);
            for (h = 0; h < n_heads; ++h) {
                float *xh = x + (size_t)t * ldx + (size_t)h * head_dim;
                float a = xh[i], b = xh[i + half];
                xh[i] = a * c - b * s;
                xh[i + half] = b * c + a * s;
            }
        }
    }
}

void trv_attention_head_pos(const float *q, const float *k, const float *v,
                            uint32_t ld, uint32_t T, uint32_t d,
                            int32_t half_window, const uint8_t *key_mask,
                            const uint32_t *pos, float *out, uint32_t ldo)
{
    const float scale = 1.0f / sqrtf((float)d);
    uint32_t i, j, c, j0 = 0, j1 = 0;

    for (i = 0; i < T; ++i) {
        const float *qi = q + (size_t)i * ld;
        float *acc = out + (size_t)i * ldo;
        float mx = -INFINITY, l = 0.0f;
        uint32_t lo = 0, hi = T;

        if (half_window >= 0) {
            uint32_t hw = (uint32_t)half_window;
            if (!pos) {
                lo = (i > hw) ? i - hw : 0u;
                hi = (i + hw + 1u < T) ? i + hw + 1u : T;
            } else {
                /* positions increase, so the window slides forward with i */
                uint32_t p = pos[i], plo = p > hw ? p - hw : 0u;
                while (j0 < T && pos[j0] < plo) ++j0;
                if (j1 < j0) j1 = j0;
                while (j1 < T && pos[j1] <= p + hw) ++j1;
                lo = j0;
                hi = j1;
            }
        }
        for (c = 0; c < d; ++c) acc[c] = 0.0f;

        for (j = lo; j < hi; ++j) {
            const float *kj, *vj;
            float s = 0.0f, p;
            if (key_mask && !key_mask[j]) continue;
            kj = k + (size_t)j * ld;
            vj = v + (size_t)j * ld;
            for (c = 0; c < d; ++c) s += qi[c] * kj[c];
            s *= scale;
            if (s > mx) { /* new running max: rescale what we have so far */
                float corr = expf(mx - s); /* exp(-inf) == 0 on the first key */
                l *= corr;
                for (c = 0; c < d; ++c) acc[c] *= corr;
                mx = s;
            }
            p = expf(s - mx);
            l += p;
            for (c = 0; c < d; ++c) acc[c] += p * vj[c];
        }
        if (l > 0.0f) {
            float inv = 1.0f / l;
            for (c = 0; c < d; ++c) acc[c] *= inv;
        }
    }
}

void trv_attention_head(const float *q, const float *k, const float *v,
                        uint32_t ld, uint32_t T, uint32_t d,
                        int32_t half_window, const uint8_t *key_mask,
                        float *out, uint32_t ldo)
{
    trv_attention_head_pos(q, k, v, ld, T, d, half_window, key_mask, NULL, out, ldo);
}
