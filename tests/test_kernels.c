/*
 * Test harness: reads inputs from a .trv file (written by check_kernels.py),
 * runs every kernel, writes raw float32 outputs for comparison with NumPy.
 *
 *   test_kernels cases.trv out_dir
 *
 * Copyright 2026 Vadhyar Sai Charan
 * SPDX-License-Identifier: Apache-2.0
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "trivikrama.h"
#include "trivikrama_file.h"

#define MAXF 4096
#define HEAD_DIM 8

static trv_model g_model;
static trv_file_ctx g_fctx;

static void die(const char *what, trv_status st)
{
    fprintf(stderr, "FAIL: %s (status %d)\n", what, (int)st);
    exit(1);
}

static trv_tensor load_f32(const char *name, float *dst)
{
    trv_tensor t;
    trv_status st = trv_model_find(&g_model, name, &t);
    if (st != TRV_OK) die(name, st);
    if (trv_tensor_numel(&t) > MAXF) die("tensor too large for harness", TRV_ERR_ARG);
    st = trv_tensor_read_f32(&g_model, &t, 0, dst, trv_tensor_numel(&t));
    if (st != TRV_OK) die(name, st);
    return t;
}

static void save(const char *dir, const char *name, const float *x, uint32_t n)
{
    char path[512];
    FILE *f;
    snprintf(path, sizeof path, "%s/%s.bin", dir, name);
    f = fopen(path, "wb");
    if (!f || fwrite(x, sizeof(float), n, f) != n) die(path, TRV_ERR_IO);
    fclose(f);
}

int main(int argc, char **argv)
{
    static float x[MAXF], b[MAXF], y[MAXF], g[MAXF], be[MAXF];
    static float q[MAXF], k[MAXF], v[MAXF], o[MAXF], mk[MAXF];
    static int8_t xq[MAXF];
    static float xs[MAXF];
    static uint32_t tile[64]; /* 256 bytes: deliberately tiny to force many tiles */
    static uint8_t mask[MAXF];
    trv_source src;
    trv_tensor t, w;
    trv_status st;
    uint32_t T, K, N, D, W2, H, i, half;
    const char *out;

    if (argc != 3) {
        fprintf(stderr, "usage: %s cases.trv out_dir\n", argv[0]);
        return 2;
    }
    out = argv[2];
    if ((st = trv_source_file_open(&src, &g_fctx, argv[1])) != TRV_OK) die("open", st);
    if ((st = trv_model_open(&g_model, &src)) != TRV_OK) die("model_open", st);

    /* 1. streamed int8 linear, computed as two column ranges into one output */
    t = load_f32("x", x);
    T = t.dims[0];
    K = t.dims[1];
    load_f32("b", b);
    if ((st = trv_model_find(&g_model, "w", &w)) != TRV_OK) die("w", st);
    N = w.dims[0];
    trv_quantize_act_q8(x, T, K, K, xq, xs);
    half = N / 2;
    {
        uint32_t before = g_fctx.bytes_read;
        st = trv_linear_q8_stream(&g_model, &w, xq, xs, T, b, 0, half, y, N,
                                  tile, sizeof tile);
        if (st != TRV_OK) die("linear[0:half]", st);
        st = trv_linear_q8_stream(&g_model, &w, xq, xs, T, b, half, N, y + half, N,
                                  tile, sizeof tile);
        if (st != TRV_OK) die("linear[half:N]", st);
        printf("linear: T=%u K=%u N=%u, weight bytes=%u, bytes streamed=%u (%s)\n",
               T, K, N, w.nbytes, g_fctx.bytes_read - before,
               g_fctx.bytes_read - before == w.nbytes ? "each weight read once"
                                                      : "MISMATCH");
        if (g_fctx.bytes_read - before != w.nbytes) die("weights re-read", TRV_ERR_ARG);
    }
    save(out, "linear", y, T * N);

    /* buffer-too-small path must be reported, not crash */
    st = trv_linear_q8_stream(&g_model, &w, xq, xs, T, b, 0, N, y, N, tile, K);
    if (st != TRV_ERR_BUFFER) die("expected TRV_ERR_BUFFER", st);

    /* 2. layernorm (with and without beta) */
    t = load_f32("ln_x", x);
    T = t.dims[0];
    D = t.dims[1];
    load_f32("ln_g", g);
    load_f32("ln_b", be);
    memcpy(y, x, sizeof(float) * T * D);
    trv_layernorm(x, T, D, D, g, be, 1e-5f);
    save(out, "layernorm", x, T * D);
    trv_layernorm(y, T, D, D, g, NULL, 1e-5f);
    save(out, "layernorm_nobias", y, T * D);

    /* 3. GeGLU, in place */
    t = load_f32("gg_x", x);
    T = t.dims[0];
    W2 = t.dims[1];
    trv_geglu(x, T, W2 / 2, W2, x, W2);
    for (i = 0; i < T; ++i) memmove(y + i * (W2 / 2), x + i * W2, sizeof(float) * (W2 / 2));
    save(out, "geglu", y, T * (W2 / 2));

    /* 4. RoPE */
    t = load_f32("rope_x", x);
    T = t.dims[0];
    D = t.dims[1];
    trv_rope(x, T, D / HEAD_DIM, HEAD_DIM, D, 3, 10000.0f);
    save(out, "rope", x, T * D);

    /* 5. attention: head 0 global + key mask, head 1 sliding window */
    t = load_f32("q", q);
    T = t.dims[0];
    D = t.dims[1];
    H = D / HEAD_DIM;
    load_f32("k", k);
    load_f32("v", v);
    load_f32("kmask", mk);
    for (i = 0; i < T; ++i) mask[i] = mk[i] > 0.5f;
    memset(o, 0, sizeof(float) * T * D);
    trv_attention_head(q, k, v, D, T, HEAD_DIM, -1, mask, o, D);
    if (H > 1) trv_attention_head(q + HEAD_DIM, k + HEAD_DIM, v + HEAD_DIM, D, T,
                                  HEAD_DIM, 2, NULL, o + HEAD_DIM, D);
    save(out, "attention", o, T * D);

    /* 6. embedding-style row lookup from the Q8 tensor */
    st = trv_tensor_read_q8_row(&g_model, &w, N - 1, y, xq);
    if (st != TRV_OK) die("read_q8_row", st);
    save(out, "q8_row", y, K);

    trv_source_file_close(&g_fctx);
    printf("C harness OK\n");
    return 0;
}
