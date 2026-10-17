/*
 * Streamed linear layer: int8 or int4 weights, read tile by tile from any
 * trv_source, applied to every token before the next tile is read.
 *
 *   - runtime hooks: SIMD dot product, multiple worker cores
 *   - double buffering: with an async source the next tile loads while the
 *     current one is being multiplied
 *
 * Copyright 2026 Vadhyar Sai Charan
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trivikrama.h"

#include <math.h>
#include <string.h>

/* ---- runtime ------------------------------------------------------------- */

static trv_runtime g_rt;

/* cross-call prefetch state */
static uint32_t g_min_scratch;
static struct {
    int set;
    trv_tensor w;
    uint32_t group, n_begin, n_end;
} g_next;
static struct {
    int valid;
    const trv_source *src;
    const void *buf;
    uint32_t tile_bytes, w_offset, n0, r, half;
} g_pf;

void trv_set_runtime(const trv_runtime *rt)
{
    if (rt)
        g_rt = *rt;
    else
        memset(&g_rt, 0, sizeof g_rt);
}

const trv_runtime *trv_get_runtime(void) { return &g_rt; }

static void dot_groups_c(const int8_t *x, const int8_t *w, uint32_t K, uint32_t chunk,
                         int32_t *acc)
{
    uint32_t j, k, n = K / chunk;
    for (j = 0; j < n; ++j) {
        const int8_t *a = x + (size_t)j * chunk, *b = w + (size_t)j * chunk;
        int32_t s = 0;
        for (k = 0; k < chunk; ++k) s += (int32_t)a[k] * (int32_t)b[k];
        acc[j] = s;
    }
}

static uint32_t n_workers(void)
{
    return (g_rt.parallel && g_rt.n_workers > 1) ? g_rt.n_workers : 1u;
}

/* ---- sizes ------------------------------------------------------------------ */

static uint32_t up4(uint32_t n) { return (n + 3u) & ~3u; }

uint32_t trv_weight_row_bytes(uint8_t dtype, uint32_t K)
{
    if (dtype == TRV_DTYPE_Q8_ROWS) return K + 4u; /* row + its scale */
    if (dtype == TRV_DTYPE_Q4_G32) return TRV_Q4_ROW_BYTES(K);
    return 0;
}

/* inner dot-product chunk: one activation group (int8), or the smaller of the
 * activation group and the 32-wide weight group (int4) */
static uint32_t chunk_for(uint8_t dtype, uint32_t group)
{
    if (dtype == TRV_DTYPE_Q4_G32) return group < TRV_Q4_GROUP ? group : TRV_Q4_GROUP;
    return group;
}

static uint32_t worker_scratch(uint8_t dtype, uint32_t K, uint32_t group)
{
    uint32_t c = chunk_for(dtype, group), bytes = up4(4u * (K / c));
    if (dtype == TRV_DTYPE_Q4_G32) bytes += up4(K) + up4(4u * (K / TRV_Q4_GROUP));
    return bytes;
}

uint32_t trv_linear_scratch_bytes(uint8_t dtype, uint32_t K, uint32_t group)
{
    return worker_scratch(dtype, K, group) * n_workers();
}

/* IEEE half -> float, exact */
static float half_to_float(uint16_t h)
{
    uint32_t exp = (h >> 10) & 0x1fu, man = h & 0x3ffu;
    float f;
    if (exp == 0)
        f = ldexpf((float)man, -24);
    else if (exp == 31)
        f = man ? NAN : INFINITY;
    else
        f = ldexpf((float)(man | 0x400u), (int)exp - 25);
    return (h & 0x8000u) ? -f : f;
}

/* ---- compute one tile ------------------------------------------------------ */

typedef struct {
    uint8_t dtype;
    uint32_t K, T, group, chunk, ngx, row_bytes, n_begin, n0, r;
    const int8_t *xq;
    const float *xscale, *bias;
    float *y;
    uint32_t ldy;
    int accumulate;
    const uint8_t *tile;
    uint8_t *scratch;
    uint32_t scratch_per_worker;
    trv_dot_groups_fn dot;
} lin_job;

static void store(const lin_job *J, uint32_t t, uint32_t n, float val)
{
    float *dst = J->y + (size_t)t * J->ldy + (n - J->n_begin);
    if (J->accumulate)
        *dst += val;
    else
        *dst = val;
}

static void lin_task(void *arg, uint32_t worker, uint32_t nw)
{
    const lin_job *J = (const lin_job *)arg;
    uint32_t i0 = (uint32_t)((uint64_t)J->r * worker / nw);
    uint32_t i1 = (uint32_t)((uint64_t)J->r * (worker + 1) / nw);
    uint8_t *sc = J->scratch + (size_t)worker * J->scratch_per_worker;
    uint32_t K = J->K, c = J->chunk, nc = K / c, i, t, j;
    int32_t *acc = (int32_t *)(void *)sc;

    for (i = i0; i < i1; ++i) {
        uint32_t n = J->n0 + i;
        float b = J->bias ? J->bias[n] : 0.0f;

        if (J->dtype == TRV_DTYPE_Q8_ROWS) {
            /* tile = [r float row scales][r rows of K int8] */
            const int8_t *wr = (const int8_t *)(J->tile + 4u * J->r) + (size_t)i * K;
            float ws;
            memcpy(&ws, J->tile + 4u * i, 4); /* unaligned-safe */
            for (t = 0; t < J->T; ++t) {
                const float *xs = J->xscale + (size_t)t * J->ngx;
                float sum = 0.0f;
                J->dot(J->xq + (size_t)t * K, wr, K, c, acc);
                for (j = 0; j < nc; ++j) sum += (float)acc[j] * xs[j];
                store(J, t, n, sum * ws + b);
            }
        } else {
            /* tile = r rows of [K/32 fp16 scales][K/2 packed nibbles];
             * unpack the row once, reuse it for every token */
            const uint8_t *row = J->tile + (size_t)i * J->row_bytes;
            const uint8_t *packed = row + 2u * (K / TRV_Q4_GROUP);
            int8_t *wq = (int8_t *)(sc + up4(4u * nc));
            float *gs = (float *)(void *)(sc + up4(4u * nc) + up4(K));
            uint32_t g, k;
            for (g = 0; g < K / TRV_Q4_GROUP; ++g) {
                uint16_t hb;
                memcpy(&hb, row + 2u * g, 2);
                gs[g] = half_to_float(hb);
            }
            for (k = 0; k < K / 2u; ++k) {
                uint8_t byte = packed[k];
                wq[2u * k] = (int8_t)((int)(byte & 15u) - 8);
                wq[2u * k + 1u] = (int8_t)((int)(byte >> 4) - 8);
            }
            for (t = 0; t < J->T; ++t) {
                const float *xs = J->xscale + (size_t)t * J->ngx;
                float sum = 0.0f;
                J->dot(J->xq + (size_t)t * K, wq, K, c, acc);
                for (j = 0; j < nc; ++j) {
                    uint32_t k0 = j * c;
                    sum += (float)acc[j] * xs[k0 / J->group] * gs[k0 / TRV_Q4_GROUP];
                }
                store(J, t, n, sum + b);
            }
        }
    }
}

static void compute_tile(lin_job *J, const uint8_t *tile, uint32_t n0, uint32_t r)
{
    uint32_t nw = n_workers();
    J->tile = tile;
    J->n0 = n0;
    J->r = r;
    if (nw > 1)
        g_rt.parallel(g_rt.parallel_ctx, lin_task, J, nw);
    else
        lin_task(J, 0, 1);
}

/* ---- fetch one tile ---------------------------------------------------------- */

static int fetch(const trv_source *s, int async, const trv_tensor *w, uint32_t n0,
                 uint32_t r, uint8_t *dst)
{
    int (*rd)(void *, uint32_t, void *, uint32_t) = async ? s->read_async : s->read;
    uint32_t N = w->dims[0], K = w->dims[1];
    if (w->dtype == TRV_DTYPE_Q8_ROWS) {
        if (rd(s->ctx, w->offset + 4u * n0, dst, 4u * r) != 0) return -1;
        return rd(s->ctx, w->offset + 4u * N + n0 * K, dst + 4u * r, r * K);
    }
    return rd(s->ctx, w->offset + n0 * TRV_Q4_ROW_BYTES(K), dst, r * TRV_Q4_ROW_BYTES(K));
}

/* ---- cross-call prefetch ------------------------------------------------------- */

void trv_linear_set_next(const trv_tensor *w, uint32_t group, uint32_t n_begin,
                         uint32_t n_end)
{
    if (!w) {
        g_next.set = 0;
        return;
    }
    g_next.set = 1;
    g_next.w = *w;
    g_next.group = group;
    g_next.n_begin = n_begin;
    g_next.n_end = n_end;
}

void trv_linear_reserve_scratch(uint32_t bytes) { g_min_scratch = up4(bytes); }

void trv_linear_drain(const trv_source *src)
{
    if (g_pf.valid && g_pf.src == src && src && src->wait) src->wait(src->ctx);
    g_pf.valid = 0;
}

static uint32_t scratch_for(uint8_t dtype, uint32_t K, uint32_t group)
{
    uint32_t s = worker_scratch(dtype, K, group) * n_workers();
    return s > g_min_scratch ? s : g_min_scratch;
}

/* ---- public entry point -------------------------------------------------------- */

trv_status trv_linear_stream(const trv_model *m, const trv_tensor *w,
                             const int8_t *xq, const float *xscale,
                             uint32_t group, uint32_t T,
                             const float *bias, uint32_t n_begin, uint32_t n_end,
                             float *y, uint32_t ldy, int accumulate,
                             void *tile_buf, uint32_t tile_bytes)
{
    const trv_source *src;
    uint32_t N, K, rowb, spw, scratch, avail, rows, n0, r;
    uint8_t *buf = (uint8_t *)tile_buf, *region;
    int async, have_next = g_next.set;
    trv_tensor next_w;
    uint32_t next_group = g_next.group, next_b = g_next.n_begin, next_e = g_next.n_end;
    lin_job J;

    next_w = g_next.w;
    g_next.set = 0; /* a hint applies to exactly one call */

    if (!m || !w || !xq || !xscale || !y || !tile_buf) return TRV_ERR_ARG;
    if (w->ndim != 2) return TRV_ERR_ARG;
    if (w->dtype != TRV_DTYPE_Q8_ROWS && w->dtype != TRV_DTYPE_Q4_G32) return TRV_ERR_ARG;
    if (((uintptr_t)tile_buf & 3u) != 0) return TRV_ERR_ARG;
    N = w->dims[0];
    K = w->dims[1];
    if (n_begin > n_end || n_end > N || K == 0) return TRV_ERR_ARG;
    if (group == 0 || K % group != 0) return TRV_ERR_ARG;
    if (w->dtype == TRV_DTYPE_Q4_G32) {
        if (K % TRV_Q4_GROUP != 0) return TRV_ERR_ARG;
        if (group % TRV_Q4_GROUP != 0 && TRV_Q4_GROUP % group != 0) return TRV_ERR_ARG;
    }
    src = m->src;
    if (n_begin == n_end) return TRV_OK;

    rowb = trv_weight_row_bytes(w->dtype, K);
    spw = worker_scratch(w->dtype, K, group);
    scratch = scratch_for(w->dtype, K, group);
    if (tile_bytes < scratch + rowb) return TRV_ERR_BUFFER;
    region = buf + scratch;
    avail = tile_bytes - scratch;

    memset(&J, 0, sizeof J);
    J.dtype = w->dtype;
    J.K = K;
    J.T = T;
    J.group = group;
    J.chunk = chunk_for(w->dtype, group);
    J.ngx = K / group;
    J.row_bytes = rowb;
    J.n_begin = n_begin;
    J.xq = xq;
    J.xscale = xscale;
    J.bias = bias;
    J.y = y;
    J.ldy = ldy;
    J.accumulate = accumulate;
    J.scratch = buf;
    J.scratch_per_worker = spw;
    J.dot = g_rt.dot_groups ? g_rt.dot_groups : dot_groups_c;

    async = src->read_async && src->wait && (avail / 2u) / rowb > 0;
    if (!async && g_pf.valid) trv_linear_drain(g_pf.src);
    if (async) {
        uint32_t half = (avail / 2u) & ~3u, cur = 0;
        uint8_t *bufs[2];
        bufs[0] = region;
        bufs[1] = region + half;
        rows = half / rowb;
        n0 = n_begin;
        r = (n_end - n0 < rows) ? n_end - n0 : rows;

        if (g_pf.valid && g_pf.src == src && g_pf.buf == tile_buf &&
            g_pf.tile_bytes == tile_bytes && g_pf.w_offset == w->offset &&
            g_pf.n0 == n0 && g_pf.r == r) {
            cur = g_pf.half; /* our first tile is already on its way */
            g_pf.valid = 0;
        } else {
            trv_linear_drain(g_pf.valid ? g_pf.src : NULL);
            if (fetch(src, 1, w, n0, r, bufs[0]) != 0) {
                src->wait(src->ctx);
                return TRV_ERR_IO;
            }
        }
        while (n0 < n_end) {
            uint32_t nn = n0 + r, rn = 0;
            if (src->wait(src->ctx) != 0) return TRV_ERR_IO; /* tile n0 is ready */
            if (nn < n_end) {                                /* start our next tile */
                rn = (n_end - nn < rows) ? n_end - nn : rows;
                if (fetch(src, 1, w, nn, rn, bufs[cur ^ 1u]) != 0) {
                    src->wait(src->ctx);
                    return TRV_ERR_IO;
                }
            } else if (have_next && next_w.ndim == 2 && next_b < next_e &&
                       (next_w.dtype == TRV_DTYPE_Q8_ROWS || next_w.dtype == TRV_DTYPE_Q4_G32) &&
                       scratch_for(next_w.dtype, next_w.dims[1], next_group) == scratch) {
                /* last tile of this call: start the next call's first tile */
                uint32_t nrows = half / trv_weight_row_bytes(next_w.dtype, next_w.dims[1]);
                if (nrows > 0) {
                    uint32_t nr = (next_e - next_b < nrows) ? next_e - next_b : nrows;
                    if (fetch(src, 1, &next_w, next_b, nr, bufs[cur ^ 1u]) != 0) {
                        src->wait(src->ctx);
                        return TRV_ERR_IO;
                    }
                    g_pf.valid = 1;
                    g_pf.src = src;
                    g_pf.buf = tile_buf;
                    g_pf.tile_bytes = tile_bytes;
                    g_pf.w_offset = next_w.offset;
                    g_pf.n0 = next_b;
                    g_pf.r = nr;
                    g_pf.half = cur ^ 1u;
                }
            }
            compute_tile(&J, bufs[cur], n0, r); /* overlaps with that read */
            n0 = nn;
            r = rn;
            cur ^= 1u;
        }
        return TRV_OK;
    }

    rows = avail / rowb;
    for (n0 = n_begin; n0 < n_end; n0 += rows) {
        r = (n_end - n0 < rows) ? n_end - n0 : rows;
        if (fetch(src, 0, w, n0, r, region) != 0) return TRV_ERR_IO;
        compute_tile(&J, region, n0, r);
    }
    return TRV_OK;
}

trv_status trv_linear_q8g_stream(const trv_model *m, const trv_tensor *w,
                                 const int8_t *xq, const float *xscale,
                                 uint32_t group, uint32_t T,
                                 const float *bias, uint32_t n_begin, uint32_t n_end,
                                 float *y, uint32_t ldy, int accumulate,
                                 void *tile_buf, uint32_t tile_bytes)
{
    return trv_linear_stream(m, w, xq, xscale, group, T, bias, n_begin, n_end, y, ldy,
                             accumulate, tile_buf, tile_bytes);
}

trv_status trv_linear_q8_stream(const trv_model *m, const trv_tensor *w,
                                const int8_t *xq, const float *xscale, uint32_t T,
                                const float *bias, uint32_t n_begin, uint32_t n_end,
                                float *y, uint32_t ldy,
                                void *tile_buf, uint32_t tile_bytes)
{
    if (!w || w->ndim != 2) return TRV_ERR_ARG;
    return trv_linear_stream(m, w, xq, xscale, w->dims[1], T, bias, n_begin, n_end, y,
                             ldy, 0, tile_buf, tile_bytes);
}
