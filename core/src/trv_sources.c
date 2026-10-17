/*
 * Portable weight sources: plain memory, and tiers that route byte ranges of
 * one model to different sources (internal flash, PSRAM, SD card...).
 * Copyright 2026 Vadhyar Sai Charan
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trivikrama.h"

#include <string.h>

/* ---- memory ----------------------------------------------------------------- */

static int mem_read(void *vctx, uint32_t off, void *dst, uint32_t n)
{
    trv_mem_ctx *c = (trv_mem_ctx *)vctx;
    if (off > c->size || n > c->size - off) return -1;
    memcpy(dst, c->base + off, n);
    c->bytes_read += n;
    return 0;
}

trv_status trv_source_mem_init(trv_source *src, trv_mem_ctx *ctx, const void *base,
                               uint32_t size)
{
    if (!src || !ctx || !base) return TRV_ERR_ARG;
    ctx->base = (const uint8_t *)base;
    ctx->size = size;
    ctx->bytes_read = 0;
    memset(src, 0, sizeof *src);
    src->read = mem_read;
    src->ctx = ctx;
    return TRV_OK;
}

/* ---- tiers -------------------------------------------------------------------- */

static int tier_route(trv_tiered_ctx *c, uint32_t off, void *dst, uint32_t n, int async)
{
    uint8_t *d = (uint8_t *)dst;
    while (n > 0) {
        trv_tier *t = NULL;
        uint32_t i, take;
        const trv_source *s;
        for (i = 0; i < c->n_tiers; ++i)
            if (off >= c->tiers[i].begin && off < c->tiers[i].end) {
                t = &c->tiers[i];
                break;
            }
        if (!t) return -1;
        take = t->end - off;
        if (take > n) take = n;
        s = t->src;
        if (async && s->read_async && s->wait) {
            if (s->read_async(s->ctx, t->src_offset + (off - t->begin), d, take) != 0)
                return -1;
        } else if (s->read(s->ctx, t->src_offset + (off - t->begin), d, take) != 0) {
            return -1;
        }
        t->bytes_read += take;
        off += take;
        d += take;
        n -= take;
    }
    return 0;
}

static int tier_read(void *vctx, uint32_t off, void *dst, uint32_t n)
{
    return tier_route((trv_tiered_ctx *)vctx, off, dst, n, 0);
}

static int tier_read_async(void *vctx, uint32_t off, void *dst, uint32_t n)
{
    return tier_route((trv_tiered_ctx *)vctx, off, dst, n, 1);
}

static int tier_wait(void *vctx)
{
    trv_tiered_ctx *c = (trv_tiered_ctx *)vctx;
    uint32_t i, j;
    int rc = 0;
    for (i = 0; i < c->n_tiers; ++i) {
        const trv_source *s = c->tiers[i].src;
        int seen = 0;
        for (j = 0; j < i; ++j) seen |= c->tiers[j].src == s; /* wait once per source */
        if (!seen && s->read_async && s->wait && s->wait(s->ctx) != 0) rc = -1;
    }
    return rc;
}

trv_status trv_source_tiered_init(trv_source *src, trv_tiered_ctx *ctx, trv_tier *tiers,
                                  uint32_t n_tiers)
{
    uint32_t i, j;
    int any_async = 0;
    if (!src || !ctx || !tiers || n_tiers == 0) return TRV_ERR_ARG;
    for (i = 0; i < n_tiers; ++i) {
        if (!tiers[i].src || !tiers[i].src->read || tiers[i].begin >= tiers[i].end)
            return TRV_ERR_ARG;
        for (j = 0; j < i; ++j)
            if (tiers[i].begin < tiers[j].end && tiers[j].begin < tiers[i].end)
                return TRV_ERR_ARG; /* overlapping ranges */
        tiers[i].bytes_read = 0;
        any_async |= tiers[i].src->read_async && tiers[i].src->wait;
    }
    ctx->tiers = tiers;
    ctx->n_tiers = n_tiers;
    memset(src, 0, sizeof *src);
    src->read = tier_read;
    src->ctx = ctx;
    if (any_async) {
        src->read_async = tier_read_async;
        src->wait = tier_wait;
    }
    return TRV_OK;
}
