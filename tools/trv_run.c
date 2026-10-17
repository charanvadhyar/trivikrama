/*
 * trv_run: run a ModernBERT .trv model on a PC as a "virtual microcontroller".
 *
 *   trv_run model.trv (--ids 12,99,7[;5,6,...] | --tokens N) [options]
 *
 *   --ids A;B       several sequences separated by ';' are run as one batch
 *   --tokens N      N deterministic ids; with --batch B, B such sequences
 *   --tile BYTES    weight tile buffer (default 65536)
 *   --budget BYTES  refuse to run if RAM (activations+scratch+tile) exceeds it
 *   --threads N     worker cores for the linear layers
 *   --async         read the next weight tile while computing the current one
 *   --sd-mbps X     simulate a storage device of X MB/s ...
 *   --sd-overhead-ms Y            ... with Y ms per read
 *   --flash-bytes B serve the first B bytes of the model from RAM ("internal
 *                   flash" tier) and only the rest from the simulated SD card
 *   --exit-after L  early exit: stop after layer L
 *   --drop-at L --drop i,j,...    token pruning: drop these original token
 *                   indices after layer L (-1 = right after the embeddings)
 *   --prune-at L --prune-keep F   generic pruning heuristic: after layer L keep
 *                   the fraction F of tokens with the largest hidden-state norm
 *   --mask 1101...  key mask (0 = padding)
 *   --dump DIR      write per-layer hidden states, final.bin and index.bin
 *
 * Prints one JSON line of measurements on stdout.
 *
 * Copyright 2026 Vadhyar Sai Charan
 * SPDX-License-Identifier: Apache-2.0
 */
#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "trivikrama.h"
#include "trivikrama_modernbert.h"
#include "trivikrama_posix.h"

#define MAX_TOKENS 8192
#define MAX_SEQS 256

typedef struct {
    const char *dir;
    int32_t exit_after;
    int32_t drop_at;
    uint32_t n_drop;
    uint32_t drop[MAX_TOKENS];
    int32_t prune_at;
    double prune_keep;
    uint32_t layers_run;
    uint32_t L;
} run_ctx;

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int cmp_desc(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x < y) - (x > y);
}

static void hook(void *user, int32_t layer, const float *h, uint32_t T, uint32_t H,
                 trv_mbert_ctl *ctl)
{
    run_ctx *r = (run_ctx *)user;
    uint32_t t, k;
    if (r->dir) {
        char path[1024];
        FILE *f;
        if (layer < 0)
            snprintf(path, sizeof path, "%s/emb.bin", r->dir);
        else
            snprintf(path, sizeof path, "%s/layer%03d.bin", r->dir, (int)layer);
        if ((f = fopen(path, "wb"))) {
            fwrite(h, sizeof(float), (size_t)T * H, f);
            fclose(f);
        }
    }
    if (layer >= 0 && layer < (int32_t)r->L) r->layers_run = (uint32_t)layer + 1;
    if (r->exit_after >= 0 && layer == r->exit_after) ctl->stop = 1;
    if (r->n_drop && layer == r->drop_at)
        for (t = 0; t < T; ++t)
            for (k = 0; k < r->n_drop; ++k)
                if (ctl->orig_index[t] == r->drop[k]) ctl->keep[t] = 0;
    if (r->prune_at == layer && r->prune_keep < 1.0 && T > 1) {
        /* keep the tokens whose hidden state has the largest L2 norm */
        static double norms[MAX_TOKENS], sorted[MAX_TOKENS];
        double thr;
        uint32_t keep_n = (uint32_t)ceil(r->prune_keep * T);
        for (t = 0; t < T; ++t) {
            double s = 0;
            for (k = 0; k < H; ++k) s += (double)h[(size_t)t * H + k] * h[(size_t)t * H + k];
            norms[t] = sorted[t] = s;
        }
        qsort(sorted, T, sizeof(double), cmp_desc);
        thr = sorted[keep_n ? keep_n - 1 : 0];
        for (t = 0; t < T; ++t) ctl->keep[t] = norms[t] >= thr;
    }
}

static uint32_t parse_list(const char *s, uint32_t *out, uint32_t cap)
{
    uint32_t n = 0;
    while (*s && n < cap) {
        char *end;
        out[n++] = (uint32_t)strtoul(s, &end, 10);
        if (*end != ',') break;
        s = end + 1;
    }
    return n;
}

static void fail(const char *what, int code)
{
    printf("{\"ok\": false, \"error\": \"%s\", \"code\": %d}\n", what, code);
    exit(1);
}

int main(int argc, char **argv)
{
    static uint32_t ids[MAX_TOKENS], lens[MAX_SEQS], index_out[MAX_TOKENS];
    static uint8_t mask[MAX_TOKENS];
    static run_ctx rc;
    const char *path = NULL, *ids_arg = NULL, *mask_arg = NULL;
    uint32_t T = 0, tokens = 0, batch = 1, n_seqs = 0, tile_bytes = 65536, threads = 1;
    uint32_t i, T_out = 0;
    long budget = -1;
    int async = 0;
    double sd_mbps = 0, sd_overhead_ms = 0;
    unsigned long flash_bytes = 0;
    trv_source file_src, tier_src, mem_src;
    trv_source *src;
    trv_filex *fx;
    trv_mem_ctx mem_ctx;
    trv_tier tiers[2];
    trv_tiered_ctx tier_ctx;
    trv_model model;
    trv_mbert mb;
    trv_runtime rt;
    trv_pool *pool = NULL;
    trv_status st;
    size_t ws, ram;
    float *h;
    void *arena, *tile;
    uint8_t *flash = NULL;
    double t0, t1;

    rc.exit_after = -2;
    rc.drop_at = -2;
    rc.prune_at = -2;
    rc.prune_keep = 1.0;
    for (i = 1; i < (uint32_t)argc; ++i) {
        const char *a = argv[i], *v = (i + 1 < (uint32_t)argc) ? argv[i + 1] : NULL;
        if (!strcmp(a, "--async")) { async = 1; continue; }
        if (a[0] == '-' && a[1] == '-' && !v) fail("missing value", 2);
        if (!strcmp(a, "--ids")) ids_arg = v;
        else if (!strcmp(a, "--tokens")) tokens = (uint32_t)atoi(v);
        else if (!strcmp(a, "--batch")) batch = (uint32_t)atoi(v);
        else if (!strcmp(a, "--tile")) tile_bytes = (uint32_t)atoi(v);
        else if (!strcmp(a, "--budget")) budget = atol(v);
        else if (!strcmp(a, "--threads")) threads = (uint32_t)atoi(v);
        else if (!strcmp(a, "--sd-mbps")) sd_mbps = atof(v);
        else if (!strcmp(a, "--sd-overhead-ms")) sd_overhead_ms = atof(v);
        else if (!strcmp(a, "--flash-bytes")) flash_bytes = strtoul(v, NULL, 10);
        else if (!strcmp(a, "--exit-after")) rc.exit_after = atoi(v);
        else if (!strcmp(a, "--drop-at")) rc.drop_at = atoi(v);
        else if (!strcmp(a, "--drop")) rc.n_drop = parse_list(v, rc.drop, MAX_TOKENS);
        else if (!strcmp(a, "--prune-at")) rc.prune_at = atoi(v);
        else if (!strcmp(a, "--prune-keep")) rc.prune_keep = atof(v);
        else if (!strcmp(a, "--mask")) mask_arg = v;
        else if (!strcmp(a, "--dump")) rc.dir = v;
        else if (!path) { path = a; continue; }
        else fail("bad arguments", 2);
        ++i;
    }
    if (!path || (!ids_arg && !tokens)) {
        fprintf(stderr, "usage: trv_run model.trv (--ids a,b;c,d | --tokens N [--batch B]) "
                        "[--threads N] [--async] [--sd-mbps X] ... (see source)\n");
        return 2;
    }

    /* storage: simulated SD card, optionally with a fast "flash" tier in front */
    fx = trv_filex_open(&file_src, path, async, sd_mbps * 1e6, sd_overhead_ms * 1e-3);
    if (!fx) fail("open", -1);
    src = &file_src;
    if (flash_bytes) {
        uint64_t size = trv_filex_size(fx);
        if (flash_bytes > size) flash_bytes = (unsigned long)size;
        flash = (uint8_t *)malloc(flash_bytes);
        if (!flash || file_src.read(file_src.ctx, 0, flash, (uint32_t)flash_bytes)) fail("flash", -1);
        trv_source_mem_init(&mem_src, &mem_ctx, flash, (uint32_t)flash_bytes);
        tiers[0].begin = 0;
        tiers[0].end = (uint32_t)flash_bytes;
        tiers[0].src = &mem_src;
        tiers[0].src_offset = 0;
        tiers[1].begin = (uint32_t)flash_bytes;
        tiers[1].end = (uint32_t)size;
        tiers[1].src = &file_src;
        tiers[1].src_offset = (uint32_t)flash_bytes;
        if (trv_source_tiered_init(&tier_src, &tier_ctx, tiers, flash_bytes < size ? 2 : 1))
            fail("tiers", -1);
        src = &tier_src;
    }

    memset(&rt, 0, sizeof rt);
    if (threads > 1) {
        pool = trv_pool_create(threads);
        trv_pool_runtime(pool, &rt);
    }
    trv_set_runtime(&rt);

    if ((st = trv_model_open(&model, src)) != TRV_OK) fail("model_open", st);
    if ((st = trv_mbert_load(&mb, &model)) != TRV_OK) fail("mbert_load", st);
    rc.L = mb.n_layers;

    /* token ids */
    if (ids_arg) {
        const char *s = ids_arg;
        while (*s && n_seqs < MAX_SEQS) {
            uint32_t n = parse_list(s, ids + T, MAX_TOKENS - T);
            const char *semi = strchr(s, ';');
            lens[n_seqs++] = n;
            T += n;
            if (!semi) break;
            s = semi + 1;
        }
    } else {
        uint32_t b, j;
        for (b = 0; b < batch && n_seqs < MAX_SEQS && T + tokens <= MAX_TOKENS; ++b) {
            for (j = 0; j < tokens; ++j)
                ids[T + j] = 100u + ((j + 31u * b) * 7919u) % (mb.vocab - 100u);
            lens[n_seqs++] = tokens;
            T += tokens;
        }
    }
    if (T == 0) fail("no tokens", 2);
    for (i = 0; i < T; ++i) mask[i] = (mask_arg && i < strlen(mask_arg)) ? mask_arg[i] == '1' : 1;

    if (tile_bytes < trv_mbert_min_tile_bytes(&mb)) tile_bytes = trv_mbert_min_tile_bytes(&mb);
    ws = trv_mbert_workspace_bytes(&mb, T);
    ram = ws + sizeof(float) * (size_t)T * mb.hidden + tile_bytes;
    if (budget >= 0 && ram > (size_t)budget) {
        printf("{\"ok\": false, \"fits\": false, \"tokens\": %u, \"ram_bytes\": %lu, "
               "\"budget\": %ld}\n", T, (unsigned long)ram, budget);
        return 3;
    }

    h = (float *)malloc(sizeof(float) * (size_t)T * mb.hidden);
    arena = malloc(ws);
    tile = malloc(tile_bytes);
    if (!h || !arena || !tile) fail("out of memory", -1);

    {
        uint64_t before_file = trv_filex_bytes_read(fx);
        uint32_t before_reads = trv_filex_reads(fx);
        unsigned long file_bytes = 0;
        for (i = 0; i < model.n_tensors; ++i) {
            trv_tensor t;
            if (trv_model_get(&model, i, NULL, &t) == TRV_OK) file_bytes += t.nbytes;
        }
        before_file = trv_filex_bytes_read(fx);
        before_reads = trv_filex_reads(fx);
        if (flash_bytes) mem_ctx.bytes_read = 0;

        t0 = now();
        st = trv_mbert_forward_batch(&mb, ids, lens, n_seqs, mask, h, &T_out, index_out,
                                     arena, ws, tile, tile_bytes, hook, &rc);
        t1 = now();
        if (st != TRV_OK) fail("forward", st);

        if (rc.dir) {
            char p2[1024];
            FILE *f;
            snprintf(p2, sizeof p2, "%s/final.bin", rc.dir);
            if ((f = fopen(p2, "wb"))) { fwrite(h, sizeof(float), (size_t)T_out * mb.hidden, f); fclose(f); }
            snprintf(p2, sizeof p2, "%s/index.bin", rc.dir);
            if ((f = fopen(p2, "wb"))) { fwrite(index_out, sizeof(uint32_t), T_out, f); fclose(f); }
        }

        printf("{\"ok\": true, \"fits\": true, \"tokens\": %u, \"sequences\": %u, "
               "\"tokens_out\": %u, \"layers\": %u, \"layers_run\": %u, \"hidden\": %u, "
               "\"ram_bytes\": %lu, \"workspace_bytes\": %lu, \"hidden_state_bytes\": %lu, "
               "\"tile_bytes\": %u, \"file_bytes\": %lu, \"bytes_streamed\": %llu, "
               "\"flash_bytes_read\": %u, \"sd_bytes_read\": %llu, \"reads\": %u, "
               "\"macs\": %llu, \"threads\": %u, \"async\": %d, \"sd_mbps\": %.1f, "
               "\"seconds\": %.3f}\n",
               T, n_seqs, T_out, mb.n_layers, rc.layers_run ? rc.layers_run : mb.n_layers,
               mb.hidden, (unsigned long)ram, (unsigned long)ws,
               (unsigned long)(sizeof(float) * (size_t)T * mb.hidden), tile_bytes,
               file_bytes,
               (unsigned long long)(trv_filex_bytes_read(fx) - before_file +
                                    (flash_bytes ? mem_ctx.bytes_read : 0)),
               flash_bytes ? mem_ctx.bytes_read : 0u,
               (unsigned long long)(trv_filex_bytes_read(fx) - before_file),
               trv_filex_reads(fx) - before_reads,
               (unsigned long long)trv_mbert_macs_batch(&mb, lens, n_seqs), threads, async,
               sd_mbps, t1 - t0);
    }

    free(h);
    free(arena);
    free(tile);
    free(flash);
    trv_set_runtime(NULL);
    trv_pool_destroy(pool);
    trv_filex_close(fx);
    return 0;
}
