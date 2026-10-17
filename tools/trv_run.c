/*
 * trv_run: run a ModernBERT .trv model on a PC as a "virtual microcontroller".
 *
 *   trv_run model.trv [--ids 12,99,7 | --tokens 32] [--tile BYTES]
 *                     [--budget BYTES] [--dump DIR]
 *
 * --budget enforces a RAM limit: if the forward pass (activations + scratch +
 * weight tile) does not fit, it refuses to run, exactly like the board would.
 * Prints one JSON line of measurements on stdout.
 *
 * Copyright 2026 Vadhyar Sai Charan
 * SPDX-License-Identifier: Apache-2.0
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "trivikrama.h"
#include "trivikrama_file.h"
#include "trivikrama_modernbert.h"

#define MAX_TOKENS 4096

typedef struct {
    const char *dir;
} dump_ctx;

static void dump_hook(void *user, int32_t layer, const float *h, uint32_t T, uint32_t H)
{
    const dump_ctx *d = (const dump_ctx *)user;
    char path[1024];
    FILE *f;
    if (!d->dir) return;
    if (layer < 0)
        snprintf(path, sizeof path, "%s/emb.bin", d->dir);
    else
        snprintf(path, sizeof path, "%s/layer%03d.bin", d->dir, (int)layer);
    f = fopen(path, "wb");
    if (!f) return;
    fwrite(h, sizeof(float), (size_t)T * H, f);
    fclose(f);
}

static uint32_t parse_ids(const char *s, uint32_t *ids)
{
    uint32_t n = 0;
    while (*s && n < MAX_TOKENS) {
        char *end;
        ids[n++] = (uint32_t)strtoul(s, &end, 10);
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
    static uint32_t ids[MAX_TOKENS];
    const char *path = NULL;
    uint32_t T = 0, tile_bytes = 4096, i;
    long budget = -1;
    dump_ctx dump = {NULL};
    trv_source src;
    trv_file_ctx fctx;
    trv_model model;
    trv_mbert mb;
    trv_status st;
    size_t ws, ram, weights = 0;
    float *h;
    void *arena, *tile;
    clock_t t0, t1;

    for (i = 1; i < (uint32_t)argc; ++i) {
        if (!strcmp(argv[i], "--ids") && i + 1 < (uint32_t)argc)
            T = parse_ids(argv[++i], ids);
        else if (!strcmp(argv[i], "--tokens") && i + 1 < (uint32_t)argc)
            T = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tile") && i + 1 < (uint32_t)argc)
            tile_bytes = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--budget") && i + 1 < (uint32_t)argc)
            budget = atol(argv[++i]);
        else if (!strcmp(argv[i], "--dump") && i + 1 < (uint32_t)argc)
            dump.dir = argv[++i];
        else if (!path)
            path = argv[i];
        else
            fail("bad arguments", 2);
    }
    if (!path || T == 0 || T > MAX_TOKENS) {
        fprintf(stderr, "usage: trv_run model.trv (--ids a,b,c | --tokens N) "
                        "[--tile BYTES] [--budget BYTES] [--dump DIR]\n");
        return 2;
    }

    if ((st = trv_source_file_open(&src, &fctx, path)) != TRV_OK) fail("open", st);
    if ((st = trv_model_open(&model, &src)) != TRV_OK) fail("model_open", st);
    if ((st = trv_mbert_load(&mb, &model)) != TRV_OK) fail("mbert_load", st);

    /* --tokens N without ids: deterministic ids that avoid special tokens */
    if (ids[0] == 0 && ids[T - 1] == 0)
        for (i = 0; i < T; ++i) ids[i] = 100u + (i * 7919u) % (mb.vocab - 100u);

    for (i = 0; i < model.n_tensors; ++i) {
        trv_tensor t;
        if (trv_model_get(&model, i, NULL, &t) == TRV_OK) weights += t.nbytes;
    }

    ws = trv_mbert_workspace_bytes(&mb, T);
    ram = ws + sizeof(float) * (size_t)T * mb.hidden + tile_bytes;
    if (budget >= 0 && ram > (size_t)budget) {
        printf("{\"ok\": false, \"fits\": false, \"tokens\": %u, \"ram_bytes\": %lu, "
               "\"budget\": %ld}\n",
               T, (unsigned long)ram, budget);
        return 3;
    }

    h = (float *)malloc(sizeof(float) * (size_t)T * mb.hidden);
    arena = malloc(ws);
    tile = malloc(tile_bytes);
    if (!h || !arena || !tile) fail("out of memory", -1);

    fctx.bytes_read = 0;
    fctx.n_reads = 0;
    t0 = clock();
    st = trv_mbert_forward(&mb, ids, NULL, T, h, arena, ws, tile, tile_bytes, dump_hook,
                           &dump);
    t1 = clock();
    if (st != TRV_OK) fail("forward", st);

    printf("{\"ok\": true, \"fits\": true, \"tokens\": %u, \"layers\": %u, \"hidden\": %u, "
           "\"ram_bytes\": %lu, \"workspace_bytes\": %lu, \"hidden_state_bytes\": %lu, "
           "\"tile_bytes\": %u, \"file_bytes\": %lu, \"bytes_streamed\": %lu, "
           "\"reads\": %u, \"macs\": %llu, \"seconds\": %.3f}\n",
           T, mb.n_layers, mb.hidden, (unsigned long)ram, (unsigned long)ws,
           (unsigned long)(sizeof(float) * (size_t)T * mb.hidden), tile_bytes,
           (unsigned long)weights, (unsigned long)fctx.bytes_read, fctx.n_reads,
           (unsigned long long)trv_mbert_macs(&mb, T),
           (double)(t1 - t0) / CLOCKS_PER_SEC);

    free(h);
    free(arena);
    free(tile);
    trv_source_file_close(&fctx);
    return 0;
}
