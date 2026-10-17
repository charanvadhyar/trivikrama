/*
 * trv_laya: answer Laya questions with the C runtime on a PC (the same code
 * that runs on the microcontroller).
 *
 *   trv_laya laya8.trv request.trq [--threads N] [--tile BYTES] [--async]
 *            [--sd-mbps X] [--sd-overhead-ms X] [--budget BYTES]
 *
 * request.trq comes from `trivi laya-prepare`. Prints JSON with one
 * probability row per question (Laya's temperature already applied).
 *
 * Copyright 2026 Vadhyar Sai Charan
 * SPDX-License-Identifier: Apache-2.0
 */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "trivikrama_laya.h"
#include "trivikrama_posix.h"

#define MAX_ROWS 256

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void fail(const char *what, int code)
{
    printf("{\"ok\": false, \"error\": \"%s\", \"code\": %d}\n", what, code);
    exit(1);
}

int main(int argc, char **argv)
{
    static trv_laya ly;
    static trv_laya_row rows[MAX_ROWS];
    const char *path = NULL, *req_path = NULL;
    uint32_t tile_bytes = 262144, threads = 1, T, nk, r, j, kk, i;
    long budget = -1;
    int async = 0;
    int32_t n_rows;
    double sd_mbps = 0, sd_overhead_ms = 0, t0, t1;
    trv_source src;
    trv_filex *fx;
    trv_model model;
    trv_runtime rt;
    trv_pool *pool = NULL;
    trv_status st;
    size_t ws, ram, req_len;
    uint32_t *req;
    float *h, *logits, *probs;
    void *arena, *tile;
    FILE *f;

    for (i = 1; i < (uint32_t)argc; ++i) {
        const char *a = argv[i], *v = (i + 1 < (uint32_t)argc) ? argv[i + 1] : NULL;
        if (!strcmp(a, "--async")) { async = 1; continue; }
        if (a[0] == '-' && a[1] == '-' && !v) fail("missing value", 2);
        if (!strcmp(a, "--tile")) tile_bytes = (uint32_t)atoi(v);
        else if (!strcmp(a, "--threads")) threads = (uint32_t)atoi(v);
        else if (!strcmp(a, "--budget")) budget = atol(v);
        else if (!strcmp(a, "--sd-mbps")) sd_mbps = atof(v);
        else if (!strcmp(a, "--sd-overhead-ms")) sd_overhead_ms = atof(v);
        else if (!path) { path = a; continue; }
        else if (!req_path) { req_path = a; continue; }
        else fail("bad arguments", 2);
        ++i;
    }
    if (!path || !req_path) {
        fprintf(stderr, "usage: trv_laya model.trv request.trq [--threads N] [--tile BYTES] "
                        "[--async] [--sd-mbps X] [--sd-overhead-ms X] [--budget BYTES]\n");
        return 2;
    }

    /* the request: read whole, 4-byte aligned, parsed in place */
    if (!(f = fopen(req_path, "rb"))) fail("open request", -1);
    fseek(f, 0, SEEK_END);
    req_len = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    req = (uint32_t *)malloc(req_len + 4);
    if (!req || fread(req, 1, req_len, f) != req_len) fail("read request", -1);
    fclose(f);
    n_rows = trv_laya_parse_request(req, req_len, rows, MAX_ROWS);
    if (n_rows <= 0) fail("bad request", n_rows);

    fx = trv_filex_open(&src, path, async, sd_mbps * 1e6, sd_overhead_ms * 1e-3);
    if (!fx) fail("open model", -1);
    memset(&rt, 0, sizeof rt);
    if (threads > 1) {
        pool = trv_pool_create(threads);
        trv_pool_runtime(pool, &rt);
    }
    trv_set_runtime(&rt);
    if ((st = trv_model_open(&model, &src)) != TRV_OK) fail("model_open", st);
    if ((st = trv_laya_load(&ly, &model)) != TRV_OK) fail("laya_load", st);

    trv_laya_totals(rows, (uint32_t)n_rows, &T, &nk);
    if (tile_bytes < trv_laya_min_tile_bytes(&ly)) tile_bytes = trv_laya_min_tile_bytes(&ly);
    ws = trv_laya_workspace_bytes(&ly, T, nk, (uint32_t)n_rows);
    ram = ws + sizeof(float) * (size_t)T * ly.hidden + tile_bytes;
    if (budget >= 0 && ram > (size_t)budget) {
        printf("{\"ok\": false, \"fits\": false, \"tokens\": %u, \"ram_bytes\": %lu, "
               "\"budget\": %ld}\n", T, (unsigned long)ram, budget);
        return 3;
    }
    h = (float *)malloc(sizeof(float) * (size_t)T * ly.hidden);
    arena = malloc(ws);
    tile = malloc(tile_bytes);
    logits = (float *)malloc(sizeof(float) * nk);
    probs = (float *)malloc(sizeof(float) * nk);
    if (!h || !arena || !tile || !logits || !probs) fail("out of memory", -1);

    {
        uint64_t before = trv_filex_bytes_read(fx);
        t0 = now();
        st = trv_laya_run(&ly, rows, (uint32_t)n_rows, h, logits, probs, arena, ws, tile,
                          tile_bytes, NULL, NULL);
        t1 = now();
        if (st != TRV_OK) fail("laya_run", st);

        printf("{\"ok\": true, \"rows\": [");
        for (r = 0, kk = 0; r < (uint32_t)n_rows; ++r) {
            printf("%s{\"qtype\": %u, \"tokens\": %u, \"temperature\": %.6g, \"probs\": [",
                   r ? ", " : "", rows[r].qtype, rows[r].n_tokens,
                   trv_laya_temperature(&ly, rows[r].qtype, rows[r].n_options));
            for (j = 0; j < rows[r].n_options; ++j)
                printf("%s%.7g", j ? ", " : "", probs[kk + j]);
            printf("], \"logits\": [");
            for (j = 0; j < rows[r].n_options; ++j)
                printf("%s%.7g", j ? ", " : "", logits[kk + j]);
            printf("]}");
            kk += rows[r].n_options;
        }
        printf("], \"tokens\": %u, \"options\": %u, \"ram_bytes\": %lu, "
               "\"workspace_bytes\": %lu, \"tile_bytes\": %u, \"bytes_streamed\": %llu, "
               "\"threads\": %u, \"async\": %d, \"sd_mbps\": %.1f, \"seconds\": %.3f}\n",
               T, nk, (unsigned long)ram, (unsigned long)ws, tile_bytes,
               (unsigned long long)(trv_filex_bytes_read(fx) - before), threads, async,
               sd_mbps, t1 - t0);
    }

    free(h);
    free(arena);
    free(tile);
    free(logits);
    free(probs);
    free(req);
    trv_set_runtime(NULL);
    trv_pool_destroy(pool);
    trv_filex_close(fx);
    return 0;
}
