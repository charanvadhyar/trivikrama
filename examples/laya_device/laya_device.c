/*
 * Laya on a device: the smallest complete program.
 *
 * Reads the model (laya8.trv) and a request (request.trq) from storage, runs
 * the encoder and the decision head, and prints one line per question.
 * Plain C99 + stdio, so the same file builds on a PC and on any board whose
 * SDK maps the SD card to fopen() (ESP-IDF does, with the card mounted at
 * /sdcard; see README.md next to this file).
 *
 *   PC:      make build/laya_device && build/laya_device laya8.trv request.trq
 *   ESP-IDF: call laya_device_main("/sdcard/laya8.trv", "/sdcard/request.trq")
 *            from app_main()
 *
 * Copyright 2026 Vadhyar Sai Charan
 * SPDX-License-Identifier: Apache-2.0
 */
#include <stdio.h>
#include <stdlib.h>

#include "trivikrama_file.h"
#include "trivikrama_laya.h"

#ifndef LAYA_MAX_ROWS
#define LAYA_MAX_ROWS 32
#endif
#ifndef LAYA_TILE_BYTES
#define LAYA_TILE_BYTES (64u * 1024u)
#endif

/* Progress: one dot per encoder layer, so a slow board shows it is alive. */
static void progress(void *user, int32_t layer, const float *h, uint32_t T, uint32_t H,
                     trv_mbert_ctl *ctl)
{
    (void)user; (void)h; (void)T; (void)H; (void)ctl;
    if (layer >= 0) {
        putchar('.');
        fflush(stdout);
    }
}

static void *load_file(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    void *buf = NULL;
    long n;
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) == 0 && (n = ftell(f)) > 0 && fseek(f, 0, SEEK_SET) == 0) {
        buf = malloc((size_t)n + 4); /* malloc is suitably aligned for the u32 fields */
        if (buf && fread(buf, 1, (size_t)n, f) != (size_t)n) {
            free(buf);
            buf = NULL;
        }
        *len = (size_t)n;
    }
    fclose(f);
    return buf;
}

int laya_device_main(const char *model_path, const char *request_path)
{
    static trv_laya ly;                     /* ~1 KB of offsets; static keeps the stack small */
    static trv_laya_row rows[LAYA_MAX_ROWS];
    trv_source src;
    trv_file_ctx fctx = {0};
    trv_model model;
    trv_status st;
    size_t req_len = 0, ws;
    uint32_t T, n_opt, tile_bytes, r, j, k;
    int32_t n_rows;
    void *req = NULL, *arena = NULL, *tile = NULL;
    float *h = NULL, *probs = NULL;
    int ret = 1;

    if (!(req = load_file(request_path, &req_len))) {
        printf("cannot read %s\n", request_path);
        goto done;
    }
    n_rows = trv_laya_parse_request(req, req_len, rows, LAYA_MAX_ROWS);
    if (n_rows <= 0) {
        printf("bad request file (%d)\n", (int)n_rows);
        goto done;
    }
    if ((st = trv_source_file_open(&src, &fctx, model_path)) != TRV_OK ||
        (st = trv_model_open(&model, &src)) != TRV_OK ||
        (st = trv_laya_load(&ly, &model)) != TRV_OK) {
        printf("cannot load %s (%d)\n", model_path, (int)st);
        goto done;
    }

    /* RAM: hidden states + workspace + one weight tile; nothing else */
    trv_laya_totals(rows, (uint32_t)n_rows, &T, &n_opt);
    ws = trv_laya_workspace_bytes(&ly, T, n_opt, (uint32_t)n_rows);
    tile_bytes = LAYA_TILE_BYTES;
    if (tile_bytes < trv_laya_min_tile_bytes(&ly)) tile_bytes = trv_laya_min_tile_bytes(&ly);
    printf("%d questions, %u tokens, %lu KB RAM\n", (int)n_rows, (unsigned)T,
           (unsigned long)((sizeof(float) * (size_t)T * ly.hidden + ws + tile_bytes) / 1024));
    h = (float *)malloc(sizeof(float) * (size_t)T * ly.hidden);
    arena = malloc(ws);
    tile = malloc(tile_bytes);
    probs = (float *)malloc(sizeof(float) * n_opt);
    if (!h || !arena || !tile || !probs) {
        printf("out of memory\n");
        goto done;
    }

    st = trv_laya_run(&ly, rows, (uint32_t)n_rows, h, NULL, probs, arena, ws, tile, tile_bytes,
                      progress, NULL);
    printf("\n");
    if (st != TRV_OK) {
        printf("run failed (%d)\n", (int)st);
        goto done;
    }

    /* Options come in the slot order of the request; `trivi laya-decode` maps
     * them back to names. Here: the raw numbers, enough to act on. */
    for (r = 0, k = 0; r < (uint32_t)n_rows; ++r) {
        const float *p = probs + k;
        uint32_t best = 0;
        float score = 0.0f;
        for (j = 0; j < rows[r].n_options; ++j) {
            if (p[j] > p[best]) best = j;
            score += (float)j * p[j];
        }
        if (rows[r].qtype == TRV_LAYA_NOUL)
            printf("q%u yes/no: %s (p_yes %.3f)\n", (unsigned)r, p[1] > 0.5f ? "yes" : "no", p[1]);
        else if (rows[r].qtype == TRV_LAYA_CHOICE)
            printf("q%u choice: option %u (p %.3f)\n", (unsigned)r, (unsigned)best, p[best]);
        else
            printf("q%u score: %.2f\n", (unsigned)r, score);
        k += rows[r].n_options;
    }
    ret = 0;

done: /* safe to call again: everything is freed on every path */
    free(h);
    free(arena);
    free(tile);
    free(probs);
    free(req);
    trv_source_file_close(&fctx);
    return ret;
}

#ifndef LAYA_DEVICE_NO_MAIN
int main(int argc, char **argv)
{
    if (argc < 3) {
        printf("usage: %s laya8.trv request.trq\n", argv[0]);
        return 2;
    }
    return laya_device_main(argv[1], argv[2]);
}
#endif
