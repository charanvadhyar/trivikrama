/*
 * trv_tokenize: run the device tokenizer on the PC (for tests).
 *
 *   trv_tokenize questions.tlq cases.bin          one line of ids per case
 *   trv_tokenize questions.tlq cases.bin --rows   JSON rows per case, as the
 *                                                 device builds them
 * cases.bin: u32 count, then per case u32 length + UTF-8 bytes.
 *
 * Copyright 2026 Vadhyar Sai Charan
 * SPDX-License-Identifier: Apache-2.0
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "trivikrama_laya_text.h"

static void *read_all(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    void *b;
    long n;
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    fseek(f, 0, SEEK_SET);
    b = malloc((size_t)n + 4);
    if (b && fread(b, 1, (size_t)n, f) != (size_t)n) {
        free(b);
        b = NULL;
    }
    fclose(f);
    *len = (size_t)n;
    return b;
}

int main(int argc, char **argv)
{
    static trv_laya_text lt;
    static trv_laya_row rows[TRV_LAYA_MAX_QUESTIONS];
    size_t tl, cl, pos = 4;
    uint32_t *tlq, n_cases, c;
    uint8_t *cases;
    int rows_mode = argc > 3 && !strcmp(argv[3], "--rows");
    trv_status st;

    if (argc < 3) {
        fprintf(stderr, "usage: trv_tokenize questions.tlq cases.bin [--rows]\n");
        return 2;
    }
    tlq = (uint32_t *)read_all(argv[1], &tl);
    cases = (uint8_t *)read_all(argv[2], &cl);
    if (!tlq || !cases || cl < 4) return 1;
    if ((st = trv_laya_text_parse(&lt, tlq, tl)) != TRV_OK) {
        fprintf(stderr, "bad .tlq (%d)\n", (int)st);
        return 1;
    }
    memcpy(&n_cases, cases, 4);
    for (c = 0; c < n_cases; ++c) {
        uint32_t len, max_ids, *ids, *work, *idb, r, j;
        size_t ww;
        int32_t n;
        memcpy(&len, cases + pos, 4);
        pos += 4;
        max_ids = rows_mode ? trv_laya_text_max_room(&lt) : 4 * len + 16;
        ww = trv_laya_tokenize_work_words(len);
        ids = (uint32_t *)malloc(sizeof(uint32_t) * (max_ids + 1));
        work = (uint32_t *)malloc(sizeof(uint32_t) * ww);
        n = trv_laya_tokenize(&lt, (const char *)cases + pos, len, ids, max_ids, work, ww);
        pos += len;
        if (n < 0) {
            printf("ERROR %d\n", (int)n);
            return 1;
        }
        if (!rows_mode) {
            for (j = 0; j < (uint32_t)n; ++j) printf("%s%u", j ? "," : "", ids[j]);
            printf("\n");
        } else {
            size_t w = trv_laya_text_row_words(&lt, (uint32_t)n);
            int32_t nr;
            idb = (uint32_t *)malloc(sizeof(uint32_t) * w);
            nr = trv_laya_text_rows(&lt, ids, (uint32_t)n, idb, w, rows);
            printf("[");
            for (r = 0; r < (uint32_t)nr; ++r) {
                printf("%s{\"qtype\": %u, \"ids\": [", r ? ", " : "", rows[r].qtype);
                for (j = 0; j < rows[r].n_tokens; ++j) printf("%s%u", j ? "," : "", rows[r].ids[j]);
                printf("], \"markers\": [");
                for (j = 0; j < rows[r].n_options; ++j)
                    printf("%s%u", j ? "," : "", rows[r].markers[j]);
                printf("]}");
            }
            printf("]\n");
            free(idb);
        }
        free(ids);
        free(work);
    }
    free(tlq);
    free(cases);
    return 0;
}
