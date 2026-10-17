/*
 * stdio weight source for PC builds and tests.
 * Copyright 2026 Vadhyar Sai Charan
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef TRIVIKRAMA_FILE_H
#define TRIVIKRAMA_FILE_H

#include <stdio.h>
#include "trivikrama.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    FILE *f;
    uint32_t bytes_read; /* running total, handy for I/O benchmarks */
    uint32_t n_reads;
} trv_file_ctx;

trv_status trv_source_file_open(trv_source *src, trv_file_ctx *ctx, const char *path);
void trv_source_file_close(trv_file_ctx *ctx);

#ifdef __cplusplus
}
#endif

#endif /* TRIVIKRAMA_FILE_H */
