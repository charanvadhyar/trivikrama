/*
 * stdio weight source for PC builds and tests.
 * On a microcontroller, write the same ~10 lines over your SD/flash driver.
 * Copyright 2026 Vadhyar Sai Charan
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trivikrama_file.h"

static int file_read(void *vctx, uint32_t offset, void *dst, uint32_t n)
{
    trv_file_ctx *ctx = (trv_file_ctx *)vctx;
    if (fseek(ctx->f, (long)offset, SEEK_SET) != 0) return -1;
    if (fread(dst, 1, n, ctx->f) != n) return -1;
    ctx->bytes_read += n;
    ctx->n_reads += 1;
    return 0;
}

trv_status trv_source_file_open(trv_source *src, trv_file_ctx *ctx, const char *path)
{
    if (!src || !ctx || !path) return TRV_ERR_ARG;
    ctx->f = fopen(path, "rb");
    if (!ctx->f) return TRV_ERR_IO;
    ctx->bytes_read = 0;
    ctx->n_reads = 0;
    src->read = file_read;
    src->ctx = ctx;
    return TRV_OK;
}

void trv_source_file_close(trv_file_ctx *ctx)
{
    if (ctx && ctx->f) {
        fclose(ctx->f);
        ctx->f = NULL;
    }
}
