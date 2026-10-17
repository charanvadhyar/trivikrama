/*
 * Trivikrama .trv reader: works over any trv_source, uses no heap.
 * Copyright 2026 Vadhyar Sai Charan
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trivikrama.h"

#include <string.h>

static uint32_t rd_u32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static int src_read(const trv_model *m, uint32_t off, void *dst, uint32_t n)
{
    return m->src->read(m->src->ctx, off, dst, n);
}

trv_status trv_model_open(trv_model *m, const trv_source *src)
{
    uint8_t h[16];
    if (!m || !src || !src->read) return TRV_ERR_ARG;
    m->src = src;
    if (src->read(src->ctx, 0, h, 16) != 0) return TRV_ERR_IO;
    if (memcmp(h, "TRVK", 4) != 0) return TRV_ERR_FORMAT;
    if (rd_u32(h + 4) != TRV_FORMAT_VERSION) return TRV_ERR_FORMAT;
    m->n_tensors = rd_u32(h + 8);
    m->table_offset = rd_u32(h + 12);
    return TRV_OK;
}

/* Read the table entry at *pos and advance *pos past it. */
static trv_status read_entry(const trv_model *m, uint32_t *pos, char *name,
                             trv_tensor *t)
{
    uint8_t b[8 + 4 * TRV_MAX_DIMS];
    uint32_t len, i;

    if (src_read(m, *pos, b, 2) != 0) return TRV_ERR_IO;
    len = (uint32_t)b[0] | ((uint32_t)b[1] << 8);
    if (len > TRV_MAX_NAME) return TRV_ERR_FORMAT;
    *pos += 2;
    if (src_read(m, *pos, name, len) != 0) return TRV_ERR_IO;
    name[len] = '\0';
    *pos += len;

    if (src_read(m, *pos, b, 2) != 0) return TRV_ERR_IO;
    t->dtype = b[0];
    t->ndim = b[1];
    if (t->ndim > TRV_MAX_DIMS) return TRV_ERR_FORMAT;
    *pos += 2;

    if (src_read(m, *pos, b, 4u * t->ndim + 8u) != 0) return TRV_ERR_IO;
    for (i = 0; i < TRV_MAX_DIMS; ++i)
        t->dims[i] = (i < t->ndim) ? rd_u32(b + 4 * i) : 1u;
    t->offset = rd_u32(b + 4u * t->ndim);
    t->nbytes = rd_u32(b + 4u * t->ndim + 4u);
    *pos += 4u * t->ndim + 8u;
    return TRV_OK;
}

trv_status trv_model_get(const trv_model *m, uint32_t index, char *name_out,
                         trv_tensor *out)
{
    char name[TRV_MAX_NAME + 1];
    uint32_t pos, i;
    if (!m || !out || index >= m->n_tensors) return TRV_ERR_ARG;
    pos = m->table_offset;
    for (i = 0; i <= index; ++i) {
        trv_status st = read_entry(m, &pos, name, out);
        if (st != TRV_OK) return st;
    }
    if (name_out) memcpy(name_out, name, strlen(name) + 1);
    return TRV_OK;
}

trv_status trv_model_find(const trv_model *m, const char *name, trv_tensor *out)
{
    char cur[TRV_MAX_NAME + 1];
    uint32_t pos, i;
    if (!m || !name || !out) return TRV_ERR_ARG;
    pos = m->table_offset;
    for (i = 0; i < m->n_tensors; ++i) {
        trv_status st = read_entry(m, &pos, cur, out);
        if (st != TRV_OK) return st;
        if (strcmp(cur, name) == 0) return TRV_OK;
    }
    return TRV_ERR_NOT_FOUND;
}

uint32_t trv_tensor_numel(const trv_tensor *t)
{
    uint32_t n = 1, i;
    for (i = 0; i < t->ndim; ++i) n *= t->dims[i];
    return n;
}

trv_status trv_tensor_read(const trv_model *m, const trv_tensor *t,
                           uint32_t byte_off, void *dst, uint32_t n)
{
    if (!m || !t || !dst) return TRV_ERR_ARG;
    if (byte_off > t->nbytes || n > t->nbytes - byte_off) return TRV_ERR_ARG;
    return src_read(m, t->offset + byte_off, dst, n) == 0 ? TRV_OK : TRV_ERR_IO;
}

trv_status trv_tensor_read_f32(const trv_model *m, const trv_tensor *t,
                               uint32_t elem_off, float *dst, uint32_t count)
{
    if (!t || t->dtype != TRV_DTYPE_F32) return TRV_ERR_ARG;
    return trv_tensor_read(m, t, 4u * elem_off, dst, 4u * count);
}

trv_status trv_tensor_read_q8_row(const trv_model *m, const trv_tensor *t,
                                  uint32_t row, float *dst, int8_t *scratch)
{
    uint32_t N, K, k;
    float s;
    trv_status st;
    if (!t || t->dtype != TRV_DTYPE_Q8_ROWS || t->ndim != 2) return TRV_ERR_ARG;
    N = t->dims[0];
    K = t->dims[1];
    if (row >= N) return TRV_ERR_ARG;
    st = trv_tensor_read(m, t, 4u * row, &s, 4);
    if (st != TRV_OK) return st;
    st = trv_tensor_read(m, t, 4u * N + row * K, scratch, K);
    if (st != TRV_OK) return st;
    for (k = 0; k < K; ++k) dst[k] = (float)scratch[k] * s;
    return TRV_OK;
}
