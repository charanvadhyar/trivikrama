/*
 * Trivikrama: run full-size transformer encoders on microcontrollers by
 * streaming weights from external storage, one tile at a time.
 *
 * Copyright 2026 Vadhyar Sai Charan
 * SPDX-License-Identifier: Apache-2.0
 *
 * Design rules for everything in this header:
 *   - plain C99, no dynamic allocation, no OS calls; the caller owns all buffers
 *   - row-major tensors; "ld" arguments are row strides in elements
 *   - weights never need to fit in RAM: they are read through a trv_source
 *   - all targets are little-endian (ESP8266, ESP32, RP2040/RP2350, x86, ARM)
 */
#ifndef TRIVIKRAMA_H
#define TRIVIKRAMA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TRV_VERSION_STRING "0.1.0"
#define TRV_FORMAT_VERSION 1u

typedef enum {
    TRV_OK = 0,
    TRV_ERR_IO = -1,        /* the weight source failed to read */
    TRV_ERR_FORMAT = -2,    /* not a valid .trv file */
    TRV_ERR_NOT_FOUND = -3, /* tensor name not in the file */
    TRV_ERR_ARG = -4,       /* bad argument (shape, dtype, range) */
    TRV_ERR_BUFFER = -5     /* scratch buffer too small */
} trv_status;

/* ------------------------------------------------------------------------ */
/* Weight source: anything that can read bytes at an offset.                 */
/* SD card, SPI flash, memory-mapped flash, a file on a PC, a network...     */
/* ------------------------------------------------------------------------ */

typedef struct {
    /* Read n bytes starting at offset into dst. Return 0 on success. */
    int (*read)(void *ctx, uint32_t offset, void *dst, uint32_t n);
    void *ctx;
} trv_source;

/* ------------------------------------------------------------------------ */
/* .trv model file                                                           */
/*                                                                           */
/*   header (16 bytes): "TRVK" | u32 version | u32 n_tensors | u32 table_off */
/*   table entry:       u16 name_len | name | u8 dtype | u8 ndim |           */
/*                      u32 dims[ndim] | u32 data_offset | u32 data_nbytes   */
/*   TRV_DTYPE_F32:     raw float32 values                                   */
/*   TRV_DTYPE_Q8_ROWS: 2-D [N, K]; N float32 row scales, then N*K int8      */
/*                      (value = int8 * row_scale, symmetric, per row)       */
/* ------------------------------------------------------------------------ */

enum { TRV_DTYPE_F32 = 0, TRV_DTYPE_Q8_ROWS = 1 };

#define TRV_MAX_DIMS 4
#define TRV_MAX_NAME 96

typedef struct {
    uint8_t dtype;
    uint8_t ndim;
    uint32_t dims[TRV_MAX_DIMS];
    uint32_t offset; /* absolute byte offset of the data in the source */
    uint32_t nbytes;
} trv_tensor;

typedef struct {
    const trv_source *src;
    uint32_t n_tensors;
    uint32_t table_offset;
} trv_model;

trv_status trv_model_open(trv_model *m, const trv_source *src);

/* Look up a tensor by name. Scans the table through the source, so it costs
 * a few small reads but no RAM. Do lookups once, keep the trv_tensor. */
trv_status trv_model_find(const trv_model *m, const char *name, trv_tensor *out);

/* Get the index-th table entry. name_out must hold TRV_MAX_NAME + 1 chars
 * (may be NULL). */
trv_status trv_model_get(const trv_model *m, uint32_t index, char *name_out,
                         trv_tensor *out);

/* Walk the table in one pass. Start with *pos = m->table_offset and call
 * n_tensors times; name_out must hold TRV_MAX_NAME + 1 chars. */
trv_status trv_model_next(const trv_model *m, uint32_t *pos, char *name_out,
                          trv_tensor *out);

uint32_t trv_tensor_numel(const trv_tensor *t);

/* Read raw bytes [byte_off, byte_off + n) of a tensor's data. */
trv_status trv_tensor_read(const trv_model *m, const trv_tensor *t,
                           uint32_t byte_off, void *dst, uint32_t n);

/* Read count float32 values starting at element elem_off of an F32 tensor. */
trv_status trv_tensor_read_f32(const trv_model *m, const trv_tensor *t,
                               uint32_t elem_off, float *dst, uint32_t count);

/* Read one de-quantized row of a Q8_ROWS tensor (e.g. an embedding lookup).
 * scratch must hold K bytes. */
trv_status trv_tensor_read_q8_row(const trv_model *m, const trv_tensor *t,
                                  uint32_t row, float *dst, int8_t *scratch);

/* ------------------------------------------------------------------------ */
/* Kernels                                                                   */
/* ------------------------------------------------------------------------ */

/* Symmetric per-token int8 quantization of activations.
 * x: [T x K] with row stride ldx.  xq: [T x K] contiguous.  xscale: [T]. */
void trv_quantize_act_q8(const float *x, uint32_t T, uint32_t K, uint32_t ldx,
                         int8_t *xq, float *xscale);

/* Group-wise version: every run of `group` values in a row gets its own scale.
 * K must be a multiple of group.
 *   xq      [T x K] written with row stride ldq
 *   xscale  [T x K/group] written with row stride lds
 * Writing into a column slice (xq + col, xscale + col/group) lets a caller
 * quantize a wide activation piece by piece, e.g. one attention head or one
 * FFN chunk at a time, without ever holding the whole row in float. */
void trv_quantize_act_q8g(const float *x, uint32_t T, uint32_t K, uint32_t ldx,
                          uint32_t group, int8_t *xq, uint32_t ldq,
                          float *xscale, uint32_t lds);

/* LayerNorm fused with group-wise quantization: reads x, writes only the int8
 * result, so the normalized activations never need a float buffer.
 * beta may be NULL. xq is [T x D] contiguous, xscale is [T x D/group]. */
void trv_layernorm_quantize(const float *x, uint32_t T, uint32_t D, uint32_t ldx,
                            const float *gamma, const float *beta, float eps,
                            uint32_t group, int8_t *xq, float *xscale);

/* Streamed linear layer: y[t][n - n_begin] = sum_k x[t][k] * W[n][k] + bias[n]
 * for output columns n in [n_begin, n_end).
 *
 *   w         Q8_ROWS tensor [N x K], read tile by tile through the model source
 *   xq,xscale quantized input from trv_quantize_act_q8 ([T x K], [T])
 *   bias      [N] floats, or NULL
 *   y         output, row stride ldy (pass y + col for a sub-range)
 *   tile_buf  scratch, 4-byte aligned; holds floor(tile_bytes / (K + 4)) rows
 *
 * Each weight tile is applied to ALL T tokens before the next tile is read,
 * so every weight is read from storage exactly once per call.
 * Use the column range to bound RAM: e.g. compute a wide FFN in chunks. */
trv_status trv_linear_q8_stream(const trv_model *m, const trv_tensor *w,
                                const int8_t *xq, const float *xscale, uint32_t T,
                                const float *bias, uint32_t n_begin, uint32_t n_end,
                                float *y, uint32_t ldy,
                                void *tile_buf, uint32_t tile_bytes);

/* General form of the streamed linear layer.
 *   group       input quantization group size (xscale is [T x K/group]);
 *               pass K for per-token scales
 *   accumulate  0: y = Wx + b    1: y += Wx + b   (adds straight into a
 *               residual stream, so no extra output buffer is needed) */
trv_status trv_linear_q8g_stream(const trv_model *m, const trv_tensor *w,
                                 const int8_t *xq, const float *xscale,
                                 uint32_t group, uint32_t T,
                                 const float *bias, uint32_t n_begin, uint32_t n_end,
                                 float *y, uint32_t ldy, int accumulate,
                                 void *tile_buf, uint32_t tile_bytes);

/* In-place LayerNorm over the last dimension. beta may be NULL (ModernBERT). */
void trv_layernorm(float *x, uint32_t T, uint32_t D, uint32_t ldx,
                   const float *gamma, const float *beta, float eps);

/* Exact (erf) GELU, in place. */
void trv_gelu(float *x, uint32_t n);

/* GeGLU as in ModernBERT: in = [a | gate] with width 2*I,
 * out[t][i] = gelu(a[t][i]) * gate[t][i].
 * out may alias in (pass the same pointer and stride) for in-place use. */
void trv_geglu(const float *in, uint32_t T, uint32_t I, uint32_t ldin,
               float *out, uint32_t ldout);

/* Rotary position embedding (rotate-half convention, as in HF Llama/ModernBERT),
 * in place, applied to n_heads heads of size head_dim laid out side by side
 * in each row. Token t has position pos0 + t. */
void trv_rope(float *x, uint32_t T, uint32_t n_heads, uint32_t head_dim,
              uint32_t ldx, uint32_t pos0, float base);

/* One attention head with online (streaming) softmax: never builds the
 * T x T score matrix; extra memory is zero beyond the output row.
 *
 *   q, k, v      point at this head's first element; row stride ld
 *   half_window  < 0 for global attention; otherwise token i attends to
 *                j with |i - j| <= half_window (ModernBERT local layers: 64)
 *   key_mask     [T] with 1 = real token, 0 = padding; or NULL
 *   out          [T x d] with row stride ldo; must not alias q, k or v
 *
 * Scores are scaled by 1/sqrt(d). Queries with no visible keys output 0. */
void trv_attention_head(const float *q, const float *k, const float *v,
                        uint32_t ld, uint32_t T, uint32_t d,
                        int32_t half_window, const uint8_t *key_mask,
                        float *out, uint32_t ldo);

#ifdef __cplusplus
}
#endif

#endif /* TRIVIKRAMA_H */
