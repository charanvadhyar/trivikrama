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

#define TRV_VERSION_STRING "0.5.0"
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
    /* Optional (leave NULL if unsupported): start a read and return at once,
     * e.g. with DMA or a second core. wait() blocks until every read started
     * so far has finished and returns 0 if all succeeded. When both are set,
     * the linear kernel double-buffers: it fetches the next weight tile while
     * computing the current one. */
    int (*read_async)(void *ctx, uint32_t offset, void *dst, uint32_t n);
    int (*wait)(void *ctx);
} trv_source;

/* Plain memory source: weights in RAM, PSRAM or memory-mapped flash. */
typedef struct {
    const uint8_t *base;
    uint32_t size;
    uint32_t bytes_read;
} trv_mem_ctx;

trv_status trv_source_mem_init(trv_source *src, trv_mem_ctx *ctx, const void *base,
                               uint32_t size);

/* Storage tiers: route byte ranges of one model to different sources, e.g.
 * the first layers from internal flash and the rest from the SD card.
 * Global offset `off` in [begin, end) is read from `src` at
 * src_offset + (off - begin). Tiers must not overlap. */
typedef struct {
    uint32_t begin, end;
    const trv_source *src;
    uint32_t src_offset;
    uint32_t bytes_read; /* per-tier counter, for benchmarks */
} trv_tier;

typedef struct {
    trv_tier *tiers;
    uint32_t n_tiers;
} trv_tiered_ctx;

trv_status trv_source_tiered_init(trv_source *src, trv_tiered_ctx *ctx, trv_tier *tiers,
                                  uint32_t n_tiers);

/* ------------------------------------------------------------------------ */
/* Runtime hooks: plug in a SIMD dot product and a second core.              */
/* All optional; with nothing set, everything runs in portable C on one core.*/
/* Outputs are bit-identical whatever you plug in (as long as the dot product */
/* is exact, which integer dot products always are).                         */
/* ------------------------------------------------------------------------ */

/* acc[j] = sum_{k in chunk j} x[k] * w[k], for chunks of `chunk` values,
 * j = 0 .. K/chunk - 1. Replace with a SIMD version (e.g. ESP32-S3 PIE). */
typedef void (*trv_dot_groups_fn)(const int8_t *x, const int8_t *w, uint32_t K,
                                  uint32_t chunk, int32_t *acc);

/* Run fn(arg, worker, n_workers) for worker = 0 .. n_workers-1, possibly in
 * parallel, and return when all have finished. */
typedef void (*trv_task_fn)(void *arg, uint32_t worker, uint32_t n_workers);
typedef void (*trv_parallel_fn)(void *ctx, trv_task_fn fn, void *arg, uint32_t n_workers);

typedef struct {
    trv_dot_groups_fn dot_groups; /* NULL = portable C */
    trv_parallel_fn parallel;     /* NULL = single core */
    void *parallel_ctx;
    uint32_t n_workers;           /* 0 or 1 = single core */
} trv_runtime;

/* Install runtime hooks (copied). Pass NULL to restore the defaults. */
void trv_set_runtime(const trv_runtime *rt);
const trv_runtime *trv_get_runtime(void);

/* ------------------------------------------------------------------------ */
/* .trv model file                                                           */
/*                                                                           */
/*   header (16 bytes): "TRVK" | u32 version | u32 n_tensors | u32 table_off */
/*   table entry:       u16 name_len | name | u8 dtype | u8 ndim |           */
/*                      u32 dims[ndim] | u32 data_offset | u32 data_nbytes   */
/*   TRV_DTYPE_F32:     raw float32 values                                   */
/*   TRV_DTYPE_Q8_ROWS: 2-D [N, K]; N float32 row scales, then N*K int8      */
/*                      (value = int8 * row_scale, symmetric, per row)       */
/*   TRV_DTYPE_Q4_G32:  2-D [N, K], K % 32 == 0; rows stored one after the   */
/*                      other, each row = K/32 fp16 group scales followed by */
/*                      K/2 bytes of 4-bit values (low nibble first, stored  */
/*                      as q + 8, q in [-7, 7]); value = q * group_scale     */
/* ------------------------------------------------------------------------ */

enum { TRV_DTYPE_F32 = 0, TRV_DTYPE_Q8_ROWS = 1, TRV_DTYPE_Q4_G32 = 2 };

#define TRV_Q4_GROUP 32u
/* bytes of one stored row of a Q4_G32 tensor */
#define TRV_Q4_ROW_BYTES(K) (2u * ((K) / TRV_Q4_GROUP) + (K) / 2u)

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
 *   w           Q8_ROWS or Q4_G32 weights
 *   group       input quantization group size (xscale is [T x K/group]);
 *               pass K for per-token scales. For Q4_G32 weights, group and
 *               32 must divide one another.
 *   accumulate  0: y = Wx + b    1: y += Wx + b   (adds straight into a
 *               residual stream, so no extra output buffer is needed)
 *   tile_buf    holds per-worker scratch (trv_linear_scratch_bytes) plus at
 *               least one weight row; with an async source it is split into
 *               two halves so the next tile loads during compute.
 * Uses the installed trv_runtime (SIMD dot product, worker cores). */
trv_status trv_linear_stream(const trv_model *m, const trv_tensor *w,
                             const int8_t *xq, const float *xscale,
                             uint32_t group, uint32_t T,
                             const float *bias, uint32_t n_begin, uint32_t n_end,
                             float *y, uint32_t ldy, int accumulate,
                             void *tile_buf, uint32_t tile_bytes);

/* Same as trv_linear_stream (kept for compatibility). */
trv_status trv_linear_q8g_stream(const trv_model *m, const trv_tensor *w,
                                 const int8_t *xq, const float *xscale,
                                 uint32_t group, uint32_t T,
                                 const float *bias, uint32_t n_begin, uint32_t n_end,
                                 float *y, uint32_t ldy, int accumulate,
                                 void *tile_buf, uint32_t tile_bytes);

/* Scratch bytes trv_linear_stream reserves at the start of tile_buf for a
 * weight tensor of row length K and dtype, for the current runtime. */
uint32_t trv_linear_scratch_bytes(uint8_t dtype, uint32_t K, uint32_t group);

/* Bytes of one stored weight row (what a tile is made of). */
uint32_t trv_weight_row_bytes(uint8_t dtype, uint32_t K);

/* Cross-call prefetch (only does anything with an async source).
 *
 * A model calls the linear layer many times in a row (one head, one FFN
 * chunk...). Each call alone is often a single tile, so there is nothing to
 * overlap inside it. trv_linear_set_next() tells the *next* call's weights in
 * advance: while the current call computes its last tile, the first tile of
 * the next call is already loading. The next call must use the same tile_buf.
 *
 * trv_linear_reserve_scratch() fixes the scratch size at the start of
 * tile_buf (use the largest any call needs), so every call splits the buffer
 * at the same place and a prefetched tile is where the next call expects it.
 *
 * Before doing a plain (sync) read of the source yourself while a prefetch
 * may be in flight, call trv_linear_drain(). */
void trv_linear_set_next(const trv_tensor *w, uint32_t group, uint32_t n_begin,
                         uint32_t n_end);
void trv_linear_reserve_scratch(uint32_t bytes);
void trv_linear_drain(const trv_source *src);

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

/* Same, with an explicit position per token (needed after token pruning or
 * when several sequences are packed into one batch). */
void trv_rope_pos(float *x, uint32_t T, uint32_t n_heads, uint32_t head_dim,
                  uint32_t ldx, const uint32_t *pos, float base);

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

/* Same, with explicit token positions for the window test |pos_i - pos_j|
 * (pos may be NULL to use 0..T-1). Positions must be increasing. */
void trv_attention_head_pos(const float *q, const float *k, const float *v,
                            uint32_t ld, uint32_t T, uint32_t d,
                            int32_t half_window, const uint8_t *key_mask,
                            const uint32_t *pos, float *out, uint32_t ldo);

#ifdef __cplusplus
}
#endif

#endif /* TRIVIKRAMA_H */
