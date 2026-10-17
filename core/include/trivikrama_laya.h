/*
 * Laya decision model (Convai Innovations) on the device: ModernBERT encoder
 * + decision head, all streamed from the same .trv file.
 * Copyright 2026 Vadhyar Sai Charan
 * SPDX-License-Identifier: Apache-2.0
 *
 * The .trv comes from `trivi export-laya`. Questions are turned into token
 * rows on a PC or phone (`trivi laya-prepare`): one row per question, each
 * with its question type, its token ids and the position of every option's
 * marker. The device runs every row of a request in ONE batched pass, so
 * asking 5 questions costs far less than 5 single passes.
 *
 * What comes out is one probability per option, already temperature-scaled
 * the way Laya does it:
 *   yes/no (noul)  probs[1] = probability of "yes"
 *   choice         argmax = the chosen option
 *   score          sum(i * probs[i]) = the expected level
 */
#ifndef TRIVIKRAMA_LAYA_H
#define TRIVIKRAMA_LAYA_H

#include "trivikrama_modernbert.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef TRV_LAYA_MAX_HEAD_LAYERS
#define TRV_LAYA_MAX_HEAD_LAYERS 4
#endif

enum { TRV_LAYA_CHOICE = 0, TRV_LAYA_SCORE = 1, TRV_LAYA_NOUL = 2 };

typedef struct {
    trv_tensor qkv, out, ff1, ff2;               /* Q8_ROWS */
    uint32_t qkv_b, out_b, ff1_b, ff2_b;         /* F32 offsets */
    uint32_t n1_w, n1_b, n2_w, n2_b;
} trv_laya_layer;

typedef struct {
    trv_mbert enc;
    uint32_t n_layers, n_heads, head_dim, ff, hidden, group;
    float eps;
    float temps[15];        /* 3 base + 3 types x 4 option-count buckets */
    uint32_t type_emb;      /* F32 [3 x hidden] */
    trv_laya_layer layers[TRV_LAYA_MAX_HEAD_LAYERS];
    uint32_t ln_w, ln_b, l1_b, l2, l2_b;
    trv_tensor l1;          /* Q8_ROWS [hidden x hidden] */
} trv_laya;

/* One question row. */
typedef struct {
    uint8_t qtype;            /* TRV_LAYA_CHOICE / SCORE / NOUL */
    uint8_t n_options;
    uint16_t n_tokens;
    const uint32_t *markers;  /* [n_options] token positions inside the row */
    const uint32_t *ids;      /* [n_tokens] */
} trv_laya_row;

trv_status trv_laya_load(trv_laya *ly, const trv_model *m);

/* Totals of a request: tokens over all rows, and options over all rows. */
void trv_laya_totals(const trv_laya_row *rows, uint32_t n_rows, uint32_t *T,
                     uint32_t *n_options);

/* Arena bytes for a request (covers the encoder and the head), excluding h
 * [T x hidden floats] and the tile buffer. */
size_t trv_laya_workspace_bytes(const trv_laya *ly, uint32_t T, uint32_t n_options,
                                uint32_t n_rows);

/* Smallest tile buffer for the current runtime. */
uint32_t trv_laya_min_tile_bytes(const trv_laya *ly);

/* Temperature Laya applies to a question type with k options. */
float trv_laya_temperature(const trv_laya *ly, uint32_t qtype, uint32_t k);

/*
 * Answer every row.
 *   h       [T x hidden] floats, T = total tokens (see trv_laya_totals)
 *   logits  [total options] raw scores, or NULL
 *   probs   [total options] probabilities, row after row (each row sums to 1)
 *   hook    optional encoder hook (progress, early exit); may be NULL
 */
trv_status trv_laya_run(const trv_laya *ly, const trv_laya_row *rows, uint32_t n_rows,
                        float *h, float *logits, float *probs,
                        void *arena, size_t arena_bytes,
                        void *tile_buf, uint32_t tile_bytes,
                        trv_mbert_hook hook, void *user);

/*
 * Request file written by `trivi laya-prepare` (little-endian, 4-byte aligned):
 *   "TRLQ" | u32 version (1) | u32 n_rows
 *   per row: u8 qtype | u8 n_options | u16 n_tokens |
 *            u32 markers[n_options] | u32 ids[n_tokens]
 * Parses a request held in memory (4-byte aligned) without copying: rows[]
 * point into buf. Returns the number of rows, or a negative trv_status.
 */
int32_t trv_laya_parse_request(const void *buf, size_t len, trv_laya_row *rows,
                               uint32_t max_rows);

#ifdef __cplusplus
}
#endif

#endif /* TRIVIKRAMA_LAYA_H */
