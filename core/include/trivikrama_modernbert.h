/*
 * ModernBERT encoder forward pass on top of the Trivikrama kernels.
 * Copyright 2026 Vadhyar Sai Charan
 * SPDX-License-Identifier: Apache-2.0
 *
 * Memory model
 *   - weights:      streamed from the trv_source; never resident
 *   - h:            the residual stream [T x hidden] floats, owned by the caller;
 *                   it also receives the final hidden states
 *   - arena:        scratch from the caller, size from trv_mbert_workspace_bytes()
 *   - tile_buf:     weight tile buffer from the caller,
 *                   at least trv_mbert_min_tile_bytes()
 *
 * Speed features (all optional, all generic):
 *   - batching:     several sequences in one pass; every weight is still read
 *                   once, so N sequences cost far less than N passes
 *   - early exit:   the hook can stop after any layer
 *   - token pruning:the hook can drop tokens between layers
 *   - int8 or int4 weights, chosen per matrix by the exporter
 *
 * The .trv file must come from `trivi export-modernbert`.
 */
#ifndef TRIVIKRAMA_MODERNBERT_H
#define TRIVIKRAMA_MODERNBERT_H

#include "trivikrama.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef TRV_MBERT_MAX_LAYERS
#define TRV_MBERT_MAX_LAYERS 48
#endif

typedef struct {
    uint32_t wqkv, wo, wi, wo_mlp;          /* weight data offsets */
    uint8_t wqkv_dt, wo_dt, wi_dt, wo_mlp_dt; /* TRV_DTYPE_Q8_ROWS or Q4_G32 */
    uint32_t attn_norm_w, attn_norm_b;      /* F32 offsets, 0 = absent */
    uint32_t mlp_norm_w, mlp_norm_b;
    int32_t half_window;                    /* -1 = global attention */
    float rope_theta;
} trv_mbert_layer;

typedef struct {
    const trv_model *m;
    uint32_t n_layers, hidden, n_heads, head_dim, inter, vocab, group;
    float norm_eps;
    uint32_t emb_tok;                       /* Q8_ROWS [vocab x hidden] */
    uint32_t emb_norm_w, emb_norm_b;
    uint32_t final_norm_w, final_norm_b;
    trv_mbert_layer layers[TRV_MBERT_MAX_LAYERS];
} trv_mbert;

/* What a hook may ask for after a layer. */
typedef struct {
    uint8_t stop;   /* set to 1: skip the remaining layers (final norm still runs) */
    uint8_t *keep;  /* [T], all 1 on entry; set entries to 0 to drop those tokens */
    const uint32_t *orig_index; /* [T] original packed index of each current token */
} trv_mbert_ctl;

/* Called with layer = -1 after the embeddings, 0..n_layers-1 after each layer,
 * and n_layers after the final norm (ctl is ignored there). */
typedef void (*trv_mbert_hook)(void *user, int32_t layer, const float *h, uint32_t T,
                               uint32_t hidden, trv_mbert_ctl *ctl);

/* Read the config and locate every tensor in one pass over the table. */
trv_status trv_mbert_load(trv_mbert *mb, const trv_model *m);

/* Scratch bytes for T packed tokens (any number of sequences), excluding h and
 * the tile buffer. */
size_t trv_mbert_workspace_bytes(const trv_mbert *mb, uint32_t T);

/* Smallest tile buffer that works with the current runtime (workers). For
 * double buffering with an async source, give at least twice the row part. */
uint32_t trv_mbert_min_tile_bytes(const trv_mbert *mb);

/* Multiply-accumulates of a forward pass over sequences of the given lengths. */
uint64_t trv_mbert_macs(const trv_mbert *mb, uint32_t T);
uint64_t trv_mbert_macs_batch(const trv_mbert *mb, const uint32_t *seq_lens,
                              uint32_t n_seqs);

/* Run one sequence. ids: [T]. key_mask: [T] (1 = real token) or NULL. */
trv_status trv_mbert_forward(const trv_mbert *mb, const uint32_t *ids,
                             const uint8_t *key_mask, uint32_t T, float *h,
                             void *arena, size_t arena_bytes,
                             void *tile_buf, uint32_t tile_bytes,
                             trv_mbert_hook hook, void *user);

/*
 * Run several sequences in one pass.
 *   ids        all tokens, sequences one after another (T = sum of seq_lens)
 *   key_mask   [T] or NULL
 *   h          [T x hidden]; on return its first *T_out rows hold the final
 *              hidden states of the tokens that were kept
 *   T_out      number of tokens left (T unless the hook dropped some); may be NULL
 *   orig_index [T] or NULL; receives the packed input index of each output row
 */
trv_status trv_mbert_forward_batch(const trv_mbert *mb, const uint32_t *ids,
                                   const uint32_t *seq_lens, uint32_t n_seqs,
                                   const uint8_t *key_mask, float *h,
                                   uint32_t *T_out, uint32_t *orig_index,
                                   void *arena, size_t arena_bytes,
                                   void *tile_buf, uint32_t tile_bytes,
                                   trv_mbert_hook hook, void *user);

#ifdef __cplusplus
}
#endif

#endif /* TRIVIKRAMA_MODERNBERT_H */
