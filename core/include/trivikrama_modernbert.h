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
 *   - tile_buf:     weight tile buffer from the caller (any size >= one row + 4)
 *
 * The .trv file must come from `trivi export-modernbert` (python side), which
 * writes the tensors under fixed names plus two small config tensors.
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
    uint32_t wqkv, wo, wi, wo_mlp;          /* Q8_ROWS data offsets */
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

/* Called with layer = -1 after the embeddings, 0..n_layers-1 after each layer,
 * and n_layers after the final norm. Use it to compare against a reference. */
typedef void (*trv_mbert_hook)(void *user, int32_t layer, const float *h,
                               uint32_t T, uint32_t hidden);

/* Read the config and locate every tensor in one pass over the table. */
trv_status trv_mbert_load(trv_mbert *mb, const trv_model *m);

/* Scratch bytes needed by trv_mbert_forward for T tokens (excludes h and the
 * tile buffer). */
size_t trv_mbert_workspace_bytes(const trv_mbert *mb, uint32_t T);

/* Multiply-accumulates in one forward pass (linear layers + attention). */
uint64_t trv_mbert_macs(const trv_mbert *mb, uint32_t T);

/*
 * Run the encoder.
 *   ids        [T] token ids
 *   key_mask   [T] 1 = real token, 0 = padding; or NULL
 *   h          [T x hidden] floats; receives the final hidden states
 */
trv_status trv_mbert_forward(const trv_mbert *mb, const uint32_t *ids,
                             const uint8_t *key_mask, uint32_t T, float *h,
                             void *arena, size_t arena_bytes,
                             void *tile_buf, uint32_t tile_bytes,
                             trv_mbert_hook hook, void *user);

#ifdef __cplusplus
}
#endif

#endif /* TRIVIKRAMA_MODERNBERT_H */
