/*
 * ModernBERT encoder forward pass.
 * Copyright 2026 Vadhyar Sai Charan
 * SPDX-License-Identifier: Apache-2.0
 *
 * One layer, with the memory each step touches:
 *
 *   h --LN+quant--> xq_in (int8)                         [T x H]
 *   for each head:  q,k,v = xq_in . Wqkv[rows of head]     [T x hd] each
 *                   rope(q,k); o = attention(q,k,v)       per sequence
 *                   quantize o -> xq_b[:, head cols]      int8
 *   h += xq_b . Wo                                        (accumulate in place)
 *   h --LN+quant--> xq_in
 *   for each FFN chunk of `group` columns:
 *                   [a | gate] = xq_in . Wi[rows of chunk]  [T x 2*group]
 *                   geglu -> quantize -> xq_b[:, chunk cols] int8
 *   h += xq_b . Wo_mlp
 *
 * T counts the tokens of every sequence in the batch, so each weight matrix
 * is read exactly once per forward pass whatever the batch size.
 */
#include "trivikrama_modernbert.h"

#include <stdlib.h>
#include <string.h>

/* config tensor layout written by the exporter */
enum {
    CFG_N_LAYERS, CFG_HIDDEN, CFG_N_HEADS, CFG_INTER, CFG_VOCAB,
    CFG_NORM_EPS, CFG_GROUP, CFG_COUNT
};

static trv_tensor mat_view(uint8_t dtype, uint32_t offset, uint32_t N, uint32_t K)
{
    trv_tensor t;
    memset(&t, 0, sizeof t);
    t.dtype = dtype;
    t.ndim = 2;
    t.dims[0] = N;
    t.dims[1] = K;
    t.dims[2] = t.dims[3] = 1;
    t.offset = offset;
    t.nbytes = dtype == TRV_DTYPE_Q8_ROWS ? 4u * N + N * K : N * TRV_Q4_ROW_BYTES(K);
    return t;
}

static int expect_vec(const trv_tensor *t, uint32_t n)
{
    return t->dtype == TRV_DTYPE_F32 && t->ndim == 1 && t->dims[0] == n;
}

/* a weight matrix may be int8 or int4; returns its dtype, or 0 if wrong */
static uint8_t expect_mat(const trv_tensor *t, uint32_t N, uint32_t K)
{
    if (t->ndim != 2 || t->dims[0] != N || t->dims[1] != K) return 0;
    if (t->dtype == TRV_DTYPE_Q8_ROWS && t->nbytes == 4u * N + N * K) return t->dtype;
    if (t->dtype == TRV_DTYPE_Q4_G32 && K % TRV_Q4_GROUP == 0 &&
        t->nbytes == N * TRV_Q4_ROW_BYTES(K))
        return t->dtype;
    return 0;
}

#define SET_MAT(off_field, dt_field, N, K)                                      \
    do {                                                                       \
        uint8_t dt_ = expect_mat(&t, (N), (K));                                \
        if (!dt_) return TRV_ERR_FORMAT;                                       \
        ly->off_field = t.offset;                                              \
        ly->dt_field = dt_;                                                    \
    } while (0)

#define SET_VEC(dst)                                                           \
    do {                                                                       \
        if (!expect_vec(&t, H)) return TRV_ERR_FORMAT;                         \
        dst = t.offset;                                                        \
    } while (0)

trv_status trv_mbert_load(trv_mbert *mb, const trv_model *m)
{
    char name[TRV_MAX_NAME + 1];
    trv_tensor t, cfg_t, lay_t;
    float cfg[CFG_COUNT];
    uint32_t pos, i, H, I, L;
    trv_status st;

    if (!mb || !m) return TRV_ERR_ARG;
    memset(mb, 0, sizeof *mb);
    mb->m = m;

    if ((st = trv_model_find(m, "trv.modernbert.config", &cfg_t)) != TRV_OK) return st;
    if (cfg_t.dtype != TRV_DTYPE_F32 || trv_tensor_numel(&cfg_t) < CFG_COUNT)
        return TRV_ERR_FORMAT;
    if ((st = trv_tensor_read_f32(m, &cfg_t, 0, cfg, CFG_COUNT)) != TRV_OK) return st;

    mb->n_layers = (uint32_t)cfg[CFG_N_LAYERS];
    mb->hidden = (uint32_t)cfg[CFG_HIDDEN];
    mb->n_heads = (uint32_t)cfg[CFG_N_HEADS];
    mb->inter = (uint32_t)cfg[CFG_INTER];
    mb->vocab = (uint32_t)cfg[CFG_VOCAB];
    mb->norm_eps = cfg[CFG_NORM_EPS];
    mb->group = (uint32_t)cfg[CFG_GROUP];
    L = mb->n_layers;
    H = mb->hidden;
    I = mb->inter;
    if (L == 0 || L > TRV_MBERT_MAX_LAYERS || mb->n_heads == 0 || H % mb->n_heads)
        return TRV_ERR_FORMAT;
    mb->head_dim = H / mb->n_heads;
    if (mb->group == 0 || mb->head_dim % mb->group || I % mb->group) return TRV_ERR_FORMAT;

    /* per-layer attention window and rope base: [L x 2] */
    if ((st = trv_model_find(m, "trv.modernbert.layers", &lay_t)) != TRV_OK) return st;
    if (lay_t.dtype != TRV_DTYPE_F32 || lay_t.ndim != 2 || lay_t.dims[0] != L ||
        lay_t.dims[1] != 2)
        return TRV_ERR_FORMAT;
    for (i = 0; i < L; ++i) {
        float v[2];
        if ((st = trv_tensor_read_f32(m, &lay_t, 2 * i, v, 2)) != TRV_OK) return st;
        mb->layers[i].half_window = (int32_t)v[0];
        mb->layers[i].rope_theta = v[1];
    }

    /* one pass over the table to find every weight */
    pos = m->table_offset;
    for (i = 0; i < m->n_tensors; ++i) {
        if ((st = trv_model_next(m, &pos, name, &t)) != TRV_OK) return st;
        if (strcmp(name, "emb.tok") == 0) {
            if (t.dtype != TRV_DTYPE_Q8_ROWS || t.ndim != 2 || t.dims[0] != mb->vocab ||
                t.dims[1] != H)
                return TRV_ERR_FORMAT;
            mb->emb_tok = t.offset;
        } else if (strcmp(name, "emb.norm.w") == 0) {
            SET_VEC(mb->emb_norm_w);
        } else if (strcmp(name, "emb.norm.b") == 0) {
            SET_VEC(mb->emb_norm_b);
        } else if (strcmp(name, "final_norm.w") == 0) {
            SET_VEC(mb->final_norm_w);
        } else if (strcmp(name, "final_norm.b") == 0) {
            SET_VEC(mb->final_norm_b);
        } else if (name[0] == 'L') {
            char *dot;
            unsigned long li = strtoul(name + 1, &dot, 10);
            trv_mbert_layer *ly;
            if (*dot != '.' || li >= L) continue;
            ly = &mb->layers[li];
            ++dot;
            if (strcmp(dot, "wqkv") == 0)
                SET_MAT(wqkv, wqkv_dt, 3 * H, H);
            else if (strcmp(dot, "wo") == 0)
                SET_MAT(wo, wo_dt, H, H);
            else if (strcmp(dot, "wi") == 0)
                SET_MAT(wi, wi_dt, 2 * I, H);
            else if (strcmp(dot, "wo_mlp") == 0)
                SET_MAT(wo_mlp, wo_mlp_dt, H, I);
            else if (strcmp(dot, "attn_norm.w") == 0)
                SET_VEC(ly->attn_norm_w);
            else if (strcmp(dot, "attn_norm.b") == 0)
                SET_VEC(ly->attn_norm_b);
            else if (strcmp(dot, "mlp_norm.w") == 0)
                SET_VEC(ly->mlp_norm_w);
            else if (strcmp(dot, "mlp_norm.b") == 0)
                SET_VEC(ly->mlp_norm_b);
        }
    }

    /* data offsets are never 0 (the header lives there), so 0 means missing */
    if (!mb->emb_tok || !mb->emb_norm_w || !mb->final_norm_w) return TRV_ERR_NOT_FOUND;
    for (i = 0; i < L; ++i) {
        const trv_mbert_layer *ly = &mb->layers[i];
        if (!ly->wqkv || !ly->wo || !ly->wi || !ly->wo_mlp || !ly->mlp_norm_w)
            return TRV_ERR_NOT_FOUND;
        if ((ly->wo_mlp_dt == TRV_DTYPE_Q4_G32 || ly->wqkv_dt == TRV_DTYPE_Q4_G32 ||
             ly->wi_dt == TRV_DTYPE_Q4_G32 || ly->wo_dt == TRV_DTYPE_Q4_G32) &&
            mb->group % TRV_Q4_GROUP != 0 && TRV_Q4_GROUP % mb->group != 0)
            return TRV_ERR_FORMAT;
    }
    return TRV_OK;
}

/* ---- arena layout ------------------------------------------------------- */

typedef struct {
    int8_t *xq_in;  /* [T x H]             normalized layer input, int8 */
    int8_t *xq_b;   /* [T x max(H, I)]     attention output / FFN hidden */
    float *xs_in;   /* [T x H/g] */
    float *xs_b;    /* [T x max(H, I)/g] */
    float *q, *k, *v, *o; /* [T x hd] each */
    float *ffn;     /* [T x 2g]            one GeGLU chunk */
    float *gamma, *beta; /* [H] */
    uint32_t *pos;  /* [T] position of each token inside its sequence */
    uint32_t *idx;  /* [T] original packed index */
    uint32_t *lens; /* [T] (at most T sequences) */
    uint8_t *mask;  /* [T] */
    uint8_t *keep;  /* [T] */
} arena_t;

static size_t up4(size_t n) { return (n + 3u) & ~(size_t)3u; }

static size_t layout(const trv_mbert *mb, uint32_t T, uint8_t *base, arena_t *a)
{
    size_t H = mb->hidden, W = mb->inter > H ? mb->inter : H, hd = mb->head_dim;
    size_t g = mb->group, off = 0;
#define TAKE(field, type, count)                                               \
    do {                                                                       \
        if (a) a->field = (type *)(void *)(base + off);                        \
        off += up4(sizeof(type) * (size_t)(count));                            \
    } while (0)
    TAKE(xq_in, int8_t, T * H);
    TAKE(xq_b, int8_t, T * W);
    TAKE(xs_in, float, T * (H / g));
    TAKE(xs_b, float, T * (W / g));
    TAKE(q, float, T * hd);
    TAKE(k, float, T * hd);
    TAKE(v, float, T * hd);
    TAKE(o, float, T * hd);
    TAKE(ffn, float, T * 2 * g);
    TAKE(gamma, float, H);
    TAKE(beta, float, H);
    TAKE(pos, uint32_t, T);
    TAKE(idx, uint32_t, T);
    TAKE(lens, uint32_t, T);
    TAKE(mask, uint8_t, T);
    TAKE(keep, uint8_t, T);
#undef TAKE
    return off;
}

size_t trv_mbert_workspace_bytes(const trv_mbert *mb, uint32_t T)
{
    return layout(mb, T, NULL, NULL);
}

static uint32_t max_scratch(const trv_mbert *mb)
{
    uint32_t H = mb->hidden, I = mb->inter, best = 0, i, j;
    for (i = 0; i < mb->n_layers; ++i) {
        const trv_mbert_layer *ly = &mb->layers[i];
        const uint8_t dts[4] = {ly->wqkv_dt, ly->wo_dt, ly->wi_dt, ly->wo_mlp_dt};
        const uint32_t Ks[4] = {H, H, H, I};
        for (j = 0; j < 4; ++j) {
            uint32_t s = trv_linear_scratch_bytes(dts[j], Ks[j], mb->group);
            if (s > best) best = s;
        }
    }
    return best;
}

uint32_t trv_mbert_min_tile_bytes(const trv_mbert *mb)
{
    uint32_t H = mb->hidden, I = mb->inter, best = 0, i, need;
    for (i = 0; i < mb->n_layers; ++i) {
        const trv_mbert_layer *ly = &mb->layers[i];
        const uint8_t dts[4] = {ly->wqkv_dt, ly->wo_dt, ly->wi_dt, ly->wo_mlp_dt};
        const uint32_t Ks[4] = {H, H, H, I};
        uint32_t j;
        for (j = 0; j < 4; ++j) {
            need = max_scratch(mb) + trv_weight_row_bytes(dts[j], Ks[j]);
            if (need > best) best = need;
        }
    }
    return best;
}

static uint64_t attention_keys(int32_t hw, uint64_t T)
{
    uint64_t keys = 0, t;
    for (t = 0; t < T; ++t) {
        uint64_t lo = (hw >= 0 && t > (uint64_t)hw) ? t - (uint64_t)hw : 0;
        uint64_t hi = (hw >= 0 && t + (uint64_t)hw + 1 < T) ? t + (uint64_t)hw + 1 : T;
        keys += hi - lo;
    }
    return keys;
}

uint64_t trv_mbert_macs_batch(const trv_mbert *mb, const uint32_t *seq_lens,
                              uint32_t n_seqs)
{
    uint64_t H = mb->hidden, I = mb->inter, hd = mb->head_dim, nh = mb->n_heads;
    uint64_t total = 0, T = 0;
    uint32_t i, s;
    for (s = 0; s < n_seqs; ++s) T += seq_lens[s];
    for (i = 0; i < mb->n_layers; ++i) {
        total += T * (3 * H * H + H * H + 2 * I * H + H * I);
        for (s = 0; s < n_seqs; ++s)
            total += 2 * attention_keys(mb->layers[i].half_window, seq_lens[s]) * hd * nh;
    }
    return total;
}

uint64_t trv_mbert_macs(const trv_mbert *mb, uint32_t T)
{
    return trv_mbert_macs_batch(mb, &T, 1);
}

/* ---- forward -------------------------------------------------------------- */

static trv_status read_vec(const trv_mbert *mb, uint32_t off, float *dst)
{
    trv_linear_drain(mb->m->src);
    return mb->m->src->read(mb->m->src->ctx, off, dst, 4u * mb->hidden) == 0
               ? TRV_OK
               : TRV_ERR_IO;
}

/* in-place float LayerNorm with weights fetched from the source */
static trv_status norm_inplace(const trv_mbert *mb, arena_t *a, float *h, uint32_t T,
                               uint32_t w_off, uint32_t b_off)
{
    trv_status st;
    if ((st = read_vec(mb, w_off, a->gamma)) != TRV_OK) return st;
    if (b_off && (st = read_vec(mb, b_off, a->beta)) != TRV_OK) return st;
    trv_layernorm(h, T, mb->hidden, mb->hidden, a->gamma, b_off ? a->beta : NULL,
                  mb->norm_eps);
    return TRV_OK;
}

/* LayerNorm + quantize into xq_in. w_off == 0 means identity (ModernBERT layer 0) */
static trv_status norm_quant(const trv_mbert *mb, arena_t *a, const float *h, uint32_t T,
                             uint32_t w_off, uint32_t b_off)
{
    uint32_t H = mb->hidden, g = mb->group;
    trv_status st;
    if (!w_off) {
        trv_quantize_act_q8g(h, T, H, H, g, a->xq_in, H, a->xs_in, H / g);
        return TRV_OK;
    }
    if ((st = read_vec(mb, w_off, a->gamma)) != TRV_OK) return st;
    if (b_off && (st = read_vec(mb, b_off, a->beta)) != TRV_OK) return st;
    trv_layernorm_quantize(h, T, H, H, a->gamma, b_off ? a->beta : NULL, mb->norm_eps,
                           g, a->xq_in, a->xs_in);
    return TRV_OK;
}

/* Remove the tokens with keep == 0 everywhere they live. Returns the new T. */
static uint32_t compact(const trv_mbert *mb, arena_t *a, float *h, uint32_t T,
                        uint32_t n_seqs)
{
    uint32_t H = mb->hidden, t, w = 0, s, start = 0;
    for (s = 0; s < n_seqs; ++s) {
        uint32_t end = start + a->lens[s], kept = 0;
        for (t = start; t < end; ++t) {
            if (!a->keep[t]) continue;
            if (w != t) {
                memmove(h + (size_t)w * H, h + (size_t)t * H, sizeof(float) * H);
                a->pos[w] = a->pos[t];
                a->idx[w] = a->idx[t];
                a->mask[w] = a->mask[t];
            }
            ++w;
            ++kept;
        }
        a->lens[s] = kept;
        start = end;
    }
    (void)T;
    return w;
}

/* Call the hook and apply what it asked for. */
static uint32_t run_hook(const trv_mbert *mb, arena_t *a, trv_mbert_hook hook, void *user,
                         int32_t layer, float *h, uint32_t T, uint32_t n_seqs,
                         int *stop)
{
    trv_mbert_ctl ctl;
    uint32_t t, dropped = 0;
    if (!hook) return T;
    memset(a->keep, 1, T);
    ctl.stop = 0;
    ctl.keep = a->keep;
    ctl.orig_index = a->idx;
    hook(user, layer, h, T, mb->hidden, &ctl);
    if (layer >= (int32_t)mb->n_layers) return T;
    if (ctl.stop) *stop = 1;
    for (t = 0; t < T; ++t) dropped += a->keep[t] == 0;
    return dropped ? compact(mb, a, h, T, n_seqs) : T;
}

static trv_status forward_batch(const trv_mbert *mb, const uint32_t *ids,
                                const uint32_t *seq_lens, uint32_t n_seqs,
                                const uint8_t *key_mask, float *h,
                                uint32_t *T_out, uint32_t *orig_index,
                                void *arena, size_t arena_bytes,
                                void *tile_buf, uint32_t tile_bytes,
                                trv_mbert_hook hook, void *user)
{
    const trv_model *m;
    uint32_t H, I, hd, g, L, i, t, s, hh, c0, T = 0, T0;
    arena_t a;
    trv_tensor emb;
    trv_status st;
    int stop = 0;

    if (!mb || !ids || !seq_lens || !h || !arena || !tile_buf || n_seqs == 0)
        return TRV_ERR_ARG;
    for (s = 0; s < n_seqs; ++s) T += seq_lens[s];
    if (T == 0 || n_seqs > T) return TRV_ERR_ARG;
    if (((uintptr_t)arena & 3u) != 0) return TRV_ERR_ARG;
    if (layout(mb, T, (uint8_t *)arena, &a) > arena_bytes) return TRV_ERR_BUFFER;
    m = mb->m;
    H = mb->hidden;
    I = mb->inter;
    hd = mb->head_dim;
    g = mb->group;
    L = mb->n_layers;
    T0 = T;

    /* per-token bookkeeping: position restarts at 0 in every sequence */
    for (s = 0, t = 0; s < n_seqs; ++s) {
        uint32_t j;
        a.lens[s] = seq_lens[s];
        for (j = 0; j < seq_lens[s]; ++j, ++t) {
            a.pos[t] = j;
            a.idx[t] = t;
            a.mask[t] = key_mask ? (key_mask[t] != 0) : 1u;
        }
    }

    /* embeddings: one row per token, then LayerNorm */
    emb = mat_view(TRV_DTYPE_Q8_ROWS, mb->emb_tok, mb->vocab, H);
    for (t = 0; t < T; ++t) {
        if (ids[t] >= mb->vocab) return TRV_ERR_ARG;
        st = trv_tensor_read_q8_row(m, &emb, ids[t], h + (size_t)t * H, a.xq_b);
        if (st != TRV_OK) return st;
    }
    if ((st = norm_inplace(mb, &a, h, T, mb->emb_norm_w, mb->emb_norm_b)) != TRV_OK) return st;
    T = run_hook(mb, &a, hook, user, -1, h, T, n_seqs, &stop);

    for (i = 0; i < L && !stop && T > 0; ++i) {
        const trv_mbert_layer *ly = &mb->layers[i];
        trv_tensor wqkv = mat_view(ly->wqkv_dt, ly->wqkv, 3 * H, H);
        trv_tensor wo = mat_view(ly->wo_dt, ly->wo, H, H);
        trv_tensor wi = mat_view(ly->wi_dt, ly->wi, 2 * I, H);
        trv_tensor wo_mlp = mat_view(ly->wo_mlp_dt, ly->wo_mlp, H, I);

        /* ---- attention ---- */
        if ((st = norm_quant(mb, &a, h, T, ly->attn_norm_w, ly->attn_norm_b)) != TRV_OK)
            return st;
        for (hh = 0; hh < mb->n_heads; ++hh) {
            uint32_t r = hh * hd, start = 0;
            /* each call announces the next one so its weights load during compute */
            trv_linear_set_next(&wqkv, g, H + r, H + r + hd);
            st = trv_linear_stream(m, &wqkv, a.xq_in, a.xs_in, g, T, NULL, r, r + hd, a.q,
                                   hd, 0, tile_buf, tile_bytes);
            if (st != TRV_OK) return st;
            trv_linear_set_next(&wqkv, g, 2 * H + r, 2 * H + r + hd);
            st = trv_linear_stream(m, &wqkv, a.xq_in, a.xs_in, g, T, NULL, H + r,
                                   H + r + hd, a.k, hd, 0, tile_buf, tile_bytes);
            if (st != TRV_OK) return st;
            if (hh + 1 < mb->n_heads)
                trv_linear_set_next(&wqkv, g, r + hd, r + 2 * hd);
            else
                trv_linear_set_next(&wo, g, 0, H);
            st = trv_linear_stream(m, &wqkv, a.xq_in, a.xs_in, g, T, NULL, 2 * H + r,
                                   2 * H + r + hd, a.v, hd, 0, tile_buf, tile_bytes);
            if (st != TRV_OK) return st;
            trv_rope_pos(a.q, T, 1, hd, hd, a.pos, ly->rope_theta);
            trv_rope_pos(a.k, T, 1, hd, hd, a.pos, ly->rope_theta);
            for (s = 0; s < n_seqs; ++s) { /* tokens only attend within their sequence */
                uint32_t n = a.lens[s];
                if (n)
                    trv_attention_head_pos(a.q + (size_t)start * hd, a.k + (size_t)start * hd,
                                           a.v + (size_t)start * hd, hd, n, hd,
                                           ly->half_window, a.mask + start, a.pos + start,
                                           a.o + (size_t)start * hd, hd);
                start += n;
            }
            trv_quantize_act_q8g(a.o, T, hd, hd, g, a.xq_b + r, H, a.xs_b + r / g, H / g);
        }
        st = trv_linear_stream(m, &wo, a.xq_b, a.xs_b, g, T, NULL, 0, H, h, H, 1, tile_buf,
                               tile_bytes);
        if (st != TRV_OK) return st;

        /* ---- GeGLU MLP, one chunk of `g` hidden units at a time ---- */
        if ((st = norm_quant(mb, &a, h, T, ly->mlp_norm_w, ly->mlp_norm_b)) != TRV_OK)
            return st;
        for (c0 = 0; c0 < I; c0 += g) {
            trv_linear_set_next(&wi, g, I + c0, I + c0 + g);
            st = trv_linear_stream(m, &wi, a.xq_in, a.xs_in, g, T, NULL, c0, c0 + g, a.ffn,
                                   2 * g, 0, tile_buf, tile_bytes);
            if (st != TRV_OK) return st;
            if (c0 + g < I)
                trv_linear_set_next(&wi, g, c0 + g, c0 + 2 * g);
            else
                trv_linear_set_next(&wo_mlp, g, 0, H);
            st = trv_linear_stream(m, &wi, a.xq_in, a.xs_in, g, T, NULL, I + c0,
                                   I + c0 + g, a.ffn + g, 2 * g, 0, tile_buf, tile_bytes);
            if (st != TRV_OK) return st;
            trv_geglu(a.ffn, T, g, 2 * g, a.ffn, 2 * g);
            trv_quantize_act_q8g(a.ffn, T, g, 2 * g, g, a.xq_b + c0, I, a.xs_b + c0 / g,
                                 I / g);
        }
        st = trv_linear_stream(m, &wo_mlp, a.xq_b, a.xs_b, g, T, NULL, 0, H, h, H, 1,
                               tile_buf, tile_bytes);
        if (st != TRV_OK) return st;

        T = run_hook(mb, &a, hook, user, (int32_t)i, h, T, n_seqs, &stop);
    }

    if ((st = norm_inplace(mb, &a, h, T, mb->final_norm_w, mb->final_norm_b)) != TRV_OK)
        return st;
    run_hook(mb, &a, hook, user, (int32_t)L, h, T, n_seqs, &stop);
    if (T_out) *T_out = T;
    if (orig_index) memcpy(orig_index, a.idx, sizeof(uint32_t) * T);
    (void)T0;
    return TRV_OK;
}

trv_status trv_mbert_forward_batch(const trv_mbert *mb, const uint32_t *ids,
                                   const uint32_t *seq_lens, uint32_t n_seqs,
                                   const uint8_t *key_mask, float *h,
                                   uint32_t *T_out, uint32_t *orig_index,
                                   void *arena, size_t arena_bytes,
                                   void *tile_buf, uint32_t tile_bytes,
                                   trv_mbert_hook hook, void *user)
{
    trv_status st;
    if (!mb || !mb->m) return TRV_ERR_ARG;
    trv_linear_reserve_scratch(max_scratch(mb));
    st = forward_batch(mb, ids, seq_lens, n_seqs, key_mask, h, T_out, orig_index, arena,
                       arena_bytes, tile_buf, tile_bytes, hook, user);
    trv_linear_set_next(NULL, 0, 0, 0);
    trv_linear_drain(mb->m->src);
    trv_linear_reserve_scratch(0);
    return st;
}

trv_status trv_mbert_forward(const trv_mbert *mb, const uint32_t *ids,
                             const uint8_t *key_mask, uint32_t T, float *h,
                             void *arena, size_t arena_bytes,
                             void *tile_buf, uint32_t tile_bytes,
                             trv_mbert_hook hook, void *user)
{
    return trv_mbert_forward_batch(mb, ids, &T, 1, key_mask, h, NULL, NULL, arena,
                                   arena_bytes, tile_buf, tile_bytes, hook, user);
}
