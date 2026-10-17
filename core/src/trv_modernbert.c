/*
 * ModernBERT encoder forward pass.
 * Copyright 2026 Vadhyar Sai Charan
 * SPDX-License-Identifier: Apache-2.0
 *
 * One layer, with the memory each step touches:
 *
 *   h --LN+quant--> xq_in (int8)                         [T x H]
 *   for each head:  q,k,v = xq_in . Wqkv[rows of head]     [T x hd] each
 *                   rope(q,k); o = attention(q,k,v)       [T x hd]
 *                   quantize o -> xq_b[:, head cols]      int8
 *   h += xq_b . Wo                                        (accumulate in place)
 *   h --LN+quant--> xq_in
 *   for each FFN chunk of `group` columns:
 *                   [a | gate] = xq_in . Wi[rows of chunk]  [T x 2*group]
 *                   geglu -> quantize -> xq_b[:, chunk cols] int8
 *   h += xq_b . Wo_mlp
 *
 * Every weight matrix is read exactly once per forward pass.
 */
#include "trivikrama_modernbert.h"

#include <stdlib.h>
#include <string.h>

/* config tensor layout written by the exporter */
enum {
    CFG_N_LAYERS, CFG_HIDDEN, CFG_N_HEADS, CFG_INTER, CFG_VOCAB,
    CFG_NORM_EPS, CFG_GROUP, CFG_COUNT
};

static trv_tensor q8_view(uint32_t offset, uint32_t N, uint32_t K)
{
    trv_tensor t;
    memset(&t, 0, sizeof t);
    t.dtype = TRV_DTYPE_Q8_ROWS;
    t.ndim = 2;
    t.dims[0] = N;
    t.dims[1] = K;
    t.dims[2] = t.dims[3] = 1;
    t.offset = offset;
    t.nbytes = 4u * N + N * K;
    return t;
}

static int expect(const trv_tensor *t, uint8_t dtype, uint32_t d0, uint32_t d1)
{
    if (t->dtype != dtype) return 0;
    if (d1 == 0) return t->ndim == 1 && t->dims[0] == d0;
    return t->ndim == 2 && t->dims[0] == d0 && t->dims[1] == d1;
}

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
    if (!expect(&lay_t, TRV_DTYPE_F32, L, 2)) return TRV_ERR_FORMAT;
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
            if (!expect(&t, TRV_DTYPE_Q8_ROWS, mb->vocab, H)) return TRV_ERR_FORMAT;
            mb->emb_tok = t.offset;
        } else if (strcmp(name, "emb.norm.w") == 0) {
            if (!expect(&t, TRV_DTYPE_F32, H, 0)) return TRV_ERR_FORMAT;
            mb->emb_norm_w = t.offset;
        } else if (strcmp(name, "emb.norm.b") == 0) {
            if (!expect(&t, TRV_DTYPE_F32, H, 0)) return TRV_ERR_FORMAT;
            mb->emb_norm_b = t.offset;
        } else if (strcmp(name, "final_norm.w") == 0) {
            if (!expect(&t, TRV_DTYPE_F32, H, 0)) return TRV_ERR_FORMAT;
            mb->final_norm_w = t.offset;
        } else if (strcmp(name, "final_norm.b") == 0) {
            if (!expect(&t, TRV_DTYPE_F32, H, 0)) return TRV_ERR_FORMAT;
            mb->final_norm_b = t.offset;
        } else if (name[0] == 'L') {
            char *dot;
            unsigned long li = strtoul(name + 1, &dot, 10);
            trv_mbert_layer *ly;
            if (*dot != '.' || li >= L) continue;
            ly = &mb->layers[li];
            ++dot;
            if (strcmp(dot, "wqkv") == 0) {
                if (!expect(&t, TRV_DTYPE_Q8_ROWS, 3 * H, H)) return TRV_ERR_FORMAT;
                ly->wqkv = t.offset;
            } else if (strcmp(dot, "wo") == 0) {
                if (!expect(&t, TRV_DTYPE_Q8_ROWS, H, H)) return TRV_ERR_FORMAT;
                ly->wo = t.offset;
            } else if (strcmp(dot, "wi") == 0) {
                if (!expect(&t, TRV_DTYPE_Q8_ROWS, 2 * I, H)) return TRV_ERR_FORMAT;
                ly->wi = t.offset;
            } else if (strcmp(dot, "wo_mlp") == 0) {
                if (!expect(&t, TRV_DTYPE_Q8_ROWS, H, I)) return TRV_ERR_FORMAT;
                ly->wo_mlp = t.offset;
            } else if (strcmp(dot, "attn_norm.w") == 0) {
                if (!expect(&t, TRV_DTYPE_F32, H, 0)) return TRV_ERR_FORMAT;
                ly->attn_norm_w = t.offset;
            } else if (strcmp(dot, "attn_norm.b") == 0) {
                if (!expect(&t, TRV_DTYPE_F32, H, 0)) return TRV_ERR_FORMAT;
                ly->attn_norm_b = t.offset;
            } else if (strcmp(dot, "mlp_norm.w") == 0) {
                if (!expect(&t, TRV_DTYPE_F32, H, 0)) return TRV_ERR_FORMAT;
                ly->mlp_norm_w = t.offset;
            } else if (strcmp(dot, "mlp_norm.b") == 0) {
                if (!expect(&t, TRV_DTYPE_F32, H, 0)) return TRV_ERR_FORMAT;
                ly->mlp_norm_b = t.offset;
            }
        }
    }

    /* data offsets are never 0 (the header lives there), so 0 means missing */
    if (!mb->emb_tok || !mb->emb_norm_w || !mb->final_norm_w) return TRV_ERR_NOT_FOUND;
    for (i = 0; i < L; ++i) {
        const trv_mbert_layer *ly = &mb->layers[i];
        if (!ly->wqkv || !ly->wo || !ly->wi || !ly->wo_mlp || !ly->mlp_norm_w)
            return TRV_ERR_NOT_FOUND;
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
#undef TAKE
    return off;
}

size_t trv_mbert_workspace_bytes(const trv_mbert *mb, uint32_t T)
{
    return layout(mb, T, NULL, NULL);
}

uint64_t trv_mbert_macs(const trv_mbert *mb, uint32_t T)
{
    uint64_t H = mb->hidden, I = mb->inter, hd = mb->head_dim, nh = mb->n_heads;
    uint64_t total = 0, i;
    for (i = 0; i < mb->n_layers; ++i) {
        int32_t hw = mb->layers[i].half_window;
        uint64_t keys = 0, t;
        /* linear layers: Wqkv (3H x H), Wo (H x H), Wi (2I x H), Wo_mlp (H x I) */
        total += (uint64_t)T * (3 * H * H + H * H + 2 * I * H + H * I);
        /* attention: QK^T and PV over the visible keys of every query */
        for (t = 0; t < T; ++t) {
            uint64_t lo = (hw >= 0 && t > (uint64_t)hw) ? t - (uint64_t)hw : 0;
            uint64_t hi = (hw >= 0 && t + (uint64_t)hw + 1 < T) ? t + (uint64_t)hw + 1 : T;
            keys += hi - lo;
        }
        total += 2 * keys * hd * nh;
    }
    return total;
}

/* ---- forward -------------------------------------------------------------- */

static trv_status read_vec(const trv_mbert *mb, uint32_t off, float *dst)
{
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

trv_status trv_mbert_forward(const trv_mbert *mb, const uint32_t *ids,
                             const uint8_t *key_mask, uint32_t T, float *h,
                             void *arena, size_t arena_bytes,
                             void *tile_buf, uint32_t tile_bytes,
                             trv_mbert_hook hook, void *user)
{
    const trv_model *m;
    uint32_t H, I, hd, g, L, i, t, hh, c0;
    arena_t a;
    trv_tensor emb;
    trv_status st;

    if (!mb || !ids || !h || !arena || !tile_buf || T == 0) return TRV_ERR_ARG;
    if (((uintptr_t)arena & 3u) != 0) return TRV_ERR_ARG;
    if (layout(mb, T, (uint8_t *)arena, &a) > arena_bytes) return TRV_ERR_BUFFER;
    m = mb->m;
    H = mb->hidden;
    I = mb->inter;
    hd = mb->head_dim;
    g = mb->group;
    L = mb->n_layers;

    /* embeddings: one row per token, then LayerNorm */
    emb = q8_view(mb->emb_tok, mb->vocab, H);
    for (t = 0; t < T; ++t) {
        if (ids[t] >= mb->vocab) return TRV_ERR_ARG;
        st = trv_tensor_read_q8_row(m, &emb, ids[t], h + (size_t)t * H, a.xq_b);
        if (st != TRV_OK) return st;
    }
    if ((st = norm_inplace(mb, &a, h, T, mb->emb_norm_w, mb->emb_norm_b)) != TRV_OK) return st;
    if (hook) hook(user, -1, h, T, H);

    for (i = 0; i < L; ++i) {
        const trv_mbert_layer *ly = &mb->layers[i];
        trv_tensor wqkv = q8_view(ly->wqkv, 3 * H, H);
        trv_tensor wo = q8_view(ly->wo, H, H);
        trv_tensor wi = q8_view(ly->wi, 2 * I, H);
        trv_tensor wo_mlp = q8_view(ly->wo_mlp, H, I);

        /* ---- attention ---- */
        if ((st = norm_quant(mb, &a, h, T, ly->attn_norm_w, ly->attn_norm_b)) != TRV_OK)
            return st;
        for (hh = 0; hh < mb->n_heads; ++hh) {
            uint32_t r = hh * hd;
            st = trv_linear_q8g_stream(m, &wqkv, a.xq_in, a.xs_in, g, T, NULL,
                                       r, r + hd, a.q, hd, 0, tile_buf, tile_bytes);
            if (st != TRV_OK) return st;
            st = trv_linear_q8g_stream(m, &wqkv, a.xq_in, a.xs_in, g, T, NULL,
                                       H + r, H + r + hd, a.k, hd, 0, tile_buf, tile_bytes);
            if (st != TRV_OK) return st;
            st = trv_linear_q8g_stream(m, &wqkv, a.xq_in, a.xs_in, g, T, NULL,
                                       2 * H + r, 2 * H + r + hd, a.v, hd, 0, tile_buf,
                                       tile_bytes);
            if (st != TRV_OK) return st;
            trv_rope(a.q, T, 1, hd, hd, 0, ly->rope_theta);
            trv_rope(a.k, T, 1, hd, hd, 0, ly->rope_theta);
            trv_attention_head(a.q, a.k, a.v, hd, T, hd, ly->half_window, key_mask,
                               a.o, hd);
            trv_quantize_act_q8g(a.o, T, hd, hd, g, a.xq_b + r, H, a.xs_b + r / g, H / g);
        }
        st = trv_linear_q8g_stream(m, &wo, a.xq_b, a.xs_b, g, T, NULL, 0, H, h, H, 1,
                                   tile_buf, tile_bytes);
        if (st != TRV_OK) return st;

        /* ---- GeGLU MLP, one chunk of `g` hidden units at a time ---- */
        if ((st = norm_quant(mb, &a, h, T, ly->mlp_norm_w, ly->mlp_norm_b)) != TRV_OK)
            return st;
        for (c0 = 0; c0 < I; c0 += g) {
            st = trv_linear_q8g_stream(m, &wi, a.xq_in, a.xs_in, g, T, NULL, c0, c0 + g,
                                       a.ffn, 2 * g, 0, tile_buf, tile_bytes);
            if (st != TRV_OK) return st;
            st = trv_linear_q8g_stream(m, &wi, a.xq_in, a.xs_in, g, T, NULL, I + c0,
                                       I + c0 + g, a.ffn + g, 2 * g, 0, tile_buf,
                                       tile_bytes);
            if (st != TRV_OK) return st;
            trv_geglu(a.ffn, T, g, 2 * g, a.ffn, 2 * g);
            trv_quantize_act_q8g(a.ffn, T, g, 2 * g, g, a.xq_b + c0, I, a.xs_b + c0 / g,
                                 I / g);
        }
        st = trv_linear_q8g_stream(m, &wo_mlp, a.xq_b, a.xs_b, g, T, NULL, 0, H, h, H, 1,
                                   tile_buf, tile_bytes);
        if (st != TRV_OK) return st;

        if (hook) hook(user, (int32_t)i, h, T, H);
    }

    if ((st = norm_inplace(mb, &a, h, T, mb->final_norm_w, mb->final_norm_b)) != TRV_OK)
        return st;
    if (hook) hook(user, (int32_t)L, h, T, H);
    return TRV_OK;
}
