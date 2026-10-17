/*
 * Laya decision head.
 * Copyright 2026 Vadhyar Sai Charan
 * SPDX-License-Identifier: Apache-2.0
 *
 * After the encoder (trv_modernbert.c), on the packed hidden states of all
 * question rows:
 *
 *   h += type_emb[qtype of the row]
 *   2 x pre-norm transformer layer (PyTorch nn.TransformerEncoderLayer,
 *       norm_first=True, ReLU):
 *       h += out_proj(attention(norm1(h)))   global, each row attends to itself
 *       h += linear2(relu(linear1(norm2(h))))
 *   for every option marker m:
 *       logit = l2 . gelu(l1 . layernorm(h[m]) + b1) + b2
 *   probs of a row = softmax(logits / temperature[type, #options])
 *
 * Same memory pattern as the encoder: weights stream tile by tile, the
 * attention runs one head at a time, the FFN one chunk at a time.
 */
#include "trivikrama_laya.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

enum { LCFG_LAYERS, LCFG_HEADS, LCFG_FF, LCFG_EPS, LCFG_COUNT };

static int is_vec(const trv_tensor *t, uint32_t n)
{
    return t->dtype == TRV_DTYPE_F32 && trv_tensor_numel(t) == n;
}

static int is_q8(const trv_tensor *t, uint32_t N, uint32_t K)
{
    return t->dtype == TRV_DTYPE_Q8_ROWS && t->ndim == 2 && t->dims[0] == N &&
           t->dims[1] == K && t->nbytes == 4u * N + N * K;
}

trv_status trv_laya_load(trv_laya *ly, const trv_model *m)
{
    char name[TRV_MAX_NAME + 1];
    trv_tensor t;
    float cfg[LCFG_COUNT];
    uint32_t pos, i, H, F, L, n_temps = 0;
    trv_status st;

    if (!ly || !m) return TRV_ERR_ARG;
    memset(ly, 0, sizeof *ly);
    if ((st = trv_mbert_load(&ly->enc, m)) != TRV_OK) return st;
    H = ly->hidden = ly->enc.hidden;
    ly->group = ly->enc.group;

    if ((st = trv_model_find(m, "laya.config", &t)) != TRV_OK) return st;
    if (t.dtype != TRV_DTYPE_F32 || trv_tensor_numel(&t) < LCFG_COUNT) return TRV_ERR_FORMAT;
    if ((st = trv_tensor_read_f32(m, &t, 0, cfg, LCFG_COUNT)) != TRV_OK) return st;
    L = ly->n_layers = (uint32_t)cfg[LCFG_LAYERS];
    ly->n_heads = (uint32_t)cfg[LCFG_HEADS];
    F = ly->ff = (uint32_t)cfg[LCFG_FF];
    ly->eps = cfg[LCFG_EPS];
    if (L > TRV_LAYA_MAX_HEAD_LAYERS || ly->n_heads == 0 || H % ly->n_heads)
        return TRV_ERR_FORMAT;
    ly->head_dim = H / ly->n_heads;
    if (ly->head_dim % ly->group || F % ly->group) return TRV_ERR_FORMAT;

    pos = m->table_offset;
    for (i = 0; i < m->n_tensors; ++i) {
        const char *s;
        if ((st = trv_model_next(m, &pos, name, &t)) != TRV_OK) return st;
        if (strncmp(name, "laya.", 5) != 0) continue;
        s = name + 5;
        if (strcmp(s, "temps") == 0) {
            if (!is_vec(&t, 15)) return TRV_ERR_FORMAT;
            if ((st = trv_tensor_read_f32(m, &t, 0, ly->temps, 15)) != TRV_OK) return st;
            n_temps = 15;
        } else if (strcmp(s, "type_emb") == 0) {
            if (!is_vec(&t, 3 * H)) return TRV_ERR_FORMAT;
            ly->type_emb = t.offset;
        } else if (strcmp(s, "score.ln.w") == 0) {
            if (!is_vec(&t, H)) return TRV_ERR_FORMAT;
            ly->ln_w = t.offset;
        } else if (strcmp(s, "score.ln.b") == 0) {
            if (!is_vec(&t, H)) return TRV_ERR_FORMAT;
            ly->ln_b = t.offset;
        } else if (strcmp(s, "score.l1") == 0) {
            if (!is_q8(&t, H, H)) return TRV_ERR_FORMAT;
            ly->l1 = t;
        } else if (strcmp(s, "score.l1_b") == 0) {
            if (!is_vec(&t, H)) return TRV_ERR_FORMAT;
            ly->l1_b = t.offset;
        } else if (strcmp(s, "score.l2") == 0) {
            if (!is_vec(&t, H)) return TRV_ERR_FORMAT;
            ly->l2 = t.offset;
        } else if (strcmp(s, "score.l2_b") == 0) {
            if (!is_vec(&t, 1)) return TRV_ERR_FORMAT;
            ly->l2_b = t.offset;
        } else if (s[0] == 'h') {
            char *dot;
            unsigned long li = strtoul(s + 1, &dot, 10);
            trv_laya_layer *ll;
            if (*dot != '.' || li >= L) continue;
            ll = &ly->layers[li];
            ++dot;
#define MAT(field, N, K)                                                       \
    do {                                                                       \
        if (!is_q8(&t, (N), (K))) return TRV_ERR_FORMAT;                       \
        ll->field = t;                                                         \
    } while (0)
#define VEC(field, n)                                                          \
    do {                                                                       \
        if (!is_vec(&t, (n))) return TRV_ERR_FORMAT;                           \
        ll->field = t.offset;                                                  \
    } while (0)
            if (strcmp(dot, "qkv") == 0) MAT(qkv, 3 * H, H);
            else if (strcmp(dot, "out") == 0) MAT(out, H, H);
            else if (strcmp(dot, "ff1") == 0) MAT(ff1, F, H);
            else if (strcmp(dot, "ff2") == 0) MAT(ff2, H, F);
            else if (strcmp(dot, "qkv_b") == 0) VEC(qkv_b, 3 * H);
            else if (strcmp(dot, "out_b") == 0) VEC(out_b, H);
            else if (strcmp(dot, "ff1_b") == 0) VEC(ff1_b, F);
            else if (strcmp(dot, "ff2_b") == 0) VEC(ff2_b, H);
            else if (strcmp(dot, "n1.w") == 0) VEC(n1_w, H);
            else if (strcmp(dot, "n1.b") == 0) VEC(n1_b, H);
            else if (strcmp(dot, "n2.w") == 0) VEC(n2_w, H);
            else if (strcmp(dot, "n2.b") == 0) VEC(n2_b, H);
#undef MAT
#undef VEC
        }
    }
    if (n_temps != 15 || !ly->type_emb || !ly->ln_w || !ly->ln_b || !ly->l1.offset ||
        !ly->l1_b || !ly->l2 || !ly->l2_b)
        return TRV_ERR_NOT_FOUND;
    for (i = 0; i < L; ++i) {
        const trv_laya_layer *ll = &ly->layers[i];
        if (!ll->qkv.offset || !ll->out.offset || !ll->ff1.offset || !ll->ff2.offset ||
            !ll->qkv_b || !ll->out_b || !ll->ff1_b || !ll->ff2_b || !ll->n1_w || !ll->n1_b ||
            !ll->n2_w || !ll->n2_b)
            return TRV_ERR_NOT_FOUND;
    }
    return TRV_OK;
}

void trv_laya_totals(const trv_laya_row *rows, uint32_t n_rows, uint32_t *T,
                     uint32_t *n_options)
{
    uint32_t r, t = 0, k = 0;
    for (r = 0; r < n_rows; ++r) {
        t += rows[r].n_tokens;
        k += rows[r].n_options;
    }
    if (T) *T = t;
    if (n_options) *n_options = k;
}

float trv_laya_temperature(const trv_laya *ly, uint32_t qtype, uint32_t k)
{
    uint32_t b = k <= 2 ? 0 : k <= 5 ? 1 : k <= 10 ? 2 : 3;
    if (qtype > 2) return 1.0f;
    return ly->temps[3 + 4 * qtype + b];
}

/* ---- head arena -------------------------------------------------------- */

typedef struct {
    int8_t *xq_in;  /* [T x H] */
    int8_t *xq_b;   /* [T x max(H, F)] */
    float *xs_in;   /* [T x H/g] */
    float *xs_b;    /* [T x max(H, F)/g] */
    float *q, *k, *v, *o; /* [T x hd] */
    float *ffn;     /* [T x g] */
    float *gamma, *beta, *vec; /* [H] */
    float *bias;    /* [3H + F + 2H]: qkv, ff1, out, ff2 biases of one layer */
    float *mk;      /* [n_options x H] gathered markers, then scorer hidden */
    int8_t *mq;     /* [n_options x H] */
    float *ms;      /* [n_options x H/g] */
    uint32_t *lens; /* [n_rows] */
} head_arena;

static size_t up4(size_t n) { return (n + 3u) & ~(size_t)3u; }

static size_t head_layout(const trv_laya *ly, uint32_t T, uint32_t nk, uint32_t n_rows,
                          uint8_t *base, head_arena *a)
{
    size_t H = ly->hidden, F = ly->ff, W = F > H ? F : H, g = ly->group, hd = ly->head_dim;
    size_t B = 5 * H + F, off = 0;
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
    TAKE(ffn, float, T * g);
    TAKE(gamma, float, H);
    TAKE(beta, float, H);
    TAKE(vec, float, H);
    TAKE(bias, float, B);
    TAKE(mk, float, nk * H);
    TAKE(mq, int8_t, nk * H);
    TAKE(ms, float, nk * (H / g));
    TAKE(lens, uint32_t, n_rows);
#undef TAKE
    return off;
}

size_t trv_laya_workspace_bytes(const trv_laya *ly, uint32_t T, uint32_t n_options,
                                uint32_t n_rows)
{
    size_t e = trv_mbert_workspace_bytes(&ly->enc, T);
    size_t hb = head_layout(ly, T, n_options, n_rows, NULL, NULL);
    /* + the packed ids and row lengths the encoder is fed */
    return (e > hb ? e : hb) + 4u * ((size_t)T + n_rows);
}

static uint32_t head_scratch(const trv_laya *ly)
{
    uint32_t a = trv_linear_scratch_bytes(TRV_DTYPE_Q8_ROWS, ly->hidden, ly->group);
    uint32_t b = trv_linear_scratch_bytes(TRV_DTYPE_Q8_ROWS, ly->ff, ly->group);
    return a > b ? a : b;
}

uint32_t trv_laya_min_tile_bytes(const trv_laya *ly)
{
    uint32_t e = trv_mbert_min_tile_bytes(&ly->enc);
    uint32_t K = ly->ff > ly->hidden ? ly->ff : ly->hidden;
    uint32_t hb = head_scratch(ly) + trv_weight_row_bytes(TRV_DTYPE_Q8_ROWS, K);
    return e > hb ? e : hb;
}

/* ---- head forward --------------------------------------------------------- */

static trv_status read_f32(const trv_laya *ly, uint32_t off, float *dst, uint32_t n)
{
    const trv_source *src = ly->enc.m->src;
    trv_linear_drain(src);
    return src->read(src->ctx, off, dst, 4u * n) == 0 ? TRV_OK : TRV_ERR_IO;
}

static trv_status norm_quant(const trv_laya *ly, head_arena *a, const float *x, uint32_t T,
                             uint32_t w_off, uint32_t b_off, int8_t *xq, float *xs)
{
    trv_status st;
    if ((st = read_f32(ly, w_off, a->gamma, ly->hidden)) != TRV_OK) return st;
    if ((st = read_f32(ly, b_off, a->beta, ly->hidden)) != TRV_OK) return st;
    trv_layernorm_quantize(x, T, ly->hidden, ly->hidden, a->gamma, a->beta, ly->eps,
                           ly->group, xq, xs);
    return TRV_OK;
}

static trv_status head_layer(const trv_laya *ly, const trv_laya_layer *ll, head_arena *a,
                             float *h, uint32_t T, uint32_t n_rows,
                             void *tile, uint32_t tile_bytes)
{
    const trv_model *m = ly->enc.m;
    uint32_t H = ly->hidden, F = ly->ff, g = ly->group, hd = ly->head_dim, hh, r, c0;
    float *b_qkv = a->bias, *b_ff1 = b_qkv + 3 * H, *b_out = b_ff1 + F, *b_ff2 = b_out + H;
    trv_status st;

    /* every small read happens before the first streamed call: a plain read
     * would cancel a tile that is already prefetching */
    if ((st = read_f32(ly, ll->qkv_b, b_qkv, 3 * H)) != TRV_OK) return st;
    if ((st = read_f32(ly, ll->ff1_b, b_ff1, F)) != TRV_OK) return st;
    if ((st = read_f32(ly, ll->out_b, b_out, H)) != TRV_OK) return st;
    if ((st = read_f32(ly, ll->ff2_b, b_ff2, H)) != TRV_OK) return st;

    /* ---- self-attention ---- */
    if ((st = norm_quant(ly, a, h, T, ll->n1_w, ll->n1_b, a->xq_in, a->xs_in)) != TRV_OK)
        return st;
    for (hh = 0; hh < ly->n_heads; ++hh) {
        uint32_t c = hh * hd, start = 0;
        trv_linear_set_next(&ll->qkv, g, H + c, H + c + hd);
        st = trv_linear_stream(m, &ll->qkv, a->xq_in, a->xs_in, g, T, b_qkv, c, c + hd,
                               a->q, hd, 0, tile, tile_bytes);
        if (st != TRV_OK) return st;
        trv_linear_set_next(&ll->qkv, g, 2 * H + c, 2 * H + c + hd);
        st = trv_linear_stream(m, &ll->qkv, a->xq_in, a->xs_in, g, T, b_qkv, H + c,
                               H + c + hd, a->k, hd, 0, tile, tile_bytes);
        if (st != TRV_OK) return st;
        if (hh + 1 < ly->n_heads)
            trv_linear_set_next(&ll->qkv, g, c + hd, c + 2 * hd);
        else
            trv_linear_set_next(&ll->out, g, 0, H);
        st = trv_linear_stream(m, &ll->qkv, a->xq_in, a->xs_in, g, T, b_qkv, 2 * H + c,
                               2 * H + c + hd, a->v, hd, 0, tile, tile_bytes);
        if (st != TRV_OK) return st;
        for (r = 0; r < n_rows; ++r) { /* every question row attends only to itself */
            uint32_t n = a->lens[r];
            if (n)
                trv_attention_head_pos(a->q + (size_t)start * hd, a->k + (size_t)start * hd,
                                       a->v + (size_t)start * hd, hd, n, hd, -1, NULL, NULL,
                                       a->o + (size_t)start * hd, hd);
            start += n;
        }
        trv_quantize_act_q8g(a->o, T, hd, hd, g, a->xq_b + c, H, a->xs_b + c / g, H / g);
    }
    st = trv_linear_stream(m, &ll->out, a->xq_b, a->xs_b, g, T, b_out, 0, H, h, H, 1, tile,
                           tile_bytes);
    if (st != TRV_OK) return st;

    /* ---- ReLU feed-forward, one chunk of g units at a time ---- */
    if ((st = norm_quant(ly, a, h, T, ll->n2_w, ll->n2_b, a->xq_in, a->xs_in)) != TRV_OK)
        return st;
    for (c0 = 0; c0 < F; c0 += g) {
        size_t i, n = (size_t)T * g;
        if (c0 + g < F)
            trv_linear_set_next(&ll->ff1, g, c0 + g, c0 + 2 * g);
        else
            trv_linear_set_next(&ll->ff2, g, 0, H);
        st = trv_linear_stream(m, &ll->ff1, a->xq_in, a->xs_in, g, T, b_ff1, c0, c0 + g,
                               a->ffn, g, 0, tile, tile_bytes);
        if (st != TRV_OK) return st;
        for (i = 0; i < n; ++i)
            if (a->ffn[i] < 0.0f) a->ffn[i] = 0.0f;
        trv_quantize_act_q8g(a->ffn, T, g, g, g, a->xq_b + c0, F, a->xs_b + c0 / g, F / g);
    }
    return trv_linear_stream(m, &ll->ff2, a->xq_b, a->xs_b, g, T, b_ff2, 0, H, h, H, 1,
                             tile, tile_bytes);
}

static trv_status head_forward(const trv_laya *ly, const trv_laya_row *rows, uint32_t n_rows,
                               float *h, uint32_t T, uint32_t nk, float *logits, float *probs,
                               head_arena *a, void *tile, uint32_t tile_bytes)
{
    const trv_model *m = ly->enc.m;
    uint32_t H = ly->hidden, g = ly->group, r, i, t, j, start, kk;
    float b2;
    trv_status st;

    /* type embedding, added to every token of the row */
    for (r = 0, start = 0; r < n_rows; ++r) {
        a->lens[r] = rows[r].n_tokens;
        if (rows[r].qtype > 2) return TRV_ERR_ARG;
        st = read_f32(ly, ly->type_emb + 4u * H * rows[r].qtype, a->vec, H);
        if (st != TRV_OK) return st;
        for (t = start; t < start + rows[r].n_tokens; ++t)
            for (j = 0; j < H; ++j) h[(size_t)t * H + j] += a->vec[j];
        start += rows[r].n_tokens;
    }

    for (i = 0; i < ly->n_layers; ++i) {
        st = head_layer(ly, &ly->layers[i], a, h, T, n_rows, tile, tile_bytes);
        if (st != TRV_OK) return st;
    }

    /* gather the option markers */
    for (r = 0, start = 0, kk = 0; r < n_rows; ++r) {
        for (j = 0; j < rows[r].n_options; ++j, ++kk) {
            uint32_t p = rows[r].markers[j];
            if (p >= rows[r].n_tokens) return TRV_ERR_ARG;
            memcpy(a->mk + (size_t)kk * H, h + (size_t)(start + p) * H, sizeof(float) * H);
        }
        start += rows[r].n_tokens;
    }

    /* scorer: LayerNorm -> Linear -> GELU -> dot */
    if ((st = norm_quant(ly, a, a->mk, nk, ly->ln_w, ly->ln_b, a->mq, a->ms)) != TRV_OK)
        return st;
    if ((st = read_f32(ly, ly->l1_b, a->bias, H)) != TRV_OK) return st;
    st = trv_linear_stream(m, &ly->l1, a->mq, a->ms, g, nk, a->bias, 0, H, a->mk, H, 0, tile,
                           tile_bytes);
    if (st != TRV_OK) return st;
    trv_gelu(a->mk, nk * H);
    if ((st = read_f32(ly, ly->l2, a->vec, H)) != TRV_OK) return st;
    if ((st = read_f32(ly, ly->l2_b, &b2, 1)) != TRV_OK) return st;

    for (r = 0, kk = 0; r < n_rows; ++r) {
        uint32_t k = rows[r].n_options;
        float temp = trv_laya_temperature(ly, rows[r].qtype, k), zmax = -INFINITY, sum = 0.0f;
        float *p = probs + kk;
        for (j = 0; j < k; ++j) {
            const float *s = a->mk + (size_t)(kk + j) * H;
            float acc = b2;
            for (i = 0; i < H; ++i) acc += s[i] * a->vec[i];
            if (logits) logits[kk + j] = acc;
            p[j] = acc / temp;
            if (p[j] > zmax) zmax = p[j];
        }
        for (j = 0; j < k; ++j) {
            p[j] = expf(p[j] - zmax);
            sum += p[j];
        }
        for (j = 0; j < k; ++j) p[j] /= sum;
        kk += k;
    }
    return TRV_OK;
}

/* the encoder may stop early, but must not drop tokens (markers point into rows) */
typedef struct {
    trv_mbert_hook hook;
    void *user;
} hook_wrap;

static void no_prune(void *user, int32_t layer, const float *h, uint32_t T, uint32_t H,
                     trv_mbert_ctl *ctl)
{
    hook_wrap *w = (hook_wrap *)user;
    uint32_t t;
    w->hook(w->user, layer, h, T, H, ctl);
    if (ctl && ctl->keep)
        for (t = 0; t < T; ++t) ctl->keep[t] = 1;
}

trv_status trv_laya_run(const trv_laya *ly, const trv_laya_row *rows, uint32_t n_rows,
                        float *h, float *logits, float *probs,
                        void *arena, size_t arena_bytes,
                        void *tile_buf, uint32_t tile_bytes,
                        trv_mbert_hook hook, void *user)
{
    uint32_t T, nk, r, t, start, *lens, *ids;
    hook_wrap w;
    head_arena a;
    trv_status st;
    size_t need, extra;

    if (!ly || !rows || !n_rows || !h || !probs || !arena || !tile_buf) return TRV_ERR_ARG;
    if (((uintptr_t)arena & 3u) != 0) return TRV_ERR_ARG;
    trv_laya_totals(rows, n_rows, &T, &nk);
    if (T == 0 || nk == 0) return TRV_ERR_ARG;
    for (r = 0; r < n_rows; ++r) { /* check everything before the long encoder pass */
        uint32_t j;
        if (rows[r].n_options == 0 || rows[r].n_tokens == 0 || !rows[r].ids ||
            !rows[r].markers || rows[r].qtype > 2)
            return TRV_ERR_ARG;
        for (j = 0; j < rows[r].n_options; ++j)
            if (rows[r].markers[j] >= rows[r].n_tokens) return TRV_ERR_ARG;
        for (j = 0; j < rows[r].n_tokens; ++j)
            if (rows[r].ids[j] >= ly->enc.vocab) return TRV_ERR_ARG;
    }

    /* the encoder wants the ids packed and the row lengths; borrow the end of
     * the arena for them (the encoder's own layout comes first) */
    extra = 4u * ((size_t)T + n_rows);
    need = trv_laya_workspace_bytes(ly, T, nk, n_rows) - extra;
    if (need + extra > arena_bytes) return TRV_ERR_BUFFER;
    ids = (uint32_t *)(void *)((uint8_t *)arena + need);
    lens = ids + T;
    for (r = 0, start = 0; r < n_rows; ++r) {
        for (t = 0; t < rows[r].n_tokens; ++t) ids[start + t] = rows[r].ids[t];
        lens[r] = rows[r].n_tokens;
        start += rows[r].n_tokens;
    }

    w.hook = hook;
    w.user = user;
    st = trv_mbert_forward_batch(&ly->enc, ids, lens, n_rows, NULL, h, NULL, NULL, arena, need,
                                 tile_buf, tile_bytes, hook ? no_prune : NULL, &w);
    if (st != TRV_OK) return st;

    head_layout(ly, T, nk, n_rows, (uint8_t *)arena, &a);
    trv_linear_reserve_scratch(head_scratch(ly));
    st = head_forward(ly, rows, n_rows, h, T, nk, logits, probs, &a, tile_buf, tile_bytes);
    trv_linear_set_next(NULL, 0, 0, 0);
    trv_linear_drain(ly->enc.m->src);
    trv_linear_reserve_scratch(0);
    return st;
}


int32_t trv_laya_parse_request(const void *buf, size_t len, trv_laya_row *rows,
                               uint32_t max_rows)
{
    const uint8_t *p = (const uint8_t *)buf, *end = p + len;
    uint32_t hdr[3], n, r;
    if (!buf || !rows || len < 12 || ((uintptr_t)buf & 3u)) return TRV_ERR_ARG;
    memcpy(hdr, p, 12);
    if (memcmp(p, "TRLQ", 4) != 0 || hdr[1] != 1) return TRV_ERR_FORMAT;
    n = hdr[2];
    if (n > max_rows) return TRV_ERR_BUFFER;
    p += 12;
    for (r = 0; r < n; ++r) {
        uint16_t T;
        if (end - p < 4) return TRV_ERR_FORMAT;
        rows[r].qtype = p[0];
        rows[r].n_options = p[1];
        memcpy(&T, p + 2, 2);
        rows[r].n_tokens = T;
        p += 4;
        if ((size_t)(end - p) < 4u * ((size_t)rows[r].n_options + T)) return TRV_ERR_FORMAT;
        rows[r].markers = (const uint32_t *)(const void *)p;
        p += 4u * rows[r].n_options;
        rows[r].ids = (const uint32_t *)(const void *)p;
        p += 4u * T;
    }
    return (int32_t)n;
}
