/*
 * Laya's tokenizer and question rows on the device.
 * Copyright 2026 Vadhyar Sai Charan
 * SPDX-License-Identifier: Apache-2.0
 *
 * Byte-level BPE as in Hugging Face `tokenizers` (GPT-2 / ModernBERT style):
 *   1. added tokens ([CLS], [SEP], ...) are split out first, leftmost-longest
 *   2. NFC normalization (decompose, canonical order, compose; Hangul by rule)
 *   3. GPT-2 pre-tokenization:
 *        's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
 *   4. BPE on the UTF-8 bytes of each piece: merge the adjacent pair with the
 *      lowest rank (leftmost on ties) until no pair is in the merge table
 * All tables live in the .tlq file; lookups are binary searches, so nothing
 * is copied into RAM.
 */
#include "trivikrama_laya_text.h"

#include <string.h>

enum { CLS_OTHER = 0, CLS_LETTER = 1, CLS_NUMBER = 2, CLS_SPACE = 3 };

#define S_BASE 0xAC00u
#define L_BASE 0x1100u
#define V_BASE 0x1161u
#define T_BASE 0x11A7u
#define L_COUNT 19u
#define V_COUNT 21u
#define T_COUNT 28u
#define N_COUNT (V_COUNT * T_COUNT)
#define S_COUNT (L_COUNT * N_COUNT)

/* ---- parsing ---------------------------------------------------------------- */

typedef struct {
    const uint8_t *p, *end;
    int bad;
} reader;

static uint32_t rd_u32(reader *r)
{
    uint32_t v;
    if (r->end - r->p < 4) {
        r->bad = 1;
        return 0;
    }
    memcpy(&v, r->p, 4);
    r->p += 4;
    return v;
}

static const uint32_t *rd_words(reader *r, size_t n)
{
    const uint32_t *w = (const uint32_t *)(const void *)r->p;
    if (n > (size_t)(r->end - r->p) / 4) {
        r->bad = 1;
        return NULL;
    }
    r->p += 4 * n;
    return w;
}

static const uint8_t *rd_blob(reader *r, uint32_t *len)
{
    const uint8_t *b;
    *len = rd_u32(r);
    if (r->bad || *len > (size_t)(r->end - r->p)) {
        r->bad = 1;
        return NULL;
    }
    b = r->p;
    r->p += (*len + 3u) & ~3u;
    if (r->p > r->end) r->bad = 1;
    return b;
}

static uint32_t class_of(const trv_laya_text *lt, uint32_t c);

trv_status trv_laya_text_parse(trv_laya_text *lt, const void *buf, size_t len)
{
    reader r;
    uint32_t i, j, n;
    if (!lt || !buf || ((uintptr_t)buf & 3u)) return TRV_ERR_ARG;
    memset(lt, 0, sizeof *lt);
    r.p = (const uint8_t *)buf;
    r.end = r.p + len;
    r.bad = 0;
    if (len < 16 || memcmp(r.p, "TRLT", 4) != 0) return TRV_ERR_FORMAT;
    r.p += 4;
    if (rd_u32(&r) != 1) return TRV_ERR_FORMAT;
    lt->flags = rd_u32(&r);
    lt->sep_id = rd_u32(&r);
    lt->mask = (const char *)rd_blob(&r, &lt->mask_len);
    lt->n_classes = rd_u32(&r);
    lt->classes = rd_words(&r, 3 * (size_t)lt->n_classes);
    if (lt->flags & 1u) {
        lt->n_decomp = rd_u32(&r);
        lt->decomp = rd_words(&r, 6 * (size_t)lt->n_decomp);
        lt->n_ccc = rd_u32(&r);
        lt->ccc = rd_words(&r, 2 * (size_t)lt->n_ccc);
        lt->n_comp = rd_u32(&r);
        lt->comp = rd_words(&r, 3 * (size_t)lt->n_comp);
        for (i = 0; i < lt->n_decomp && !r.bad; ++i)
            if (lt->decomp[6 * i + 1] > 4) return TRV_ERR_FORMAT;
    }
    lt->byte_ids = rd_words(&r, 256);
    lt->n_merges = rd_u32(&r);
    lt->merges = rd_words(&r, 4 * (size_t)lt->n_merges);
    lt->n_added = rd_u32(&r);
    lt->added = r.p;
    for (i = 0; i < lt->n_added && !r.bad; ++i) {
        uint32_t l;
        rd_u32(&r);
        rd_u32(&r);
        rd_blob(&r, &l);
    }
    n = rd_u32(&r);
    if (r.bad) return TRV_ERR_FORMAT;
    if (n > TRV_LAYA_MAX_QUESTIONS) return TRV_ERR_BUFFER;
    lt->n_questions = n;
    for (i = 0; i < n && !r.bad; ++i) {
        trv_laya_question *q = &lt->questions[i];
        q->qtype = rd_u32(&r);
        q->n_options = rd_u32(&r);
        q->head_len = rd_u32(&r);
        q->room = rd_u32(&r);
        q->name = (const char *)rd_blob(&r, &q->name_len);
        if (r.bad || q->qtype > 2 || q->n_options == 0 || q->n_options > 255 ||
            q->head_len == 0)
            return TRV_ERR_FORMAT;
        q->markers = rd_words(&r, q->n_options);
        q->slot_option = rd_words(&r, q->n_options);
        q->head = rd_words(&r, q->head_len);
        q->names = r.p;
        for (j = 0; j < q->n_options && !r.bad; ++j) {
            uint32_t l;
            rd_blob(&r, &l);
        }
        if (r.bad) return TRV_ERR_FORMAT;
        for (j = 0; j < q->n_options; ++j)
            if (q->markers[j] >= q->head_len || q->slot_option[j] >= q->n_options)
                return TRV_ERR_FORMAT;
        if ((size_t)q->head_len + q->room + 1 > 65535u) return TRV_ERR_FORMAT;
    }
    if (r.bad) return TRV_ERR_FORMAT;
    for (i = 0; i < 128; ++i) lt->ascii_class[i] = (uint8_t)class_of(lt, i);
    return TRV_OK;
}

/* ---- table lookups ------------------------------------------------------------ */

static uint32_t class_of(const trv_laya_text *lt, uint32_t c)
{
    uint32_t lo = 0, hi = lt->n_classes;
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        const uint32_t *e = lt->classes + 3 * mid;
        if (c < e[0]) hi = mid;
        else if (c > e[1]) lo = mid + 1;
        else return e[2];
    }
    return CLS_OTHER;
}

static uint32_t cls(const trv_laya_text *lt, uint32_t c)
{
    return c < 128 ? lt->ascii_class[c] : class_of(lt, c);
}

static uint32_t ccc_of(const trv_laya_text *lt, uint32_t c)
{
    uint32_t lo = 0, hi = lt->n_ccc;
    if (c < 0x300) return 0;
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2, k = lt->ccc[2 * mid];
        if (c < k) hi = mid;
        else if (c > k) lo = mid + 1;
        else return lt->ccc[2 * mid + 1];
    }
    return 0;
}

static const uint32_t *decomp_of(const trv_laya_text *lt, uint32_t c)
{
    uint32_t lo = 0, hi = lt->n_decomp;
    if (c < 0xC0) return NULL;
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2, k = lt->decomp[6 * mid];
        if (c < k) hi = mid;
        else if (c > k) lo = mid + 1;
        else return lt->decomp + 6 * mid;
    }
    return NULL;
}

static uint32_t compose(const trv_laya_text *lt, uint32_t a, uint32_t b)
{
    uint32_t lo = 0, hi = lt->n_comp;
    if (a - L_BASE < L_COUNT && b - V_BASE < V_COUNT)
        return S_BASE + ((a - L_BASE) * V_COUNT + (b - V_BASE)) * T_COUNT;
    if (a - S_BASE < S_COUNT && (a - S_BASE) % T_COUNT == 0 && b > T_BASE &&
        b - T_BASE < T_COUNT)
        return a + (b - T_BASE);
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        const uint32_t *e = lt->comp + 3 * mid;
        if (a < e[0] || (a == e[0] && b < e[1])) hi = mid;
        else if (a > e[0] || b > e[1]) lo = mid + 1;
        else return e[2];
    }
    return 0;
}

/* merge of (a, b): rank in *rank, result id returned; UINT32_MAX = none */
static uint32_t merge_of(const trv_laya_text *lt, uint32_t a, uint32_t b, uint32_t *rank)
{
    uint32_t lo = 0, hi = lt->n_merges;
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        const uint32_t *e = lt->merges + 4 * mid;
        if (a < e[0] || (a == e[0] && b < e[1])) hi = mid;
        else if (a > e[0] || b > e[1]) lo = mid + 1;
        else {
            *rank = e[2];
            return e[3];
        }
    }
    return UINT32_MAX;
}

/* ---- UTF-8 ------------------------------------------------------------------- */

/* decode one code point; invalid bytes become U+FFFD one at a time */
static uint32_t utf8_next(const uint8_t *s, size_t n, size_t *i)
{
    uint32_t c = s[*i], need, cp, k;
    if (c < 0x80) {
        *i += 1;
        return c;
    }
    if (c >= 0xC2 && c <= 0xDF) { need = 1; cp = c & 0x1F; }
    else if (c >= 0xE0 && c <= 0xEF) { need = 2; cp = c & 0x0F; }
    else if (c >= 0xF0 && c <= 0xF4) { need = 3; cp = c & 0x07; }
    else { *i += 1; return 0xFFFD; }
    for (k = 1; k <= need; ++k) {
        uint32_t b;
        if (*i + k >= n) { *i += 1; return 0xFFFD; }
        b = s[*i + k];
        if ((b & 0xC0) != 0x80) { *i += 1; return 0xFFFD; }
        cp = (cp << 6) | (b & 0x3F);
    }
    if ((need == 2 && (cp < 0x800 || (cp >= 0xD800 && cp < 0xE000))) ||
        (need == 3 && (cp < 0x10000 || cp > 0x10FFFF))) {
        *i += 1;
        return 0xFFFD;
    }
    *i += need + 1;
    return cp;
}

static int is_space_cp(uint32_t c)
{
    return (c >= 9 && c <= 13) || c == 32 || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F ||
           c == 0x205F || c == 0x3000;
}

/* start of the code point that ends at pos (pos > start) */
static size_t utf8_prev(const uint8_t *s, size_t start, size_t pos)
{
    size_t p = pos - 1;
    while (p > start && (s[p] & 0xC0) == 0x80 && pos - p < 4) --p;
    return p;
}

/* ---- tokenizer ------------------------------------------------------------------ */

size_t trv_laya_tokenize_work_words(size_t len)
{
    /* text copy + code points (NFC may expand 4x) + BPE ids of one piece */
    return len / 4 + 2 + 4 * len + 4 + 16 * len + 16;
}

typedef struct {
    const trv_laya_text *lt;
    uint32_t *ids, max_ids, n;
    uint32_t *bpe;            /* scratch for one piece */
    size_t bpe_cap;
} out_t;

static void emit(out_t *o, uint32_t id)
{
    if (o->n < o->max_ids) o->ids[o->n] = id;
    o->n++;
}

static int put_utf8(uint32_t *dst, size_t cap, size_t *n, uint32_t c, const uint32_t *byte_ids)
{
    uint8_t b[4];
    size_t k, m;
    if (c < 0x80) { b[0] = (uint8_t)c; m = 1; }
    else if (c < 0x800) { b[0] = (uint8_t)(0xC0 | (c >> 6)); b[1] = (uint8_t)(0x80 | (c & 0x3F)); m = 2; }
    else if (c < 0x10000) {
        b[0] = (uint8_t)(0xE0 | (c >> 12)); b[1] = (uint8_t)(0x80 | ((c >> 6) & 0x3F));
        b[2] = (uint8_t)(0x80 | (c & 0x3F)); m = 3;
    } else {
        b[0] = (uint8_t)(0xF0 | (c >> 18)); b[1] = (uint8_t)(0x80 | ((c >> 12) & 0x3F));
        b[2] = (uint8_t)(0x80 | ((c >> 6) & 0x3F)); b[3] = (uint8_t)(0x80 | (c & 0x3F)); m = 4;
    }
    if (*n + m > cap) return -1;
    for (k = 0; k < m; ++k) dst[(*n)++] = byte_ids[b[k]];
    return 0;
}

/* BPE over one pre-token (code points cps[0..n)) */
static int bpe_piece(out_t *o, const uint32_t *cps, size_t n)
{
    const trv_laya_text *lt = o->lt;
    uint32_t *w = o->bpe;
    size_t m = 0, i;
    for (i = 0; i < n; ++i)
        if (put_utf8(w, o->bpe_cap, &m, cps[i], lt->byte_ids)) return -1;
    while (m > 1) {
        uint32_t best_rank = UINT32_MAX, best_id = 0, rank;
        size_t at = 0;
        for (i = 0; i + 1 < m; ++i) {
            uint32_t id = merge_of(lt, w[i], w[i + 1], &rank);
            if (id != UINT32_MAX && rank < best_rank) {
                best_rank = rank;
                best_id = id;
                at = i;
            }
        }
        if (best_rank == UINT32_MAX) break;
        w[at] = best_id;
        memmove(w + at + 1, w + at + 2, sizeof(uint32_t) * (m - at - 2));
        --m;
    }
    for (i = 0; i < m; ++i) emit(o, w[i]);
    return 0;
}

/* NFC of cps[0..n) in place (the buffer has room for the decomposition) */
static size_t nfc(const trv_laya_text *lt, const uint8_t *s, size_t len, uint32_t *cp,
                  size_t cap, int *err)
{
    size_t i = 0, n = 0, a, b, out;
    uint32_t starter_valid = 0, last = 0;
    size_t starter = 0;
    /* decompose while decoding */
    while (i < len) {
        uint32_t c = utf8_next(s, len, &i);
        if (!(lt->flags & 1u)) {
            if (n >= cap) { *err = 1; return 0; }
            cp[n++] = c;
            continue;
        }
        if (c - S_BASE < S_COUNT) {
            uint32_t x = c - S_BASE;
            if (n + 3 > cap) { *err = 1; return 0; }
            cp[n++] = L_BASE + x / N_COUNT;
            cp[n++] = V_BASE + (x % N_COUNT) / T_COUNT;
            if (x % T_COUNT) cp[n++] = T_BASE + x % T_COUNT;
        } else {
            const uint32_t *d = decomp_of(lt, c);
            uint32_t k;
            if (n + 4 > cap) { *err = 1; return 0; }
            if (d) for (k = 0; k < d[1]; ++k) cp[n++] = d[2 + k];
            else cp[n++] = c;
        }
    }
    if (!(lt->flags & 1u)) return n;
    /* canonical ordering: stable insertion sort of each run of non-starters */
    for (a = 0; a < n;) {
        if (!ccc_of(lt, cp[a])) { ++a; continue; }
        for (b = a; b < n && ccc_of(lt, cp[b]); ++b) {}
        {
            size_t x, y;
            for (x = a + 1; x < b; ++x) {
                uint32_t v = cp[x], kv = ccc_of(lt, v);
                for (y = x; y > a && ccc_of(lt, cp[y - 1]) > kv; --y) cp[y] = cp[y - 1];
                cp[y] = v;
            }
        }
        a = b;
    }
    /* canonical composition, writing in place behind the read position */
    for (a = 0, out = 0; a < n; ++a) {
        uint32_t c = cp[a], cc = ccc_of(lt, c);
        if (starter_valid) {
            int blocked = out - 1 != starter && (last == 0 || last >= cc);
            if (!blocked) {
                uint32_t x = compose(lt, cp[starter], c);
                if (x) {
                    cp[starter] = x;
                    continue;
                }
            }
        }
        if (cc == 0) {
            starter = out;
            starter_valid = 1;
        }
        last = cc;
        cp[out++] = c;
    }
    return out;
}

/* GPT-2 split + BPE of one stretch of text without added tokens */
static int encode_plain(out_t *o, const uint8_t *s, size_t len, uint32_t *cp, size_t cap)
{
    const trv_laya_text *lt = o->lt;
    int err = 0;
    size_t n = nfc(lt, s, len, cp, cap, &err), i = 0, j;
    if (err) return -1;
    while (i < n && o->n < o->max_ids) {
        uint32_t c = cp[i], nx = i + 1 < n ? cp[i + 1] : UINT32_MAX, k;
        if (c == 0x27 && (nx == 's' || nx == 't' || nx == 'm' || nx == 'd')) {
            j = i + 2;
        } else if (c == 0x27 && i + 2 < n &&
                   ((nx == 'r' && cp[i + 2] == 'e') || (nx == 'v' && cp[i + 2] == 'e') ||
                    (nx == 'l' && cp[i + 2] == 'l'))) {
            j = i + 3;
        } else if (c == 0x20 && nx != UINT32_MAX && (k = cls(lt, nx)) != CLS_SPACE) {
            for (j = i + 2; j < n && cls(lt, cp[j]) == k; ++j) {}
        } else if ((k = cls(lt, c)) != CLS_SPACE) {
            for (j = i + 1; j < n && cls(lt, cp[j]) == k; ++j) {}
        } else {
            for (j = i; j < n && cls(lt, cp[j]) == CLS_SPACE; ++j) {}
            if (j < n && j - i >= 2) --j; /* the last space starts the next word */
        }
        if (bpe_piece(o, cp + i, j - i)) return -1;
        i = j;
    }
    return 0;
}

int32_t trv_laya_tokenize(const trv_laya_text *lt, const char *text, size_t len,
                          uint32_t *ids, uint32_t max_ids, uint32_t *work, size_t work_words)
{
    uint8_t *t;
    uint32_t *cp;
    size_t tl = 0, i = 0, start = 0, cap, k;
    out_t o;

    if (!lt || (!text && len) || !ids || !work) return TRV_ERR_ARG;
    if (work_words < trv_laya_tokenize_work_words(len)) return TRV_ERR_BUFFER;
    t = (uint8_t *)work;
    cp = work + len / 4 + 2;
    cap = 4 * len + 4;
    o.lt = lt;
    o.ids = ids;
    o.max_ids = max_ids;
    o.n = 0;
    o.bpe = cp + cap;
    o.bpe_cap = 16 * len + 16;

    /* Laya: text.replace(mask_token, " ") */
    while (i < len) {
        if (lt->mask_len && i + lt->mask_len <= len &&
            memcmp(text + i, lt->mask, lt->mask_len) == 0) {
            t[tl++] = ' ';
            i += lt->mask_len;
        } else {
            t[tl++] = (uint8_t)text[i++];
        }
    }

    /* split out added tokens: leftmost-longest, with lstrip / rstrip */
    for (i = 0; i < tl && o.n < max_ids;) {
        const uint8_t *rec = lt->added, *best = NULL;
        uint32_t best_len = 0, best_id = 0, best_flags = 0;
        for (k = 0; k < lt->n_added; ++k) {
            uint32_t id, fl, l;
            memcpy(&id, rec, 4);
            memcpy(&fl, rec + 4, 4);
            memcpy(&l, rec + 8, 4);
            if (l && l > best_len && i + l <= tl && memcmp(t + i, rec + 12, l) == 0) {
                best = rec;
                best_len = l;
                best_id = id;
                best_flags = fl;
            }
            rec += 12 + ((l + 3u) & ~3u);
        }
        if (!best) {
            ++i;
            continue;
        }
        {
            size_t a = i, b = i + best_len;
            if (best_flags & 1u) { /* lstrip: swallow whitespace before it */
                while (a > start) {
                    size_t p = utf8_prev(t, start, a), q = p;
                    if (!is_space_cp(utf8_next(t, tl, &q))) break;
                    a = p;
                }
            }
            if (best_flags & 2u) { /* rstrip */
                while (b < tl) {
                    size_t q = b;
                    if (!is_space_cp(utf8_next(t, tl, &q))) break;
                    b = q;
                }
            }
            if (a > start && encode_plain(&o, t + start, a - start, cp, cap)) return TRV_ERR_BUFFER;
            emit(&o, best_id);
            start = i = b;
        }
    }
    if (start < tl && o.n < max_ids && encode_plain(&o, t + start, tl - start, cp, cap))
        return TRV_ERR_BUFFER;
    return (int32_t)(o.n < max_ids ? o.n : max_ids);
}

/* ---- rows --------------------------------------------------------------------------- */

uint32_t trv_laya_text_max_room(const trv_laya_text *lt)
{
    uint32_t i, m = 0;
    for (i = 0; i < lt->n_questions; ++i)
        if (lt->questions[i].room > m) m = lt->questions[i].room;
    return m;
}

size_t trv_laya_text_row_words(const trv_laya_text *lt, uint32_t n_text)
{
    size_t i, w = 0;
    for (i = 0; i < lt->n_questions; ++i) {
        const trv_laya_question *q = &lt->questions[i];
        w += q->head_len + (n_text < q->room ? n_text : q->room) + 1;
    }
    return w;
}

int32_t trv_laya_text_rows(const trv_laya_text *lt, const uint32_t *text_ids, uint32_t n_text,
                           uint32_t *id_buf, size_t id_words, trv_laya_row *rows)
{
    uint32_t i;
    size_t off = 0;
    if (!lt || (!text_ids && n_text) || !id_buf || !rows) return TRV_ERR_ARG;
    if (id_words < trv_laya_text_row_words(lt, n_text)) return TRV_ERR_BUFFER;
    for (i = 0; i < lt->n_questions; ++i) {
        const trv_laya_question *q = &lt->questions[i];
        uint32_t keep = n_text < q->room ? n_text : q->room, *dst = id_buf + off;
        memcpy(dst, q->head, sizeof(uint32_t) * q->head_len);
        if (keep) memcpy(dst + q->head_len, text_ids, sizeof(uint32_t) * keep);
        dst[q->head_len + keep] = lt->sep_id;
        rows[i].qtype = (uint8_t)q->qtype;
        rows[i].n_options = (uint8_t)q->n_options;
        rows[i].n_tokens = (uint16_t)(q->head_len + keep + 1);
        rows[i].markers = q->markers;
        rows[i].ids = dst;
        off += q->head_len + keep + 1;
    }
    return (int32_t)lt->n_questions;
}

const char *trv_laya_option_name(const trv_laya_question *q, uint32_t option, uint32_t *len)
{
    const uint8_t *p = q->names;
    uint32_t i, l = 0;
    if (option >= q->n_options) return NULL;
    for (i = 0;; ++i) {
        memcpy(&l, p, 4);
        if (i == option) break;
        p += 4 + ((l + 3u) & ~3u);
    }
    if (len) *len = l;
    return (const char *)(p + 4);
}
