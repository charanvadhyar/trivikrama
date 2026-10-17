/*
 * Laya on the device from plain text: tokenizer + questions (.tlq file).
 * Copyright 2026 Vadhyar Sai Charan
 * SPDX-License-Identifier: Apache-2.0
 *
 * `trivi laya-questions` writes a .tlq with Laya's tokenizer and a set of
 * questions already formatted the way Laya formats them. With it, a device
 * turns a typed line into Laya's input rows by itself:
 *
 *   trv_laya_text_parse()     the .tlq, in place (no copies)
 *   trv_laya_tokenize()       text -> token ids (NFC, GPT-2 split, byte BPE)
 *   trv_laya_text_rows()      per question: head + ids[:room] + [SEP]
 *   trv_laya_run()            (trivikrama_laya.h) -> probabilities
 *   trv_laya_option_name()    probabilities -> option names
 *
 * The tokenizer matches Hugging Face `tokenizers` exactly (checked by
 * tests/check_tokenizer.py and, on the real Laya tokenizer, by
 * tools/check_laya_text.py).
 */
#ifndef TRIVIKRAMA_LAYA_TEXT_H
#define TRIVIKRAMA_LAYA_TEXT_H

#include "trivikrama_laya.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef TRV_LAYA_MAX_QUESTIONS
#define TRV_LAYA_MAX_QUESTIONS 64
#endif

typedef struct {
    uint32_t qtype;            /* TRV_LAYA_CHOICE / SCORE / NOUL */
    uint32_t n_options;
    uint32_t head_len;         /* tokens of the formatted question */
    uint32_t room;             /* text tokens that fit after it */
    const char *name;          /* question id, e.g. "guard.injection" (not 0-terminated) */
    uint32_t name_len;
    const uint32_t *markers;   /* [n_options] marker position of each slot */
    const uint32_t *slot_option; /* [n_options] which option sits in each slot */
    const uint32_t *head;      /* [head_len] token ids */
    const uint8_t *names;      /* option names: u32 length + bytes, 4-byte padded */
} trv_laya_question;

typedef struct {
    uint32_t flags, sep_id;
    const char *mask;          /* "[MASK]": replaced by a space in the text, as Laya does */
    uint32_t mask_len;
    uint32_t n_classes, n_decomp, n_ccc, n_comp, n_merges, n_added;
    const uint32_t *classes;   /* [n][3] lo, hi, class */
    const uint32_t *decomp;    /* [n][6] cp, len, 4 code points */
    const uint32_t *ccc;       /* [n][2] cp, combining class */
    const uint32_t *comp;      /* [n][3] a, b, composite */
    const uint32_t *byte_ids;  /* [256] */
    const uint32_t *merges;    /* [n][4] left, right, rank, result */
    const uint8_t *added;      /* records: u32 id, u32 flags, u32 len, bytes (padded) */
    uint8_t ascii_class[128];
    uint32_t n_questions;
    trv_laya_question questions[TRV_LAYA_MAX_QUESTIONS];
} trv_laya_text;

/* Parse a .tlq held in memory (4-byte aligned, little-endian); the struct
 * points into buf, which must stay alive. */
trv_status trv_laya_text_parse(trv_laya_text *lt, const void *buf, size_t len);

/* 32-bit words of work memory trv_laya_tokenize needs for len bytes of text. */
size_t trv_laya_tokenize_work_words(size_t len);

/*
 * UTF-8 text -> token ids, like Laya's tokenizer(text, add_special_tokens=False)
 * after replacing "[MASK]" with a space. Stops once max_ids ids are out
 * (Laya keeps the first `room` tokens of a text). Returns the number of ids,
 * or a negative trv_status.
 */
int32_t trv_laya_tokenize(const trv_laya_text *lt, const char *text, size_t len,
                          uint32_t *ids, uint32_t max_ids, uint32_t *work, size_t work_words);

/* Largest room of any question: tokenize at most this many ids. */
uint32_t trv_laya_text_max_room(const trv_laya_text *lt);

/* id_buf words needed by trv_laya_text_rows for n_text text tokens. */
size_t trv_laya_text_row_words(const trv_laya_text *lt, uint32_t n_text);

/* One row per question (rows[n_questions]); returns n_questions or < 0. */
int32_t trv_laya_text_rows(const trv_laya_text *lt, const uint32_t *text_ids, uint32_t n_text,
                           uint32_t *id_buf, size_t id_words, trv_laya_row *rows);

/* Name of option `option` (not slot) of a question; *len gets its length. */
const char *trv_laya_option_name(const trv_laya_question *q, uint32_t option, uint32_t *len);

#ifdef __cplusplus
}
#endif

#endif /* TRIVIKRAMA_LAYA_TEXT_H */
