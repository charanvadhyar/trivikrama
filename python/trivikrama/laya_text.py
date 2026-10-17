"""Text in, answers out, on the device: tokenizer + questions in one small file.

The device needs two files:
    laya8.trv   the model                     (trivi export-laya)
    guard.tlq   tokenizer + questions          (trivi laya-questions)

A .tlq holds everything that turns a typed line into Laya's input rows:
  * the tokenizer (byte-level BPE as in Laya's tokenizer.json: NFC
    normalization, GPT-2 pre-tokenization, merges, added tokens), with the
    Unicode tables it needs (letter/number/space classes, NFC data)
  * every question already formatted the way Laya formats it
    ([CLS] <type> question: ... [SEP] [MASK] opt0 [MASK] opt1 ... [SEP]),
    since that part never depends on the text; plus how many text tokens fit
    after it, and the option names for printing answers

On the device a line becomes, per question:  head + text_ids[:room] + [SEP],
which is exactly what laya's build_sequence makes for a string input.

`TextTokenizer` is a Python mirror of the C tokenizer (core/src/trv_laya_text.c),
used by the tests to check C == mirror == Hugging Face.

Copyright 2026 Vadhyar Sai Charan
SPDX-License-Identifier: Apache-2.0
"""

import json
import struct
import unicodedata

MAGIC = b"TRLT"
VERSION = 1
FLAG_NFC = 1
NO_ID = 0xFFFFFFFF  # byte with no token (never produced by valid UTF-8)

CLS_OTHER, CLS_LETTER, CLS_NUMBER, CLS_SPACE = 0, 1, 2, 3
WHITESPACE = [9, 10, 11, 12, 13, 32, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B),
              0x2028, 0x2029, 0x202F, 0x205F, 0x3000]

S_BASE, L_BASE, V_BASE, T_BASE = 0xAC00, 0x1100, 0x1161, 0x11A7
L_COUNT, V_COUNT, T_COUNT = 19, 21, 28
N_COUNT, S_COUNT = V_COUNT * T_COUNT, 19 * 21 * 28


# ---------------------------------------------------------------------------
# GPT-2 byte <-> unicode mapping used by ByteLevel
# ---------------------------------------------------------------------------


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


# ---------------------------------------------------------------------------
# Unicode tables
# ---------------------------------------------------------------------------


def char_classes(empirical=True):
    """[(lo, hi, cls)] ranges for letter / number / whitespace (others are 0).

    With `tokenizers` installed, the classes are measured from its own
    pre-tokenizer, so they match the Unicode version the real tokenizer uses
    exactly; otherwise Python's unicodedata is used.
    """
    ws = set(WHITESPACE)
    classify = None
    if empirical:
        try:
            from tokenizers import pre_tokenizers

            pt = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
            n = lambda s: len(pt.pre_tokenize_str(s))  # noqa: E731

            def classify(c):
                ch = chr(c)
                if n("a" + ch) == 1:
                    return CLS_LETTER
                if n("1" + ch) == 1:
                    return CLS_NUMBER
                if n("!" + ch) == 1:
                    return CLS_OTHER
                return CLS_SPACE
        except ImportError:
            classify = None
    if classify is None:
        def classify(c):
            if c in ws:
                return CLS_SPACE
            cat = unicodedata.category(chr(c))
            return CLS_LETTER if cat[0] == "L" else CLS_NUMBER if cat[0] == "N" else CLS_OTHER
    ranges, cur = [], None
    for c in range(0x110000):
        k = CLS_OTHER if 0xD800 <= c < 0xE000 else classify(c)
        if k == CLS_OTHER:
            cur = None
            continue
        if cur and cur[2] == k and cur[1] == c - 1:
            cur[1] = c
        else:
            cur = [c, c, k]
            ranges.append(cur)
    return [tuple(r) for r in ranges]


def nfc_tables():
    """(decompositions, combining classes, compositions) for NFC, Hangul excluded."""
    decomp, ccc, comp = [], [], []
    for c in range(0x110000):
        if 0xD800 <= c < 0xE000 or S_BASE <= c < S_BASE + S_COUNT:
            continue
        ch = chr(c)
        k = unicodedata.combining(ch)
        if k:
            ccc.append((c, k))
        d = unicodedata.normalize("NFD", ch)
        if d != ch:
            if len(d) > 4:
                raise ValueError(f"decomposition of U+{c:04X} longer than 4")
            decomp.append((c, [ord(x) for x in d]))
        raw = unicodedata.decomposition(ch)
        if raw and not raw.startswith("<"):
            parts = [int(x, 16) for x in raw.split()]
            if len(parts) == 2 and unicodedata.normalize("NFC", ch) == ch:
                comp.append((parts[0], parts[1], c))
    comp.sort()
    try:
        from tokenizers import normalizers
    except ImportError:
        return decomp, ccc, comp
    return _match_tokenizers_nfc(normalizers.NFC().normalize_str, decomp, ccc, comp)


def _match_tokenizers_nfc(hf_nfc, decomp, ccc, comp):
    """Drop table entries the `tokenizers` NFC does not know (it can use an
    older Unicode version than Python), so the device normalizes exactly like
    the real tokenizer."""
    # a combining class 0 < k < 240 moves c in front of U+0345 (class 240)
    ccc = [(c, k) for c, k in ccc
           if k >= 240 or hf_nfc("x\u0345" + chr(c)) == unicodedata.normalize("NFC", "x\u0345" + chr(c))]
    known = {c for c, _ in ccc}
    comp = [(a, b, c) for a, b, c in comp if hf_nfc(chr(a) + chr(b)) == chr(c)]
    tmp = TextTokenizer({"byte_ids": [0] * 256, "merges": [], "added": [], "nfc": True},
                        classes=[], nfc=([], ccc, comp))
    decomp = [(c, d) for c, d in decomp
              if all(x in known or unicodedata.combining(chr(x)) == 0 for x in d)
              and hf_nfc(chr(c)) == "".join(map(chr, tmp.nfc(d)))]
    return decomp, ccc, comp


# ---------------------------------------------------------------------------
# tokenizer.json -> tables
# ---------------------------------------------------------------------------


def _check(cond, what):
    if not cond:
        raise ValueError(f"unsupported tokenizer: {what}")


def _flatten(node, kind):
    if node is None:
        return []
    if node.get("type") == "Sequence":
        key = "normalizers" if kind == "normalizer" else "pretokenizers"
        return [x for sub in node[key] for x in _flatten(sub, kind)]
    return [node]


def load_tokenizer_json(path):
    """Read a Hugging Face tokenizer.json (path or parsed dict) into the device tables."""
    if isinstance(path, dict):
        j = path
    else:
        with open(path, encoding="utf-8") as f:
            j = json.load(f)
    norms = _flatten(j.get("normalizer"), "normalizer")
    _check(all(n["type"] == "NFC" for n in norms), f"normalizer {norms}")
    pres = _flatten(j.get("pre_tokenizer"), "pre_tokenizer")
    _check(len(pres) == 1 and pres[0]["type"] == "ByteLevel", f"pre_tokenizer {pres}")
    _check(pres[0].get("use_regex", True) and not pres[0].get("add_prefix_space", False),
           "ByteLevel needs use_regex=true, add_prefix_space=false")
    m = j["model"]
    _check(m["type"] == "BPE", f"model {m['type']}")
    _check(not m.get("dropout") and not m.get("byte_fallback") and not m.get("ignore_merges")
           and not m.get("continuing_subword_prefix") and not m.get("end_of_word_suffix"),
           "BPE options (dropout / byte_fallback / ignore_merges / prefixes)")
    vocab = m["vocab"]
    b2u = bytes_to_unicode()
    # Bytes that never occur in UTF-8 (0xC0, 0xC1, 0xF5-0xFF) may be missing from
    # the vocabulary; tokenizers skips a missing symbol when there is no unk token.
    byte_ids = [vocab.get(b2u[b], NO_ID) for b in range(256)]
    missing = [b for b in range(256) if byte_ids[b] == NO_ID]
    _check(all(b in (0xC0, 0xC1) or b >= 0xF5 for b in missing),
           f"bytes {missing} missing from the vocabulary")
    merges = []
    for rank, mg in enumerate(m["merges"]):
        a, b = mg if isinstance(mg, list) else mg.split(" ", 1)
        if a in vocab and b in vocab and a + b in vocab:
            merges.append((vocab[a], vocab[b], rank, vocab[a + b]))
    seen, uniq = set(), []
    for mg in merges:  # the first (lowest-rank) merge of a pair wins, as in HF
        if (mg[0], mg[1]) not in seen:
            seen.add((mg[0], mg[1]))
            uniq.append(mg)
    added = []
    for t in j.get("added_tokens", []):
        _check(not t.get("single_word"), f"single_word added token {t['content']!r}")
        _check(unicodedata.normalize("NFC", t["content"]) == t["content"],
               f"added token {t['content']!r} changes under NFC")
        added.append((t["id"], t["content"].encode("utf-8"), bool(t.get("lstrip")),
                       bool(t.get("rstrip")), bool(t.get("normalized", not t.get("special")))))
    return {"nfc": bool(norms), "byte_ids": byte_ids, "merges": sorted(uniq),
            "added": added}


# ---------------------------------------------------------------------------
# Python mirror of the C tokenizer
# ---------------------------------------------------------------------------


class TextTokenizer:
    """Exactly the algorithm of core/src/trv_laya_text.c, in Python."""

    def __init__(self, tables, classes=None, nfc=None):
        self.t = tables
        self.classes = classes if classes is not None else char_classes()
        self.decomp, self.ccc, self.comp = nfc if nfc is not None else nfc_tables()
        self._cls = {}
        for lo, hi, k in self.classes:
            if hi - lo < 4096:
                for c in range(lo, hi + 1):
                    self._cls[c] = k
        self._ranges = [r for r in self.classes if r[1] - r[0] >= 4096]
        self._dec = dict(self.decomp)
        self._cc = dict(self.ccc)
        self._comp = {(a, b): c for a, b, c in self.comp}
        self._merge = {(a, b): (r, n) for a, b, r, n in tables["merges"]}

    def cls(self, c):
        k = self._cls.get(c)
        if k is not None:
            return k
        for lo, hi, kk in self._ranges:
            if lo <= c <= hi:
                return kk
        return CLS_OTHER

    # -- NFC --
    def nfc(self, cps):
        out = []
        for c in cps:
            if S_BASE <= c < S_BASE + S_COUNT:
                s = c - S_BASE
                out += [L_BASE + s // N_COUNT, V_BASE + (s % N_COUNT) // T_COUNT]
                if s % T_COUNT:
                    out.append(T_BASE + s % T_COUNT)
            else:
                out += self._dec.get(c, [c])
        i = 0  # canonical ordering: stable sort of every run of non-starters
        while i < len(out):
            if self._cc.get(out[i], 0) == 0:
                i += 1
                continue
            j = i
            while j < len(out) and self._cc.get(out[j], 0):
                j += 1
            out[i:j] = sorted(out[i:j], key=lambda c: self._cc[c])
            i = j
        res, starter, last = [], -1, 0
        for c in out:
            cc = self._cc.get(c, 0)
            if starter >= 0:
                blocked = len(res) - 1 != starter and (last == 0 or last >= cc)
                if not blocked:
                    x = self._compose(res[starter], c)
                    if x is not None:
                        res[starter] = x
                        continue
            if cc == 0:
                starter = len(res)
            last = cc
            res.append(c)
        return res

    def _compose(self, a, b):
        if L_BASE <= a < L_BASE + L_COUNT and V_BASE <= b < V_BASE + V_COUNT:
            return S_BASE + ((a - L_BASE) * V_COUNT + (b - V_BASE)) * T_COUNT
        if S_BASE <= a < S_BASE + S_COUNT and (a - S_BASE) % T_COUNT == 0 and \
                T_BASE < b < T_BASE + T_COUNT:
            return a + (b - T_BASE)
        return self._comp.get((a, b))

    # -- pre-tokenization (GPT-2 regex) --
    def pretokenize(self, cps):
        """'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"""
        n, i, out, K = len(cps), 0, [], self.cls
        while i < n:
            c = cps[i]
            nx = cps[i + 1] if i + 1 < n else -1
            if c == 0x27 and nx in (0x73, 0x74, 0x6D, 0x64):          # 's 't 'm 'd
                j = i + 2
            elif c == 0x27 and i + 2 < n and (nx, cps[i + 2]) in ((0x72, 0x65), (0x76, 0x65),
                                                                  (0x6C, 0x6C)):  # 're 've 'll
                j = i + 3
            elif c == 0x20 and nx >= 0 and K(nx) != CLS_SPACE:      # " word", " 123", " !!"
                k, j = K(nx), i + 2
                while j < n and K(cps[j]) == k:
                    j += 1
            elif K(c) != CLS_SPACE:                                   # "word", "123", "!!"
                k, j = K(c), i + 1
                while j < n and K(cps[j]) == k:
                    j += 1
            else:                                                     # whitespace
                j = i
                while j < n and K(cps[j]) == CLS_SPACE:
                    j += 1
                if j < n and j - i >= 2:  # leave the last one to start the next word
                    j -= 1
            out.append(cps[i:j])
            i = j
        return out

    # -- BPE --
    def bpe(self, data):
        ids = [self.t["byte_ids"][b] for b in data if self.t["byte_ids"][b] != NO_ID]
        while len(ids) > 1:
            best, at = None, -1
            for p in range(len(ids) - 1):
                m = self._merge.get((ids[p], ids[p + 1]))
                if m and (best is None or m[0] < best[0]):
                    best, at = m, p
            if best is None:
                break
            ids[at:at + 2] = [best[1]]
        return ids

    def _split(self, text, normalized):
        """[(id or None, start, stop)] exactly like tokenizers' AddedVocabulary.find_matches:
        leftmost-longest non-overlapping matches, then lstrip/rstrip widen a match's span
        (a later match inside the widened span is still emitted, as tokenizers does)."""
        ws = set(map(chr, WHITESPACE))
        toks = [(tid, content.decode("utf-8"), ls, rs) for tid, content, ls, rs, nm in
                self.t["added"] if nm == normalized and content]
        out, i, start_offset = [], 0, 0
        while i < len(text):
            best = None
            for tok in toks:
                if text.startswith(tok[1], i) and (best is None or len(tok[1]) > len(best[1])):
                    best = tok
            if best is None:
                i += 1
                continue
            tid, content, ls, rs = best
            start, stop = i, i + len(content)
            i = stop
            if ls:
                k = start
                while k > 0 and text[k - 1] in ws:
                    k -= 1
                start = max(k, start_offset)
            if rs:
                while stop < len(text) and text[stop] in ws:
                    stop += 1
            if start_offset < start:
                out.append((None, start_offset, start))
            out.append((tid, start, stop))
            start_offset = stop
        if start_offset < len(text):
            out.append((None, start_offset, len(text)))
        return out

    def encode(self, text):
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        ids = []
        for tid, a, b in self._split(text, False):        # special tokens, on the raw text
            if tid is not None:
                ids.append(tid)
                continue
            cps = [ord(c) for c in text[a:b]]
            if self.t["nfc"]:
                cps = self.nfc(cps)
            norm = "".join(map(chr, cps))
            for tid2, c, d in self._split(norm, True):     # normalized added tokens
                if tid2 is not None:
                    ids.append(tid2)
                    continue
                for piece in self.pretokenize(cps[c:d]):
                    ids += self.bpe("".join(map(chr, piece)).encode("utf-8"))
        return ids


# ---------------------------------------------------------------------------
# questions (from the laya package) and the .tlq file
# ---------------------------------------------------------------------------


def agent_tokenizer(agent):
    """The tables of the tokenizer Laya itself loaded (its fast tokenizer's JSON)."""
    backend = getattr(agent.tok, "backend_tokenizer", None)
    if backend is None:
        raise ValueError("Laya's tokenizer is not a fast tokenizer; pass tokenizer.json")
    return load_tokenizer_json(json.loads(backend.to_str()))


def laya_questions(agent, questions):
    """Laya's own question heads: [(qid, qtype, head_ids, markers, room, slot_option, names)]."""
    from laya.agent import Agent
    from laya.common import QTYPES, build_head

    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    out = []
    for qid, qdef in questions.items():
        agent._check_question(qid, qdef)
        q = Agent._to_internal(qdef)
        order = q.get("option_order")
        head, markers, _ = build_head(agent.tok, q, head_max_len, option_order=order)
        if q["t"] == "choice":
            names = [str(k) for k in q["crit"]]
        elif q["t"] == "score":
            names = [str(i) for i in range(len(q["crit"]))]
        else:
            names = ["no", "yes"]
        if len(markers) != len(names) or max(markers) >= max_len - 1:
            raise ValueError(f"question {qid!r} does not fit in max_len={max_len}")
        slot_option = list(order) if order is not None else list(range(len(names)))
        out.append((qid, QTYPES[q["t"]], list(map(int, head)), list(map(int, markers)),
                    max(0, max_len - len(head) - 1), slot_option, names))
    return out, int(agent.tok.sep_token_id), agent.tok.mask_token


def _u32(*v):
    return struct.pack(f"<{len(v)}I", *v)


def _blob(b):
    return _u32(len(b)) + b + b"\0" * (-len(b) % 4)


def write_tlq(path, tables, questions, sep_id, mask_token="[MASK]", classes=None, nfc=None):
    """Write tokenizer + questions. Returns the file size."""
    classes = classes if classes is not None else char_classes()
    decomp, ccc, comp = nfc if nfc is not None else nfc_tables()
    out = [MAGIC, _u32(VERSION, FLAG_NFC if tables["nfc"] else 0, sep_id)]
    out.append(_blob(mask_token.encode("utf-8")))
    out.append(_u32(len(classes)))
    out += [_u32(*r) for r in classes]
    if tables["nfc"]:
        out.append(_u32(len(decomp)))
        out += [_u32(c, len(d), *(d + [0] * (4 - len(d)))) for c, d in decomp]
        out.append(_u32(len(ccc)))
        out += [_u32(c, k) for c, k in ccc]
        out.append(_u32(len(comp)))
        out += [_u32(*x) for x in comp]
    out.append(_u32(*tables["byte_ids"]))
    out.append(_u32(len(tables["merges"])))
    out += [_u32(*m) for m in tables["merges"]]
    out.append(_u32(len(tables["added"])))
    for tid, content, ls, rs, nm in tables["added"]:
        out.append(_u32(tid, ls | (rs << 1) | (nm << 2)) + _blob(content))
    out.append(_u32(len(questions)))
    for qid, qtype, head, markers, room, slot_option, names in questions:
        out.append(_u32(qtype, len(markers), len(head), room))
        out.append(_blob(qid.encode("utf-8")))
        out.append(_u32(*markers) + _u32(*slot_option) + _u32(*head))
        for nm in names:
            out.append(_blob(nm.encode("utf-8")))
    data = b"".join(out)
    with open(path, "wb") as f:
        f.write(data)
    return len(data)


def rows_for_text(tokenizer, questions, sep_id, text, mask_token="[MASK]"):
    """Python mirror of trv_laya_text_rows: Laya rows for a string input."""
    ids = tokenizer.encode(text.replace(mask_token, " "))
    rows = []
    for qid, qtype, head, markers, room, slot_option, names in questions:
        rows.append({"qtype": qtype, "ids": head + ids[:room] + [sep_id], "markers": markers})
    return rows
