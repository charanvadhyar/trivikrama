"""Laya (Convai Innovations' decision model) on Trivikrama.

Laya = a ModernBERT encoder + a small decision head:

    h  = encoder(tokens)                                   # [T x 1024]
    h += type_emb[question type]                           # choice / score / yes-no
    h  = 2 x pre-norm transformer layers (16 heads, ReLU FFN 4096)
    logit[option] = scorer(h[marker of option])            # LayerNorm, Linear, GELU, Linear->1
    p = softmax(logits / temperature[type, #options])

The C runtime (core/src/trv_laya.c) runs all of it on the device. What stays on
the PC (or phone) for now is turning text + questions into token ids, exactly
the way the `laya` package does (`prepare`), and turning the probabilities
back into Laya-style answers (`decode`).

Copyright 2026 Vadhyar Sai Charan
SPDX-License-Identifier: Apache-2.0
Laya itself is (c) Convai Innovations, Apache License 2.0.
"""

import glob
import json
import math
import os
import struct

import numpy as np

from .format import DTYPE_F32, DTYPE_Q8_ROWS

QTYPES = {"choice": 0, "score": 1, "noul": 2}
BUCKETS = ("2", "3-5", "6-10", "11+")
TEMP_MIN, TEMP_MAX = 0.5, 5.0
REQUEST_MAGIC = b"TRLQ"


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def clamp_temperature(t):
    """Same rule as laya.common.clamp_temperature: a number in [0.5, 5], else 1.0."""
    if isinstance(t, bool):
        return 1.0
    try:
        t = float(t)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(t):
        return 1.0
    return min(max(t, TEMP_MIN), TEMP_MAX)


def bucket_index(k):
    return 0 if k <= 2 else 1 if k <= 5 else 2 if k <= 10 else 3


def temperature_table(rl_cfg):
    """[3 base temperatures] + [3 types x 4 option-count buckets], all clamped.

    Laya uses temperature_by_options["<type>:<bucket>"] when present, else the
    base temperature of the type; the table is pre-filled with that fallback.
    """
    base = rl_cfg.get("temperature", [1.0, 1.0, 1.0])
    base = [clamp_temperature(t) for t in base]
    by = rl_cfg.get("temperature_by_options", {}) or {}
    table = list(base)
    for qt, name in enumerate(("choice", "score", "noul")):
        for b in BUCKETS:
            key = f"{name}:{b}"
            table.append(clamp_temperature(by[key]) if key in by else base[qt])
    return np.array(table, dtype=np.float32)


def head_tensors(sd, rl_cfg, hidden):
    """Laya's decision head as .trv tensors. sd: name -> float32 array."""
    n_layers = int(rl_cfg.get("head_layers", 0)) or len(
        {k.split(".")[2] for k in sd if k.startswith("head.layers.")})
    n_heads = max(1, hidden // 64)
    ff = sd["head.layers.0.linear1.weight"].shape[0] if n_layers else 4 * hidden
    out = [("laya.config", np.array([n_layers, n_heads, ff, 1e-5], np.float32), DTYPE_F32),
           ("laya.temps", temperature_table(rl_cfg), DTYPE_F32),
           ("laya.type_emb", sd["type_emb.weight"], DTYPE_F32)]
    for i in range(n_layers):
        p = f"head.layers.{i}."
        q = f"laya.h{i}."
        out += [
            (q + "qkv", sd[p + "self_attn.in_proj_weight"], DTYPE_Q8_ROWS),
            (q + "qkv_b", sd[p + "self_attn.in_proj_bias"], DTYPE_F32),
            (q + "out", sd[p + "self_attn.out_proj.weight"], DTYPE_Q8_ROWS),
            (q + "out_b", sd[p + "self_attn.out_proj.bias"], DTYPE_F32),
            (q + "ff1", sd[p + "linear1.weight"], DTYPE_Q8_ROWS),
            (q + "ff1_b", sd[p + "linear1.bias"], DTYPE_F32),
            (q + "ff2", sd[p + "linear2.weight"], DTYPE_Q8_ROWS),
            (q + "ff2_b", sd[p + "linear2.bias"], DTYPE_F32),
            (q + "n1.w", sd[p + "norm1.weight"], DTYPE_F32),
            (q + "n1.b", sd[p + "norm1.bias"], DTYPE_F32),
            (q + "n2.w", sd[p + "norm2.weight"], DTYPE_F32),
            (q + "n2.b", sd[p + "norm2.bias"], DTYPE_F32),
        ]
    out += [
        ("laya.score.ln.w", sd["scorer.0.weight"], DTYPE_F32),
        ("laya.score.ln.b", sd["scorer.0.bias"], DTYPE_F32),
        ("laya.score.l1", sd["scorer.1.weight"], DTYPE_Q8_ROWS),
        ("laya.score.l1_b", sd["scorer.1.bias"], DTYPE_F32),
        ("laya.score.l2", sd["scorer.3.weight"].reshape(-1), DTYPE_F32),
        ("laya.score.l2_b", sd["scorer.3.bias"].reshape(-1), DTYPE_F32),
    ]
    return [(n, np.asarray(a, dtype=np.float32), d) for n, a, d in out]


def load_checkpoint(laya_dir):
    """(state dict as float32 numpy, encoder config, rl_agent_config) of a Laya folder."""
    from .cli import _load_safetensors  # handles bf16 via torch when needed

    files = sorted(glob.glob(os.path.join(laya_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors in {laya_dir}")
    sd = {}
    for f in files:
        sd.update(_load_safetensors(f))
    sd = {k: np.asarray(v, dtype=np.float32) for k, v in sd.items()}
    with open(os.path.join(laya_dir, "encoder", "config.json")) as f:
        enc_cfg = json.load(f)
    rl_path = os.path.join(laya_dir, "rl_agent_config.json")
    rl_cfg = json.load(open(rl_path)) if os.path.exists(rl_path) else {}
    return sd, enc_cfg, rl_cfg


def export_laya(laya_dir, out_path, weights="int8", calib_ids=None, max_drop=0.005,
                target_cosine=None, log=print, checkpoint=None):
    """Encoder + decision head + temperatures in one .trv. Returns file size."""
    from .modernbert import export_modernbert

    sd, enc_cfg, rl_cfg = checkpoint or load_checkpoint(laya_dir)
    extra = head_tensors(sd, rl_cfg, enc_cfg["hidden_size"])
    size, ignored = export_modernbert(sd, enc_cfg, out_path, weights=weights, calib_ids=calib_ids,
                                      max_drop=max_drop, target_cosine=target_cosine, log=log,
                                      extra_tensors=extra)
    skipped = [n for n in ignored if not n.startswith(("head.", "scorer.", "type_emb",
                                                       "temperature"))]
    return size, skipped


# ---------------------------------------------------------------------------
# requests: text + questions -> token rows (on the PC, with Laya's own code)
# ---------------------------------------------------------------------------


class Preparer:
    """Builds request rows with the `laya` package itself, so they are exactly Laya's."""

    def __init__(self, laya_dir=None, agent=None):
        import laya  # noqa: F401  (needs: pip install laya)
        from laya.agent import Agent

        self.agent = agent if agent is not None else laya.load(laya_dir, device="cpu")
        self.Agent = Agent
        if getattr(self.agent, "parallel_options", False):
            raise ValueError("this Laya checkpoint uses the parallel option layout, "
                             "which Trivikrama does not run yet")

    def rows(self, state, questions):
        qids = list(questions)
        for q in qids:  # the same validation laya.predict does
            self.agent._check_question(q, questions[q])
        internal = {q: self.Agent._to_internal(questions[q]) for q in qids}
        items = self.agent._encode_state(state, qids, internal)
        meta = []
        for qid, item in zip(qids, items):
            q = internal[qid]
            if q["t"] == "choice":
                options = [str(k) for k in q["crit"]]
            elif q["t"] == "score":
                options = [str(i) for i in range(len(q["crit"]))]
            else:
                options = ["false", "true"]
            meta.append({"qid": qid, "type": q["t"], "options": options,
                         "option_order": q.get("option_order")})
        rows = [{"qtype": int(it["qtype"]), "ids": [int(i) for i in it["ids"]],
                 "markers": [int(m) for m in it["markers"]]} for it in items]
        return rows, meta


def write_request(path, rows):
    """Binary request for the device: "TRLQ" u32 version u32 n_rows, then per row
    u8 qtype, u8 n_options, u16 n_tokens, u32 markers[n_options], u32 ids[n_tokens]
    (little-endian; everything stays 4-byte aligned, so a device can use it in place)."""
    with open(path, "wb") as f:
        f.write(REQUEST_MAGIC + struct.pack("<II", 1, len(rows)))
        for r in rows:
            k, T = len(r["markers"]), len(r["ids"])
            if k > 255 or T > 65535:
                raise ValueError("request row too large")
            f.write(struct.pack("<BBH", r["qtype"], k, T))
            f.write(struct.pack(f"<{k}I", *r["markers"]))
            f.write(struct.pack(f"<{T}I", *r["ids"]))


def decode(probs_rows, meta):
    """Probabilities per row -> answers shaped like laya's predict()."""
    answers = {}
    for p, m in zip(probs_rows, meta):
        p = np.asarray(p, dtype=np.float64)
        k = len(p)
        order = m.get("option_order")
        if order is not None and len(order) == k:  # slot order -> option order (unpermute_probs)
            q = np.empty_like(p)
            q[np.asarray(order, dtype=int)] = p
            p = q
        top = round(float(p.max()), 4)  # laya's answer_confidence
        if k >= 2:  # laya's confidence_from_probs: 1 - entropy / log(k)
            nz = p[p > 0]
            entropy_conf = round(float(1.0 + (nz * np.log(nz)).sum() / math.log(k)), 4)
        else:
            entropy_conf = 1.0
        if m["type"] == "noul":
            answers[m["qid"]] = {"type": "noul", "noul": round(float(p[1]), 4),
                                 "confidence": top, "answer_confidence": top}
        elif m["type"] == "choice":
            answers[m["qid"]] = {"type": "choice", "choice": m["options"][int(p.argmax())],
                                 "probabilities": {o: round(float(v), 4)
                                                   for o, v in zip(m["options"], p)},
                                 "confidence": entropy_conf, "answer_confidence": top}
        else:
            answers[m["qid"]] = {"type": "score",
                                 "score": round(float((np.arange(k) * p).sum()), 4),
                                 "probabilities": {str(i): round(float(v), 4)
                                                   for i, v in enumerate(p)},
                                 "confidence": entropy_conf, "answer_confidence": top}
    return answers


def run_device(binary, model, rows, threads=1, tile=262144, extra_args=()):
    """Run build/trv_laya on rows; returns (list of probability rows, stats)."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        req = os.path.join(d, "request.trq")
        write_request(req, rows)
        res = subprocess.run([binary, model, req, "--threads", str(threads), "--tile", str(tile),
                              *extra_args], capture_output=True, text=True)
    if res.returncode:
        raise RuntimeError(f"trv_laya failed: {res.stdout} {res.stderr}")
    out = json.loads(res.stdout)
    return [r["probs"] for r in out["rows"]], out


# ---------------------------------------------------------------------------
# NumPy reference of the head (for tests)
# ---------------------------------------------------------------------------


def reference_head(h, qtype, markers, w, temps, eps=1e-5):
    """Float64 Laya head on one row's encoder output h [T x H]. Returns (logits, probs).

    w holds the head tensors under their .trv names (de-quantized is fine).
    """
    from . import reference as ref

    H = h.shape[1]
    n_layers, n_heads, ff = (int(v) for v in w["laya.config"][:3])
    hd = H // n_heads
    x = h.astype(np.float64) + w["laya.type_emb"][qtype]

    def ln(v, g, b):
        mu = v.mean(-1, keepdims=True)
        var = ((v - mu) ** 2).mean(-1, keepdims=True)
        return (v - mu) / np.sqrt(var + eps) * g + b

    for i in range(n_layers):
        p = f"laya.h{i}."
        y = ln(x, w[p + "n1.w"], w[p + "n1.b"])
        qkv = y @ w[p + "qkv"].T + w[p + "qkv_b"]
        o = np.zeros_like(x)
        for hh in range(n_heads):
            sl = slice(hh * hd, (hh + 1) * hd)
            o[:, sl] = ref.attention_head(qkv[:, :H][:, sl], qkv[:, H:2 * H][:, sl],
                                          qkv[:, 2 * H:][:, sl])
        x = x + o @ w[p + "out"].T + w[p + "out_b"]
        y = ln(x, w[p + "n2.w"], w[p + "n2.b"])
        f = np.maximum(y @ w[p + "ff1"].T + w[p + "ff1_b"], 0.0)
        x = x + f @ w[p + "ff2"].T + w[p + "ff2_b"]
    m = ln(x[markers], w["laya.score.ln.w"], w["laya.score.ln.b"])
    s = ref.gelu(m @ w["laya.score.l1"].T + w["laya.score.l1_b"])
    logits = s @ w["laya.score.l2"] + w["laya.score.l2_b"][0]
    t = temps[3 + qtype * 4 + bucket_index(len(markers))]
    z = logits / t
    pr = np.exp(z - z.max())
    return logits, pr / pr.sum()
