"""ModernBERT support: export a Hugging Face checkpoint to .trv, and a NumPy
forward pass used to check the C implementation.

The exporter renames tensors to a fixed scheme the C loader understands:

    emb.tok  emb.norm.{w,b}  final_norm.{w,b}
    L{i}.wqkv  L{i}.wo  L{i}.wi  L{i}.wo_mlp  L{i}.attn_norm.{w,b}  L{i}.mlp_norm.{w,b}
    trv.modernbert.config   [n_layers, hidden, heads, intermediate, vocab, eps, group]
    trv.modernbert.layers   [L x 2]: (half_window or -1, rope_theta) per layer

Copyright 2026 Vadhyar Sai Charan
SPDX-License-Identifier: Apache-2.0
"""

import math
import re

import numpy as np

from . import reference as ref
from .format import DTYPE_F32, DTYPE_Q4_G32, DTYPE_Q8_ROWS, read_trv, write_trv
from .quantize import (dequantize_q4_g32, dequantize_rows_q8, quantize_q4_g32,
                       quantize_rows_q8)

_RENAME = [
    (r"embeddings\.tok_embeddings\.weight", "emb.tok"),
    (r"embeddings\.norm\.weight", "emb.norm.w"),
    (r"embeddings\.norm\.bias", "emb.norm.b"),
    (r"final_norm\.weight", "final_norm.w"),
    (r"final_norm\.bias", "final_norm.b"),
    (r"layers\.(\d+)\.attn\.Wqkv\.weight", r"L\1.wqkv"),
    (r"layers\.(\d+)\.attn\.Wo\.weight", r"L\1.wo"),
    (r"layers\.(\d+)\.mlp\.Wi\.weight", r"L\1.wi"),
    (r"layers\.(\d+)\.mlp\.Wo\.weight", r"L\1.wo_mlp"),
    (r"layers\.(\d+)\.attn_norm\.weight", r"L\1.attn_norm.w"),
    (r"layers\.(\d+)\.attn_norm\.bias", r"L\1.attn_norm.b"),
    (r"layers\.(\d+)\.mlp_norm\.weight", r"L\1.mlp_norm.w"),
    (r"layers\.(\d+)\.mlp_norm\.bias", r"L\1.mlp_norm.b"),
]
_UNSUPPORTED = re.compile(r"layers\.\d+\.(attn\.(Wqkv|Wo)|mlp\.(Wi|Wo))\.bias")


def rename_state_dict(state_dict):
    """Map Hugging Face ModernBERT names to Trivikrama names.

    The encoder may sit under any prefix ("model.", "encoder.", "backbone."...);
    it is detected from the token-embedding tensor. Tensors outside the encoder
    (task heads) are returned in ignored_names. Returns (renamed, ignored_names).
    """
    anchor = "embeddings.tok_embeddings.weight"
    hits = [n for n in state_dict if n.endswith(anchor)]
    if len(hits) != 1:
        raise ValueError(f"expected exactly one '*{anchor}' tensor, found {len(hits)}")
    prefix = hits[0][: -len(anchor)]
    out, ignored = {}, []
    for name, arr in state_dict.items():
        if not name.startswith(prefix):
            ignored.append(name)
            continue
        base = name[len(prefix):]
        if _UNSUPPORTED.fullmatch(base):
            raise ValueError(f"linear biases are not supported yet: {name}")
        for pat, rep in _RENAME:
            if re.fullmatch(pat, base):
                out[re.sub(pat, rep, base)] = np.asarray(arr, dtype=np.float32)
                break
        else:
            ignored.append(name)
    return out, ignored


def _get(cfg, *keys, default=None):
    for k in keys:
        if k in cfg and cfg[k] is not None:
            return cfg[k]
    return default


def layer_settings(cfg):
    """[(half_window, rope_theta)] per layer from a HF config dict.

    Handles both the original ModernBERT config keys
    (global_attn_every_n_layers, global_rope_theta, local_rope_theta)
    and newer ones (layer_types, rope_parameters).
    """
    L = cfg["num_hidden_layers"]
    local = int(_get(cfg, "local_attention", default=128))
    every = int(_get(cfg, "global_attn_every_n_layers", default=3))
    types = cfg.get("layer_types")
    rp = cfg.get("rope_parameters") or {}

    def theta(kind):
        sub = rp.get(kind) if isinstance(rp.get(kind), dict) else None
        if sub and "rope_theta" in sub:
            return float(sub["rope_theta"])
        if kind == "full_attention":
            return float(_get(cfg, "global_rope_theta", "rope_theta", default=160000.0))
        return float(_get(cfg, "local_rope_theta", default=10000.0))

    out = []
    for i in range(L):
        if types:
            is_global = types[i] in ("full_attention", "global")
        else:
            is_global = i % every == 0
        kind = "full_attention" if is_global else "sliding_attention"
        out.append((-1 if is_global else local // 2, theta(kind)))
    return out


def pick_group(cfg):
    hd = cfg["hidden_size"] // cfg["num_attention_heads"]
    return math.gcd(hd, cfg["intermediate_size"])


def int4_error(w):
    """Relative reconstruction error (Frobenius) of int4 group-32 quantization."""
    q, s16 = quantize_q4_g32(w)
    d = dequantize_q4_g32(q, s16)
    return float(np.linalg.norm(d - w) / max(np.linalg.norm(w), 1e-30))


_MATRIX = re.compile(r"L\d+\.(wqkv|wo|wi|wo_mlp)$")


class _Quantized:
    """Weights view that de-quantizes matrices on access (keeps memory low)."""

    def __init__(self, base, q8, q4, use4):
        self.base, self.q8, self.q4, self.use4 = base, q8, q4, use4

    def __contains__(self, name):
        return name in self.base

    def get(self, name, default=None):
        return self[name] if name in self.base else default

    def __getitem__(self, name):
        if name in self.q8:
            if name in self.use4:
                return dequantize_q4_g32(*self.q4[name])
            return dequantize_rows_q8(*self.q8[name])
        return self.base[name]


def _token_cosines(outs, refs):
    return [((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-30))
            for a, b in zip(outs, refs)]


def _mean_cosine(outs, refs):
    c = np.concatenate(_token_cosines(outs, refs))
    return float(c.mean()), float(c.min())


def choose_precision(w, cfg, calib_ids, target_cosine=None, max_drop=0.005, log=print):
    """Pick which layer matrices can be int4 while the encoder output stays close.

    1. Run the float model on the calibration sequences and record the input
       of every layer matrix.
    2. Rank matrices by how much int4 changes *their own output* on those real
       inputs (activation-aware, not just weight error).
    3. Find the largest number k of least-sensitive matrices that can be int4
       (the rest int8) while the mean token cosine to the float output stays
       at least `target_cosine` if given, otherwise at most `max_drop` below
       the all-int8 model (the default: a relative budget, so a few odd tokens
       that are hard even for int8 do not block every int4 choice).

    Returns (set of int4 names, info dict).
    """
    names = sorted((n for n in w if _MATRIX.match(n)), key=_order_key)
    capture = {}
    refs = [float_forward(w, cfg, ids, capture=capture) for ids in calib_ids]
    q8 = {n: quantize_rows_q8(w[n]) for n in names}
    q4 = {n: quantize_q4_g32(w[n]) for n in names}

    sens = {}
    for n in names:
        X = np.concatenate(capture[n]).astype(np.float32)
        y = X @ w[n].T
        e = X @ (dequantize_q4_g32(*q4[n]) - w[n]).T
        sens[n] = float(np.linalg.norm(e) / max(np.linalg.norm(y), 1e-30))
    order = sorted(names, key=sens.get)

    cache = {}

    def quality(k):
        if k not in cache:
            view = _Quantized(w, q8, q4, set(order[:k]))
            outs = [float_forward(view, cfg, ids) for ids in calib_ids]
            cache[k] = _mean_cosine(outs, refs)
            log(f"  int4 for {k:3d}/{len(names)} matrices: mean cosine {cache[k][0]:.5f} "
                f"(min {cache[k][1]:.4f})")
        return cache[k][0]

    base = quality(0)
    # show the tokens that are hard even for int8 (outliers drag the mean down)
    view8 = _Quantized(w, q8, q4, set())
    per_tok = _token_cosines([float_forward(view8, cfg, ids) for ids in calib_ids], refs)
    worst = sorted(((float(c[t]), si, t) for si, c in enumerate(per_tok) for t in range(len(c))))[:3]
    log("  hardest tokens for int8 (cosine, sequence, position, token id): " + ", ".join(
        f"{c:.3f} s{si} p{t} id{calib_ids[si][t]}" for c, si, t in worst))
    if target_cosine is None:
        target_cosine = base - max_drop
        log(f"  budget: int4 may lower the mean cosine by {max_drop} -> threshold {target_cosine:.5f}")
    if base < target_cosine:
        log(f"  even all-int8 ({base:.5f}) is below the target {target_cosine}; keeping int8")
        k = 0
    elif quality(len(names)) >= target_cosine:
        k = len(names)
    else:  # quality falls as k grows: binary search the boundary
        lo, hi = 0, len(names)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if quality(mid) >= target_cosine:
                lo = mid
            else:
                hi = mid
        k = lo
    chosen = set(order[:k])
    info = dict(n_int4=k, n_total=len(names), cosine=cache[k][0], min_cosine=cache[k][1],
                int8_cosine=cache[0][0], threshold=target_cosine, sensitivity=sens)
    return chosen, info


def _order_key(name):
    """config first, then layers in numeric order, then final norm, embeddings last.

    Layer data ends up in forward order in the file, so a prefix of the file
    (e.g. copied to internal flash as a fast tier) holds the earliest layers,
    and the big embedding table, read only a row per token, sits at the end.
    """
    if name.startswith("trv."):
        return (0, 0, name)
    m = re.match(r"L(\d+)\.", name)
    if m:
        return (1, int(m.group(1)), name)
    if name.startswith("final_norm"):
        return (2, 0, name)
    return (3, 0, name)


def export_modernbert(state_dict, cfg, path, group=None, weights="int8", calib_ids=None,
                      target_cosine=None, max_drop=0.005, report=None, log=print):
    """Write a ModernBERT encoder to a .trv file. Returns (file_bytes, ignored).

    weights:
      "int8"  every layer matrix int8 (the safe default)
      "int4"  every layer matrix int4
      "auto"  calibrated: int4 only where the encoder output on calib_ids stays
              within max_drop (mean token cosine) of the all-int8 model, or
              above target_cosine if given; the rest int8 (see choose_precision)
    calib_ids: list of token-id lists for "auto"; without it, random ids are
    used, which works mechanically but says little about real text.
    The embedding table always stays int8: only a few rows are read per
    inference, so int4 would save storage but not time.
    report, if a list, receives (name, "int4"/"int8", sensitivity) for "auto".
    """
    if weights not in ("int8", "int4", "auto"):
        raise ValueError("weights must be 'int8', 'int4' or 'auto'")
    w, ignored = rename_state_dict(state_dict)
    H = cfg["hidden_size"]
    group = group or pick_group(cfg)
    hd = H // cfg["num_attention_heads"]
    if hd % group or cfg["intermediate_size"] % group:
        raise ValueError(f"group {group} must divide head_dim {hd} and intermediate size")
    if weights != "int8" and group % 32 and 32 % group:
        raise ValueError(f"int4 needs the activation group ({group}) and 32 to divide one another")

    use4 = set()
    if weights == "int4":
        use4 = {n for n in w if _MATRIX.match(n) and w[n].shape[1] % 32 == 0}
    elif weights == "auto":
        if not calib_ids:
            log("warning: no calibration text; using random token ids")
            r = np.random.default_rng(0)
            calib_ids = [r.integers(5, cfg["vocab_size"], 16).tolist() for _ in range(4)]
        use4, info = choose_precision(w, cfg, calib_ids, target_cosine, max_drop, log=log)
        log(f"auto precision: {info['n_int4']}/{info['n_total']} matrices int4, "
            f"mean cosine {info['cosine']:.5f} (all-int8: {info['int8_cosine']:.5f}, "
            f"threshold {info['threshold']:.5f})")
        if report is not None:
            for n, e in sorted(info["sensitivity"].items(), key=lambda kv: _order_key(kv[0])):
                report.append((n, "int4" if n in use4 else "int8", e))

    config = np.array(
        [
            cfg["num_hidden_layers"],
            H,
            cfg["num_attention_heads"],
            cfg["intermediate_size"],
            cfg["vocab_size"],
            float(_get(cfg, "norm_eps", "layer_norm_eps", default=1e-5)),
            group,
        ],
        dtype=np.float32,
    )
    layers = np.array(layer_settings(cfg), dtype=np.float32)
    tensors = [
        ("trv.modernbert.config", config, DTYPE_F32),
        ("trv.modernbert.layers", layers, DTYPE_F32),
    ]
    for name in sorted(w, key=_order_key):
        arr = w[name]
        if arr.ndim != 2:
            dtype = DTYPE_F32
        elif name in use4:
            dtype = DTYPE_Q4_G32
        else:
            dtype = DTYPE_Q8_ROWS
        tensors.append((name, arr, dtype))
    return write_trv(path, tensors), ignored


# ---------------------------------------------------------------------------
# NumPy forward pass
# ---------------------------------------------------------------------------

F32 = np.float32


def _seqsum(x):
    """Left-to-right float32 sum over the last axis, the way the C loops add."""
    return np.cumsum(x, axis=-1, dtype=F32)[..., -1]


def _q8g(x, G):
    T, K = x.shape
    xr = x.reshape(T, K // G, G)
    m = np.abs(xr).max(-1)
    s = np.where(m > 0, m / F32(127.0), F32(1.0)).astype(F32)
    q = np.clip(np.rint(xr / s[..., None]), -127, 127).astype(np.int8)
    return q.reshape(T, K), s


def _ln(x, g, b, eps):
    D = x.shape[1]
    mean = _seqsum(x) / F32(D)
    d = x - mean[:, None]
    var = _seqsum(d * d) / F32(D)
    inv = F32(1.0) / np.sqrt(var + F32(eps))
    v = (d * inv[:, None]) * g
    return (v + b if b is not None else v).astype(F32)


def _lin_q4(xq, xs, G, wq, ws16, rows):
    """Mirror of trv_linear_stream with Q4_G32 weights."""
    T, K = xq.shape
    c = min(G, 32)
    nc = K // c
    wq = wq[rows]
    gs = ws16[rows].astype(F32)
    acc = np.einsum(
        "tjk,njk->tnj",
        xq.reshape(T, nc, c).astype(np.int64),
        wq.reshape(len(wq), nc, c).astype(np.int64),
    )
    s = np.zeros(acc.shape[:2], dtype=F32)
    for j in range(nc):
        k0 = j * c
        s = s + (acc[:, :, j].astype(F32) * xs[:, None, k0 // G]) * gs[None, :, k0 // 32]
    return s


def _lin_q8g(xq, xs, G, wq, ws, rows):
    """Mirror of trv_linear_stream for output rows `rows` (int8 or int4 weights)."""
    if ws.dtype == np.float16:
        return _lin_q4(xq, xs, G, wq, ws, rows)
    T, K = xq.shape
    ng = K // G
    wq = wq[rows]
    acc = np.einsum(
        "tgk,ngk->tng",
        xq.reshape(T, ng, G).astype(np.int64),
        wq.reshape(len(wq), ng, G).astype(np.int64),
    )
    s = np.zeros(acc.shape[:2], dtype=F32)
    for g in range(ng):
        s = s + acc[:, :, g].astype(F32) * xs[:, None, g]
    return (s * ws[rows][None, :]).astype(F32)


def float_forward(weights, cfg, ids, capture=None):
    """Float32 ModernBERT forward (fast, BLAS matmuls). Returns final hidden states.

    capture, if a dict, collects the input of every layer matrix under its
    name ("L3.wi" ...), for activation-aware calibration.
    """
    L, H = cfg["num_hidden_layers"], cfg["hidden_size"]
    nh, I = cfg["num_attention_heads"], cfg["intermediate_size"]
    hd = H // nh
    eps = float(_get(cfg, "norm_eps", "layer_norm_eps", default=1e-5))
    settings = cfg.get("layers") or layer_settings(cfg)
    gb = weights.get
    keep = (lambda n, x: capture.setdefault(n, []).append(x)) if capture is not None else (
        lambda n, x: None)

    def ln(x, g, b):
        mu = x.mean(-1, keepdims=True)
        v = ((x - mu) ** 2).mean(-1, keepdims=True)
        y = (x - mu) / np.sqrt(v + F32(eps)) * g
        return (y + b if b is not None else y).astype(F32)

    h = ln(np.asarray(weights["emb.tok"], dtype=F32)[ids], weights["emb.norm.w"], gb("emb.norm.b"))
    for i in range(L):
        half_window, theta = settings[i]
        p = f"L{i}."
        x = h if p + "attn_norm.w" not in weights else ln(h, weights[p + "attn_norm.w"],
                                                          gb(p + "attn_norm.b"))
        keep(p + "wqkv", x)
        qkv = x @ weights[p + "wqkv"].T
        out = np.zeros((len(ids), H), dtype=F32)
        for hh in range(nh):
            sl = slice(hh * hd, (hh + 1) * hd)
            qh = ref.rope(qkv[:, :H][:, sl], 1, hd, base=theta)
            kh = ref.rope(qkv[:, H:2 * H][:, sl], 1, hd, base=theta)
            out[:, sl] = ref.attention_head(qh, kh, qkv[:, 2 * H:][:, sl], half_window=half_window)
        keep(p + "wo", out)
        h = h + out @ weights[p + "wo"].T
        x = ln(h, weights[p + "mlp_norm.w"], gb(p + "mlp_norm.b"))
        keep(p + "wi", x)
        ag = x @ weights[p + "wi"].T
        act = (ref.gelu(ag[:, :I]) * ag[:, I:]).astype(F32)
        keep(p + "wo_mlp", act)
        h = h + act @ weights[p + "wo_mlp"].T
    return ln(h, weights["final_norm.w"], gb("final_norm.b"))


def load_trv_modernbert(path):
    """Read an exported .trv into (weights, cfg) for forward(quantized=True)."""
    raw = read_trv(path, dequantize=False)
    c = raw.pop("trv.modernbert.config")[1]
    lay = raw.pop("trv.modernbert.layers")[1]
    cfg = dict(
        num_hidden_layers=int(c[0]),
        hidden_size=int(c[1]),
        num_attention_heads=int(c[2]),
        intermediate_size=int(c[3]),
        vocab_size=int(c[4]),
        norm_eps=float(c[5]),
        group=int(c[6]),
        layers=[(int(a), float(b)) for a, b in lay],
    )
    return {k: v for k, (_, v) in raw.items()}, cfg


def forward(weights, cfg, ids, quantized, hook=None):
    """ModernBERT encoder in NumPy.

    quantized=False: plain float64 math on float weights (renamed names); this
        checks the architecture against Hugging Face.
    quantized=True: mirrors the C pipeline step by step (int8 weights as
        (q, scale) pairs, group-wise int8 activations, float32 everywhere);
        this checks the C code.
    hook(layer, h) is called like the C hook: -1, 0..L-1, L.
    """
    L, H = cfg["num_hidden_layers"], cfg["hidden_size"]
    nh, I = cfg["num_attention_heads"], cfg["intermediate_size"]
    hd = H // nh
    eps = float(_get(cfg, "norm_eps", "layer_norm_eps", default=1e-5))
    settings = cfg.get("layers") or layer_settings(cfg)
    G = cfg.get("group") or pick_group(cfg)
    T = len(ids)
    hook = hook or (lambda *_: None)
    gb = lambda n: weights.get(n)

    if quantized:
        q, s = weights["emb.tok"]
        h = q[ids].astype(F32) * s[ids][:, None]
        h = _ln(h, weights["emb.norm.w"], gb("emb.norm.b"), eps)
    else:
        h = weights["emb.tok"][ids].astype(np.float64)
        h = ref.layernorm(h, weights["emb.norm.w"], gb("emb.norm.b"), eps)
    hook(-1, h)

    for i in range(L):
        half_window, theta = settings[i]
        p = f"L{i}."
        if quantized:
            if p + "attn_norm.w" in weights:
                xq, xs = _q8g(_ln(h, weights[p + "attn_norm.w"], gb(p + "attn_norm.b"), eps), G)
            else:
                xq, xs = _q8g(h, G)
            wq, ws = weights[p + "wqkv"]
            oq = np.zeros((T, H), dtype=np.int8)
            os_ = np.zeros((T, H // G), dtype=F32)
            for hh in range(nh):
                r = hh * hd
                qh = _lin_q8g(xq, xs, G, wq, ws, slice(r, r + hd))
                kh = _lin_q8g(xq, xs, G, wq, ws, slice(H + r, H + r + hd))
                vh = _lin_q8g(xq, xs, G, wq, ws, slice(2 * H + r, 2 * H + r + hd))
                qh = ref.rope(qh, 1, hd, base=theta).astype(F32)
                kh = ref.rope(kh, 1, hd, base=theta).astype(F32)
                o = ref.attention_head(qh, kh, vh, half_window=half_window).astype(F32)
                oq[:, r : r + hd], os_[:, r // G : (r + hd) // G] = _q8g(o, G)
            h = h + _lin_q8g(oq, os_, G, *weights[p + "wo"], slice(None))

            xq, xs = _q8g(_ln(h, weights[p + "mlp_norm.w"], gb(p + "mlp_norm.b"), eps), G)
            wq, ws = weights[p + "wi"]
            fq = np.zeros((T, I), dtype=np.int8)
            fs = np.zeros((T, I // G), dtype=F32)
            for c0 in range(0, I, G):
                a = _lin_q8g(xq, xs, G, wq, ws, slice(c0, c0 + G))
                gate = _lin_q8g(xq, xs, G, wq, ws, slice(I + c0, I + c0 + G))
                act = (ref.gelu(a) * gate).astype(F32)
                fq[:, c0 : c0 + G], fs[:, c0 // G : (c0 + G) // G] = _q8g(act, G)
            h = h + _lin_q8g(fq, fs, G, *weights[p + "wo_mlp"], slice(None))
        else:
            x = h
            if p + "attn_norm.w" in weights:
                x = ref.layernorm(h, weights[p + "attn_norm.w"], gb(p + "attn_norm.b"), eps)
            qkv = x @ weights[p + "wqkv"].T.astype(np.float64)
            out = np.zeros((T, H))
            for hh in range(nh):
                sl = slice(hh * hd, (hh + 1) * hd)
                qh = ref.rope(qkv[:, :H][:, sl], 1, hd, base=theta)
                kh = ref.rope(qkv[:, H : 2 * H][:, sl], 1, hd, base=theta)
                vh = qkv[:, 2 * H :][:, sl]
                out[:, sl] = ref.attention_head(qh, kh, vh, half_window=half_window)
            h = h + out @ weights[p + "wo"].T.astype(np.float64)
            x = ref.layernorm(h, weights[p + "mlp_norm.w"], gb(p + "mlp_norm.b"), eps)
            ag = x @ weights[p + "wi"].T.astype(np.float64)
            h = h + (ref.gelu(ag[:, :I]) * ag[:, I:]) @ weights[p + "wo_mlp"].T.astype(np.float64)
        hook(i, h)

    if quantized:
        h = _ln(h, weights["final_norm.w"], gb("final_norm.b"), eps)
    else:
        h = ref.layernorm(h, weights["final_norm.w"], gb("final_norm.b"), eps)
    hook(L, h)
    return h
