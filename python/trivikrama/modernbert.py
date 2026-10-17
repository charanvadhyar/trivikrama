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
from .quantize import dequantize_q4_g32, quantize_q4_g32

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


def export_modernbert(state_dict, cfg, path, group=None, weights="int8", int4_max_error=0.12,
                      report=None):
    """Write a ModernBERT encoder to a .trv file. Returns (file_bytes, ignored).

    weights: "int8", "int4", or "auto" (int4 for each layer matrix whose int4
    reconstruction error is at most int4_max_error, int8 otherwise). The
    embedding table always stays int8: only a few rows are read per inference,
    so int4 would save storage but not time. report, if a list, receives
    (name, chosen dtype name, int4 error) per layer matrix.
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
        elif name == "emb.tok" or weights == "int8" or arr.shape[1] % 32:
            dtype = DTYPE_Q8_ROWS
        elif weights == "int4":
            dtype = DTYPE_Q4_G32
        else:
            err = int4_error(arr)
            dtype = DTYPE_Q4_G32 if err <= int4_max_error else DTYPE_Q8_ROWS
            if report is not None:
                report.append((name, "int4" if dtype == DTYPE_Q4_G32 else "int8", err))
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
