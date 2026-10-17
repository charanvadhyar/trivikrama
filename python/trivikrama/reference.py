"""NumPy reference implementations of the C kernels.

Use these to check the C core layer by layer: run the same inputs through
both and compare. Written for clarity, not speed.

Copyright 2026 Vadhyar Sai Charan
SPDX-License-Identifier: Apache-2.0
"""

import math

import numpy as np

from .quantize import quantize_rows_q8

_erf = np.vectorize(math.erf, otypes=[np.float64])


def linear_q8(x, w_q, w_scale, bias=None):
    """Same arithmetic as trv_linear_q8_stream: int8 x int8 -> int32, then rescale."""
    xq, xs = quantize_rows_q8(x)
    acc = xq.astype(np.int64) @ w_q.astype(np.int64).T
    y = (acc.astype(np.float32) * xs[:, None]) * w_scale[None, :]
    if bias is not None:
        y = y + bias.astype(np.float32)[None, :]
    return y.astype(np.float32)


def layernorm(x, gamma, beta=None, eps=1e-5):
    x = x.astype(np.float64)
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    y = (x - mu) / np.sqrt(var + eps) * gamma
    if beta is not None:
        y = y + beta
    return y


def gelu(x):
    x = x.astype(np.float64)
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def geglu(x):
    a, gate = np.split(x.astype(np.float64), 2, axis=-1)
    return gelu(a) * gate


def rope(x, n_heads, head_dim, pos0=0, base=10000.0):
    x = x.astype(np.float64).copy()
    T = x.shape[0]
    half = head_dim // 2
    inv_freq = base ** (-2.0 * np.arange(half) / head_dim)
    ang = (pos0 + np.arange(T))[:, None] * inv_freq[None, :]
    c, s = np.cos(ang), np.sin(ang)
    for h in range(n_heads):
        a = x[:, h * head_dim : h * head_dim + half].copy()
        b = x[:, h * head_dim + half : (h + 1) * head_dim].copy()
        x[:, h * head_dim : h * head_dim + half] = a * c - b * s
        x[:, h * head_dim + half : (h + 1) * head_dim] = b * c + a * s
    return x


def attention_head(q, k, v, half_window=-1, key_mask=None):
    """Plain softmax attention for one head, with the same masking rules as C."""
    q, k, v = (a.astype(np.float64) for a in (q, k, v))
    T, d = q.shape
    scores = q @ k.T / math.sqrt(d)
    allowed = np.ones((T, T), dtype=bool)
    if half_window >= 0:
        idx = np.arange(T)
        allowed &= np.abs(idx[:, None] - idx[None, :]) <= half_window
    if key_mask is not None:
        allowed &= key_mask.astype(bool)[None, :]
    out = np.zeros((T, d))
    for i in range(T):
        if not allowed[i].any():
            continue
        s = np.where(allowed[i], scores[i], -np.inf)
        p = np.exp(s - s.max())
        out[i] = (p / p.sum()) @ v
    return out
