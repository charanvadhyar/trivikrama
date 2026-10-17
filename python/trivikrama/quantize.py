"""Symmetric int8 quantization, bit-compatible with the C core.

Copyright 2026 Vadhyar Sai Charan
SPDX-License-Identifier: Apache-2.0
"""

import numpy as np


def quantize_rows_q8(x):
    """Per-row symmetric int8: x ~= q * scale[:, None].

    Matches trv_quantize_act_q8 in C exactly (float32 math, round-half-even,
    clamp to [-127, 127], scale 1.0 for all-zero rows).
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {x.shape}")
    maxabs = np.max(np.abs(x), axis=1)
    scale = np.where(maxabs > 0, maxabs / np.float32(127.0), np.float32(1.0)).astype(np.float32)
    q = np.rint(x / scale[:, None])
    q = np.clip(q, -127, 127).astype(np.int8)
    return q, scale


def dequantize_rows_q8(q, scale):
    return q.astype(np.float32) * scale.astype(np.float32)[:, None]


Q4_GROUP = 32


def quantize_q4_g32(x):
    """4-bit symmetric quantization in groups of 32 along each row.

    Returns (q int8 in [-7, 7] of shape [N, K], scales float16 [N, K/32]).
    Matches what the C kernel expects: value = q * float(scale).
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    N, K = x.shape
    if K % Q4_GROUP:
        raise ValueError(f"int4 needs rows that are a multiple of {Q4_GROUP} long, got {K}")
    xg = x.reshape(N, K // Q4_GROUP, Q4_GROUP)
    m = np.abs(xg).max(-1)
    s16 = np.where(m > 0, m / np.float32(7.0), np.float32(1.0)).astype(np.float16)
    s16 = np.where(s16 == 0, np.float16(1.0), s16).astype(np.float16)
    q = np.clip(np.rint(xg / s16.astype(np.float32)[..., None]), -7, 7).astype(np.int8)
    return q.reshape(N, K), s16


def dequantize_q4_g32(q, s16):
    N, K = q.shape
    s = s16.astype(np.float32)
    return (q.reshape(N, K // Q4_GROUP, Q4_GROUP).astype(np.float32) * s[..., None]).reshape(N, K)


def pack_q4_rows(q, s16):
    """Row-major blob: each row = K/32 fp16 scales, then K/2 packed bytes."""
    N, K = q.shape
    u = (q.astype(np.int16) + 8).astype(np.uint8)
    packed = (u[:, 0::2] | (u[:, 1::2] << 4)).astype(np.uint8)
    rows = np.concatenate([s16.astype("<f2").view(np.uint8).reshape(N, -1), packed], axis=1)
    return rows.tobytes()


def unpack_q4_rows(blob, N, K):
    row = 2 * (K // Q4_GROUP) + K // 2
    a = np.frombuffer(blob, dtype=np.uint8).reshape(N, row)
    s16 = a[:, : 2 * (K // Q4_GROUP)].copy().view("<f2").reshape(N, K // Q4_GROUP)
    packed = a[:, 2 * (K // Q4_GROUP):]
    q = np.empty((N, K), dtype=np.int8)
    q[:, 0::2] = (packed & 15).astype(np.int8) - 8
    q[:, 1::2] = (packed >> 4).astype(np.int8) - 8
    return q, s16.astype(np.float16)
