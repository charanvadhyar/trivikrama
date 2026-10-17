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
