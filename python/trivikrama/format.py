"""The .trv model file: a flat, seekable layout the C core reads tile by tile.

    header (16 bytes): b"TRVK" | u32 version | u32 n_tensors | u32 table_offset
    table entry:       u16 name_len | name | u8 dtype | u8 ndim |
                       u32 dims[ndim] | u32 data_offset | u32 data_nbytes
    DTYPE_F32:         raw float32
    DTYPE_Q8_ROWS:     [N, K]: N float32 row scales, then N*K int8

All integers little-endian. Data blocks are 16-byte aligned.

Copyright 2026 Vadhyar Sai Charan
SPDX-License-Identifier: Apache-2.0
"""

import struct

import numpy as np

from .quantize import quantize_rows_q8

MAGIC = b"TRVK"
VERSION = 1
DTYPE_F32 = 0
DTYPE_Q8_ROWS = 1
MAX_NAME = 96
MAX_DIMS = 4
ALIGN = 16
MAX_FILE_BYTES = 2**32 - 1


def _align(n, a=ALIGN):
    return (n + a - 1) // a * a


def _encode(arr, dtype):
    if dtype == DTYPE_F32:
        a = np.ascontiguousarray(arr, dtype="<f4")
        return a.shape, a.tobytes()
    if dtype == DTYPE_Q8_ROWS:
        if arr.ndim != 2:
            raise ValueError("Q8_ROWS tensors must be 2-D")
        q, scale = quantize_rows_q8(arr)
        return arr.shape, scale.astype("<f4").tobytes() + q.tobytes()
    raise ValueError(f"unknown dtype {dtype}")


def write_trv(path, tensors):
    """Write tensors to a .trv file.

    tensors: iterable of (name, array, dtype) with dtype DTYPE_F32 or DTYPE_Q8_ROWS.
    Returns the total file size in bytes.
    """
    items = []
    for name, arr, dtype in tensors:
        name_b = name.encode("utf-8")
        if len(name_b) > MAX_NAME:
            raise ValueError(f"tensor name longer than {MAX_NAME} bytes: {name}")
        shape, blob = _encode(np.asarray(arr), dtype)
        if not 1 <= len(shape) <= MAX_DIMS:
            raise ValueError(f"{name}: need 1..{MAX_DIMS} dims, got {shape}")
        items.append((name_b, dtype, shape, blob))

    table_offset = 16
    table_size = sum(2 + len(n) + 2 + 4 * len(s) + 8 for n, _, s, _ in items)
    offset = _align(table_offset + table_size)
    placed = []
    for name_b, dtype, shape, blob in items:
        placed.append((name_b, dtype, shape, blob, offset))
        offset = _align(offset + len(blob))
    if offset > MAX_FILE_BYTES:
        raise ValueError(".trv files are limited to 4 GiB (u32 offsets)")

    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<III", VERSION, len(placed), table_offset))
        for name_b, dtype, shape, blob, off in placed:
            f.write(struct.pack("<H", len(name_b)) + name_b)
            f.write(struct.pack("<BB", dtype, len(shape)))
            f.write(struct.pack(f"<{len(shape)}I", *shape))
            f.write(struct.pack("<II", off, len(blob)))
        for _, _, _, blob, off in placed:
            f.write(b"\0" * (off - f.tell()))
            f.write(blob)
        f.write(b"\0" * (offset - f.tell()))
    return offset


def read_trv(path, dequantize=True):
    """Read a .trv file into {name: (dtype, array)}.

    Q8_ROWS tensors come back de-quantized to float32 unless dequantize=False,
    in which case the value is (q_int8, scale_f32).
    """
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != MAGIC:
        raise ValueError("not a .trv file")
    version, n, table = struct.unpack_from("<III", data, 4)
    if version != VERSION:
        raise ValueError(f"unsupported .trv version {version}")
    out = {}
    pos = table
    for _ in range(n):
        (ln,) = struct.unpack_from("<H", data, pos)
        pos += 2
        name = data[pos : pos + ln].decode("utf-8")
        pos += ln
        dtype, ndim = struct.unpack_from("<BB", data, pos)
        pos += 2
        shape = struct.unpack_from(f"<{ndim}I", data, pos)
        pos += 4 * ndim
        off, nbytes = struct.unpack_from("<II", data, pos)
        pos += 8
        blob = data[off : off + nbytes]
        if dtype == DTYPE_F32:
            out[name] = (dtype, np.frombuffer(blob, dtype="<f4").reshape(shape).copy())
        elif dtype == DTYPE_Q8_ROWS:
            N, K = shape
            scale = np.frombuffer(blob[: 4 * N], dtype="<f4").copy()
            q = np.frombuffer(blob[4 * N :], dtype=np.int8).reshape(N, K).copy()
            out[name] = (dtype, q.astype(np.float32) * scale[:, None] if dequantize else (q, scale))
        else:
            raise ValueError(f"{name}: unknown dtype {dtype}")
    return out
