"""Trivikrama: run full-size transformer encoders on microcontrollers.

Copyright 2026 Vadhyar Sai Charan
SPDX-License-Identifier: Apache-2.0
"""

from .format import DTYPE_F32, DTYPE_Q4_G32, DTYPE_Q8_ROWS, read_trv, write_trv
from .quantize import dequantize_rows_q8, quantize_rows_q8

__version__ = "0.3.0"
__all__ = [
    "DTYPE_F32",
    "DTYPE_Q8_ROWS",
    "DTYPE_Q4_G32",
    "read_trv",
    "write_trv",
    "quantize_rows_q8",
    "dequantize_rows_q8",
]
