"""Check every C kernel against the NumPy reference.

    python tests/check_kernels.py build/test_kernels

Copyright 2026 Vadhyar Sai Charan
SPDX-License-Identifier: Apache-2.0
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

from trivikrama import DTYPE_F32, DTYPE_Q8_ROWS, read_trv, write_trv
from trivikrama import reference as ref

HEAD_DIM = 8


def main(binary):
    rng = np.random.default_rng(1234)
    T, K, N, D, I, H = 7, 40, 24, 16, 12, 2
    f32 = lambda *s: rng.standard_normal(s).astype(np.float32)

    cases = {
        "x": f32(T, K),
        "w": f32(N, K) * 0.3,
        "b": f32(N),
        "ln_x": f32(T, D) * 3 + 1,
        "ln_g": f32(D),
        "ln_b": f32(D),
        "gg_x": f32(T, 2 * I),
        "rope_x": f32(T, H * HEAD_DIM),
        "q": f32(T, H * HEAD_DIM),
        "k": f32(T, H * HEAD_DIM),
        "v": f32(T, H * HEAD_DIM),
        "kmask": np.array([1, 1, 1, 1, 1, 0, 0], dtype=np.float32),
    }

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cases.trv")
        write_trv(path, [(n, a, DTYPE_Q8_ROWS if n == "w" else DTYPE_F32) for n, a in cases.items()])
        res = subprocess.run([binary, path, tmp], capture_output=True, text=True)
        print(res.stdout, end="")
        if res.returncode != 0:
            print(res.stderr)
            sys.exit("C harness failed")
        load = lambda name, shape: np.fromfile(os.path.join(tmp, name + ".bin"), dtype="<f4").reshape(shape)

        w_q, w_s = read_trv(path, dequantize=False)["w"][1]
        att = np.zeros((T, H * HEAD_DIM))
        sl = lambda h: slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
        att[:, sl(0)] = ref.attention_head(cases["q"][:, sl(0)], cases["k"][:, sl(0)], cases["v"][:, sl(0)],
                                           key_mask=cases["kmask"])
        att[:, sl(1)] = ref.attention_head(cases["q"][:, sl(1)], cases["k"][:, sl(1)], cases["v"][:, sl(1)],
                                           half_window=2)

        checks = [
            ("linear (int8, streamed)", load("linear", (T, N)), ref.linear_q8(cases["x"], w_q, w_s, cases["b"]), 1e-6),
            ("layernorm", load("layernorm", (T, D)), ref.layernorm(cases["ln_x"], cases["ln_g"], cases["ln_b"]), 1e-5),
            ("layernorm (no beta)", load("layernorm_nobias", (T, D)), ref.layernorm(cases["ln_x"], cases["ln_g"]), 1e-5),
            ("geglu", load("geglu", (T, I)), ref.geglu(cases["gg_x"]), 1e-5),
            ("rope", load("rope", (T, H * HEAD_DIM)), ref.rope(cases["rope_x"], H, HEAD_DIM, pos0=3), 1e-5),
            ("attention (mask + window)", load("attention", (T, H * HEAD_DIM)), att, 1e-5),
            ("q8 row lookup", load("q8_row", (K,)), w_q[N - 1].astype(np.float32) * w_s[N - 1], 0.0),
        ]

    # how close is int8 to the true float result? (informational)
    true = cases["x"] @ cases["w"].T + cases["b"]
    q_err = np.abs(ref.linear_q8(cases["x"], w_q, w_s, cases["b"]) - true).max() / np.abs(true).max()

    failed = False
    for name, got, want, tol in checks:
        err = float(np.max(np.abs(got - want)))
        ok = err <= tol * max(1.0, float(np.max(np.abs(want))))
        failed |= not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name:28s} max abs err {err:.2e}")
    print(f"  info  int8 vs float32 linear: relative error {q_err:.2%}")
    if failed:
        sys.exit("some kernels do not match the reference")
    print("all kernels match the NumPy reference")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "build/test_kernels")
