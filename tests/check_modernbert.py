"""Check the C ModernBERT forward pass, layer by layer, against Hugging Face.

Builds a small randomly initialized ModernBERT with `transformers` (no download),
exports it to .trv, runs the C binary, and makes three comparisons:

  1. NumPy float forward  vs Hugging Face   -> is our reading of the architecture right?
  2. C (int8)             vs NumPy mirror   -> does the C code do what we think?
  3. C (int8)             vs Hugging Face   -> how much does int8 cost?

    python tests/check_modernbert.py build/trv_run

Copyright 2026 Vadhyar Sai Charan
SPDX-License-Identifier: Apache-2.0
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

from trivikrama import modernbert as mb


def cosine(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


def hf_reference(seed=0):
    import torch
    from transformers import ModernBertConfig, ModernBertModel

    torch.manual_seed(seed)
    cfg = ModernBertConfig(
        vocab_size=512,
        hidden_size=64,
        num_hidden_layers=4,          # layers 0 and 3 global, 1 and 2 local
        num_attention_heads=4,        # head_dim 16
        intermediate_size=96,
        local_attention=8,            # half window 4, so T=20 really exercises it
        global_attn_every_n_layers=3,
        max_position_embeddings=256,
        initializer_range=0.05,
        attn_implementation="eager",
    )
    model = ModernBertModel(cfg).eval()
    T = 20
    ids = torch.randint(5, cfg.vocab_size, (1, T), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True)
    hs = [x[0].numpy().astype(np.float64) for x in out.hidden_states]
    final = out.last_hidden_state[0].numpy().astype(np.float64)
    state = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    return cfg.to_dict(), state, ids[0].tolist(), hs, final


def main(binary):
    cfg, state, ids, hf_hidden, hf_final = hf_reference()
    L = cfg["num_hidden_layers"]
    # HF hidden_states = (embeddings, after layer 0, ..., after layer L-1); final norm after
    assert len(hf_hidden) == L + 1, f"unexpected hidden_states length {len(hf_hidden)}"
    hf = {-1: hf_hidden[0], **{i: hf_hidden[i + 1] for i in range(L)}, L: hf_final}
    names = {-1: "embeddings", L: "final norm", **{i: f"layer {i}" for i in range(L)}}
    ok = True

    print("1. NumPy float forward vs Hugging Face (architecture check)")
    w, ignored = mb.rename_state_dict(state)
    mine = {}
    mb.forward(w, cfg, ids, quantized=False, hook=lambda i, h: mine.__setitem__(i, h))
    for i in sorted(hf):
        e = rel(mine[i], hf[i])
        good = e < 1e-4
        ok &= good
        print(f"   {'PASS' if good else 'FAIL'}  {names[i]:11s} max rel err {e:.1e}")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tiny.trv")
        mb.export_modernbert(state, cfg, path)
        res = subprocess.run(
            [binary, path, "--ids", ",".join(map(str, ids)), "--tile", "512", "--dump", tmp],
            capture_output=True, text=True,
        )
        if res.returncode != 0:
            print(res.stdout, res.stderr)
            sys.exit("C forward pass failed")
        stats = json.loads(res.stdout)
        T, H = len(ids), cfg["hidden_size"]
        load = lambda f: np.fromfile(os.path.join(tmp, f), dtype="<f4").reshape(T, H)
        c = {-1: load("emb.bin"), **{i: load(f"layer{i:03d}.bin") for i in range(L + 1)}}

        weights, qcfg = mb.load_trv_modernbert(path)
        mirror = {}
        mb.forward(weights, qcfg, ids, quantized=True, hook=lambda i, h: mirror.__setitem__(i, h))

    print("2. C (int8) vs NumPy mirror of the same pipeline (C correctness)")
    for i in sorted(hf):
        e = rel(c[i], mirror[i])
        good = e < 2e-3
        ok &= good
        print(f"   {'PASS' if good else 'FAIL'}  {names[i]:11s} max rel err {e:.1e}")

    print("3. C (int8) vs Hugging Face float32 (cost of int8)")
    for i in sorted(hf):
        cs = cosine(c[i], hf[i])
        good = cs > 0.99
        ok &= good
        print(f"   {'PASS' if good else 'FAIL'}  {names[i]:11s} cosine {cs:.5f}   max rel err {rel(c[i], hf[i]):.1e}")

    print(
        f"   streamed {stats['bytes_streamed']:,} bytes of a {stats['file_bytes']:,}-byte model "
        f"in {stats['reads']:,} reads; arena {stats['workspace_bytes']:,} B"
    )
    if not ok:
        sys.exit("ModernBERT check FAILED")
    print("ModernBERT forward pass matches Hugging Face")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "build/trv_run")
