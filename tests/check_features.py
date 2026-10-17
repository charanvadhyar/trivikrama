"""Check the speed features of the C runtime. Needs only NumPy.

Every feature must give exactly the same numbers as the plain path (or, for
int4, match the NumPy mirror), so the speedups cost nothing in correctness.

    python tests/check_features.py build/trv_run

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

H, L, NH, I, V = 64, 4, 4, 96, 512
CFG = dict(hidden_size=H, num_hidden_layers=L, num_attention_heads=NH, intermediate_size=I,
           vocab_size=V, local_attention=8, global_attn_every_n_layers=3, norm_eps=1e-5,
           global_rope_theta=160000.0, local_rope_theta=10000.0)

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name:46s} {detail}")


def random_state(seed=0):
    r = np.random.default_rng(seed)
    n = lambda *s: (r.standard_normal(s) * 0.05).astype(np.float32)
    sd = {"embeddings.tok_embeddings.weight": n(V, H), "embeddings.norm.weight": 1 + n(H),
          "final_norm.weight": 1 + n(H)}
    for i in range(L):
        p = f"layers.{i}."
        if i:
            sd[p + "attn_norm.weight"] = 1 + n(H)
        sd[p + "attn.Wqkv.weight"] = n(3 * H, H)
        sd[p + "attn.Wo.weight"] = n(H, H)
        sd[p + "mlp_norm.weight"] = 1 + n(H)
        sd[p + "mlp.Wi.weight"] = n(2 * I, H)
        sd[p + "mlp.Wo.weight"] = n(H, I)
    return sd


class Runner:
    def __init__(self, binary, tmp):
        self.binary, self.tmp, self.n = binary, tmp, 0

    def __call__(self, model, *args):
        self.n += 1
        d = os.path.join(self.tmp, f"run{self.n}")
        os.makedirs(d)
        res = subprocess.run([self.binary, model, "--dump", d, *map(str, args)],
                             capture_output=True, text=True)
        stats = json.loads(res.stdout) if res.stdout.strip() else {}
        stats["returncode"] = res.returncode
        if res.returncode == 0:
            T = stats["tokens_out"]
            stats["final"] = np.fromfile(os.path.join(d, "final.bin"), "<f4").reshape(T, H)
            stats["index"] = np.fromfile(os.path.join(d, "index.bin"), "<u4")
            stats["dir"] = d
        elif res.returncode != 3:
            print(res.stdout, res.stderr)
            sys.exit(f"trv_run failed: {args}")
        return stats


def main(binary):
    sd = random_state()
    r = np.random.default_rng(7)
    a = r.integers(5, V, 20).tolist()
    b = r.integers(5, V, 13).tolist()
    ids_a = ",".join(map(str, a))
    ids_b = ",".join(map(str, b))

    with tempfile.TemporaryDirectory() as tmp:
        run = Runner(binary, tmp)
        m8 = os.path.join(tmp, "m8.trv")
        m4 = os.path.join(tmp, "m4.trv")
        mb.export_modernbert(sd, CFG, m8, weights="int8")
        mb.export_modernbert(sd, CFG, m4, weights="int4")

        print("int4 weights")
        base8 = run(m8, "--ids", ids_a, "--tile", 2048)
        base4 = run(m4, "--ids", ids_a, "--tile", 2048)
        w4, c4 = mb.load_trv_modernbert(m4)
        mirror = mb.forward(w4, c4, a, quantized=True)
        wf, _ = mb.rename_state_dict(sd)
        flt = mb.forward(wf, CFG, a, quantized=False)
        rel = np.abs(base4["final"] - mirror).max() / np.abs(mirror).max()
        check("C int4 == NumPy int4 mirror", rel < 2e-3, f"max rel err {rel:.1e}")
        cos4 = float((base4["final"].ravel() @ flt.ravel())
                     / np.linalg.norm(base4["final"]) / np.linalg.norm(flt))
        cos8 = float((base8["final"].ravel() @ flt.ravel())
                     / np.linalg.norm(base8["final"]) / np.linalg.norm(flt))
        check("int4 close to float", cos4 > 0.99, f"cosine int4 {cos4:.5f}  int8 {cos8:.5f}")
        ratio = base4["bytes_streamed"] / base8["bytes_streamed"]
        check("int4 streams fewer bytes", ratio < 0.65, f"{ratio:.2f}x the int8 bytes")

        for label, model, base in (("int8", m8, base8), ("int4", m4, base4)):
            print(f"multi-core, async, tiers ({label})")
            t2 = run(model, "--ids", ids_a, "--tile", 2048, "--threads", 2)
            check("2 threads == 1 thread (bit-exact)", np.array_equal(t2["final"], base["final"]))
            t4 = run(model, "--ids", ids_a, "--tile", 2048, "--threads", 4)
            check("4 threads == 1 thread (bit-exact)", np.array_equal(t4["final"], base["final"]))
            asy = run(model, "--ids", ids_a, "--tile", 2048, "--async")
            check("async double-buffered == sync (bit-exact)",
                  np.array_equal(asy["final"], base["final"]))
            check("async reads exactly the same bytes", asy["bytes_streamed"] == base["bytes_streamed"],
                  f"{asy['bytes_streamed']:,} B")
            both = run(model, "--ids", ids_a, "--tile", 2048, "--async", "--threads", 2)
            check("async + 2 threads == sync (bit-exact)",
                  np.array_equal(both["final"], base["final"]))
            size = os.path.getsize(model)
            tier = run(model, "--ids", ids_a, "--tile", 2048, "--flash-bytes", size // 3)
            check("flash tier + SD tier == single source (bit-exact)",
                  np.array_equal(tier["final"], base["final"]))
            split = tier["flash_bytes_read"] + tier["sd_bytes_read"]
            check("tier byte counters add up", split == base["bytes_streamed"],
                  f"flash {tier['flash_bytes_read']:,} + SD {tier['sd_bytes_read']:,}")
            tasy = run(model, "--ids", ids_a, "--tile", 2048, "--flash-bytes", size // 3,
                       "--async", "--threads", 2)
            check("tiers + async + threads == plain (bit-exact)",
                  np.array_equal(tasy["final"], base["final"]))

        print("batching")
        sa = run(m8, "--ids", ids_a, "--tile", 2048)
        sb = run(m8, "--ids", ids_b, "--tile", 2048)
        bt = run(m8, "--ids", f"{ids_a};{ids_b}", "--tile", 2048)
        same = (np.array_equal(bt["final"][: len(a)], sa["final"])
                and np.array_equal(bt["final"][len(a):], sb["final"]))
        check("batch [A;B] == A and B run separately", same)
        extra = bt["bytes_streamed"] - sa["bytes_streamed"]
        check("batch reads the weights once", extra < 0.05 * sa["bytes_streamed"],
              f"2 sequences: {bt['bytes_streamed']:,} B vs {sa['bytes_streamed']:,} B for one")

        print("token pruning")
        drop = [3, 4, 11, 17]
        mask = "".join("0" if i in drop else "1" for i in range(len(a)))
        masked = run(m8, "--ids", ids_a, "--tile", 2048, "--mask", mask)
        pruned = run(m8, "--ids", ids_a, "--tile", 2048, "--drop-at", -1,
                     "--drop", ",".join(map(str, drop)))
        keep = [i for i in range(len(a)) if i not in drop]
        check("dropped tokens are gone", pruned["index"].tolist() == keep,
              f"{pruned['tokens_out']} of {len(a)} tokens left")
        check("pruned == masked run on the kept tokens (bit-exact)",
              np.array_equal(pruned["final"], masked["final"][keep]))
        mid = run(m8, "--ids", f"{ids_a};{ids_b}", "--tile", 2048, "--prune-at", 1,
                  "--prune-keep", 0.5)
        check("mid-model pruning in a batch runs",
              0 < mid["tokens_out"] < len(a) + len(b),
              f"{mid['tokens_out']} of {len(a) + len(b)} tokens after layer 1")

        print("early exit")
        ex = run(m8, "--ids", ids_a, "--tile", 2048, "--exit-after", 1)
        h1 = np.fromfile(os.path.join(sa["dir"], "layer001.bin"), "<f4").reshape(len(a), H)
        w8, _ = mb.load_trv_modernbert(m8)
        want = mb._ln(h1, w8["final_norm.w"], w8.get("final_norm.b"), CFG["norm_eps"])
        check("exit after layer 1 == final norm of layer 1",
              np.abs(ex["final"] - want).max() < 1e-5, f"ran {ex['layers_run']} of {L} layers")
        check("early exit streams fewer bytes",
              ex["bytes_streamed"] < 0.6 * sa["bytes_streamed"],
              f"{ex['bytes_streamed']:,} vs {sa['bytes_streamed']:,} B")

        print("RAM budget")
        over = run(m8, "--ids", ids_a, "--budget", 1000)
        check("refuses to run over the RAM budget", over["returncode"] == 3,
              f"needs {over.get('ram_bytes', 0):,} B")

    if not all(results):
        sys.exit("feature checks FAILED")
    print(f"all {len(results)} feature checks passed")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "build/trv_run")
