"""Check the exporter: int4 scale search, nested configs, calibrated precision.

Needs NumPy and safetensors (no torch).

    python tests/check_export.py build/trv_run

Copyright 2026 Vadhyar Sai Charan
SPDX-License-Identifier: Apache-2.0
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from check_features import CFG, H, random_state  # noqa: E402

from trivikrama import DTYPE_Q4_G32, read_trv  # noqa: E402
from trivikrama import modernbert as mb  # noqa: E402
from trivikrama.quantize import dequantize_q4_g32, quantize_q4_g32  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name:50s} {detail}")


def cosine(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(((a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)).mean())


def run_c(binary, model, ids, tmp):
    d = tempfile.mkdtemp(dir=tmp)
    res = subprocess.run([binary, model, "--ids", ",".join(map(str, ids)), "--dump", d],
                         capture_output=True, text=True)
    if res.returncode:
        sys.exit(f"trv_run failed: {res.stdout} {res.stderr}")
    return np.fromfile(os.path.join(d, "final.bin"), "<f4").reshape(len(ids), H)


def main(binary):
    print("int4 scale search")
    r = np.random.default_rng(0)
    w = r.standard_t(4, (256, 512)).astype(np.float32)
    e_max = np.linalg.norm(dequantize_q4_g32(*quantize_q4_g32(w, search=False)) - w)
    e_srch = np.linalg.norm(dequantize_q4_g32(*quantize_q4_g32(w, search=True)) - w)
    check("searched scales beat max scaling", e_srch < e_max,
          f"error {e_srch / np.linalg.norm(w):.4f} vs {e_max / np.linalg.norm(w):.4f}")

    sd = random_state()
    ids = r.integers(5, CFG["vocab_size"], 20).tolist()
    wf, _ = mb.rename_state_dict(sd)
    ref = mb.float_forward(wf, CFG, ids)

    with tempfile.TemporaryDirectory() as tmp:
        print("Laya-style folder: stub config.json, real one in encoder/, prefixed weights")
        from safetensors.numpy import save_file

        folder = os.path.join(tmp, "laya_like")
        os.makedirs(os.path.join(folder, "encoder"))
        with open(os.path.join(folder, "config.json"), "w") as f:
            json.dump({"model_type": "laya", "architectures": ["LayaTypedDecisions"]}, f)
        with open(os.path.join(folder, "encoder", "config.json"), "w") as f:
            json.dump(CFG, f)
        tensors = {"encoder." + k: v for k, v in sd.items()}
        tensors["scorer.0.weight"] = np.ones(H, np.float32)  # a "decision head" tensor
        save_file(tensors, os.path.join(folder, "model.safetensors"))
        out = os.path.join(tmp, "nested.trv")
        res = subprocess.run([sys.executable, "-m", "trivikrama.cli", "export-modernbert", folder,
                              "-o", out], capture_output=True, text=True,
                             env={**os.environ, "PYTHONPATH": "python"})
        check("CLI finds encoder/config.json on its own", res.returncode == 0,
              (res.stdout.splitlines() or [res.stderr])[0][:60])
        if res.returncode == 0:
            check("head tensors skipped", "scorer.0.weight" in res.stdout)
            c = run_c(binary, out, ids, tmp)
            check("exported model runs and matches float", cosine(c, ref) > 0.999,
                  f"cosine {cosine(c, ref):.5f}")

        print("calibrated auto precision")
        calib = [r.integers(5, CFG["vocab_size"], 16).tolist() for _ in range(4)]
        counts = []
        for target in (0.9998, 0.99995):
            path = os.path.join(tmp, f"auto_{target}.trv")
            rep = []
            mb.export_modernbert(sd, CFG, path, weights="auto", calib_ids=calib,
                                 target_cosine=target, report=rep, log=lambda *_: None)
            in_file = {n for n, (dt, _) in read_trv(path, dequantize=False).items()
                       if dt == DTYPE_Q4_G32}
            chosen = {n for n, d, _ in rep if d == "int4"}
            check(f"target {target}: file holds exactly the chosen int4 set", in_file == chosen,
                  f"{len(chosen)}/{len(rep)} int4")
            c = run_c(binary, path, ids, tmp)
            cs = cosine(c, ref)
            # calibration measures weight quantization; C also quantizes activations
            check(f"target {target}: C output stays near the target", cs > target - 0.002,
                  f"cosine {cs:.5f} on unseen ids")
            counts.append(len(chosen))
        check("stricter target keeps more matrices int8", counts[1] <= counts[0],
              f"{counts[0]} -> {counts[1]} int4 matrices")
        print("relative budget (default)")
        path = os.path.join(tmp, "auto_rel.trv")
        rep_rel = []
        mb.export_modernbert(sd, CFG, path, weights="auto", calib_ids=calib, max_drop=0.0001,
                             report=rep_rel, log=lambda *_: None)
        n_rel = sum(1 for _, d, _ in rep_rel if d == "int4")
        c8 = run_c(binary, os.path.join(tmp, "nested.trv"), ids, tmp)
        crel = run_c(binary, path, ids, tmp)
        drop = cosine(c8, ref) - cosine(crel, ref)
        check("max_drop budget is respected (relative to int8)", drop < 0.0001 + 0.0005,
              f"{n_rel}/{len(rep_rel)} int4, cosine drop {drop:.5f} vs int8")

        sens = dict((n, e) for n, _, e in rep)
        least = min(sens, key=sens.get)
        check("least sensitive matrix is chosen first",
              counts[1] == 0 or least in chosen, least)

    if not all(results):
        sys.exit("export checks FAILED")
    print(f"all {len(results)} export checks passed")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "build/trv_run")
