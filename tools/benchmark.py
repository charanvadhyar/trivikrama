"""Benchmark every speed feature on one model, on the PC.

    python tools/benchmark.py build/trv_run model_int8.trv [model_int4.trv]
           [--tokens 8] [--sd-mbps 100] [--threads 2]

Two settings:
  * "full speed": weights come from the OS file cache, so the laptop is
    compute-bound (threads help, storage tricks barely matter)
  * "simulated SD card": reads are slowed to --sd-mbps, so storage dominates,
    as it does on a microcontroller (async, int4, tiers, early exit help)

Copyright 2026 Vadhyar Sai Charan
SPDX-License-Identifier: Apache-2.0
"""

import argparse
import json
import os
import subprocess


def run(binary, model, *args):
    res = subprocess.run([binary, model, *map(str, args)], capture_output=True, text=True)
    stats = json.loads(res.stdout)
    if not stats.get("ok"):
        raise SystemExit(f"trv_run failed: {res.stdout} {res.stderr}")
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("binary")
    p.add_argument("model8")
    p.add_argument("model4", nargs="?")
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--sd-mbps", type=float, default=100.0)
    p.add_argument("--sd-overhead-ms", type=float, default=0.2)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--flash-mb", type=float, default=64.0)
    p.add_argument("--batch", type=int, default=4)
    a = p.parse_args()

    T, W = a.tokens, a.threads
    base = ["--tokens", T, "--tile", 262144]
    sd = ["--sd-mbps", a.sd_mbps, "--sd-overhead-ms", a.sd_overhead_ms]
    m8, m4 = a.model8, a.model4
    flash = int(a.flash_mb * 1e6)
    L = run(a.binary, m8, *base, "--exit-after", 0)["layers"]
    half = L // 2 - 1

    rows = []

    def add(group, label, model, *args, per_seq=1):
        s = run(a.binary, model, *base, *args)
        rows.append((group, label, s["seconds"] / per_seq, s["bytes_streamed"] / per_seq,
                     s["layers_run"]))
        print(f"{group:18s} {label:44s} {s['seconds'] / per_seq:7.2f} s", flush=True)

    g = "full speed"
    add(g, "int8, 1 thread (baseline)", m8)
    add(g, f"int8, {W} threads", m8, "--threads", W)
    if m4:
        add(g, f"int4, {W} threads", m4, "--threads", W)

    g = f"SD {a.sd_mbps:g} MB/s"
    add(g, "int8, sync, 1 thread (baseline)", m8, *sd)
    add(g, "int8 + async", m8, *sd, "--async")
    add(g, f"int8 + async + {W} threads", m8, *sd, "--async", "--threads", W)
    if m4:
        add(g, "int4, sync, 1 thread", m4, *sd)
        add(g, f"int4 + async + {W} threads", m4, *sd, "--async", "--threads", W)
        add(g, f"int4 + async + {W} thr + {a.flash_mb:g} MB flash tier", m4, *sd, "--async",
            "--threads", W, "--flash-bytes", flash)
        add(g, f"... + early exit after layer {half}", m4, *sd, "--async", "--threads", W,
            "--flash-bytes", flash, "--exit-after", half)
        add(g, f"... no early exit, batch of {a.batch} (per sequence)", m4, *sd, "--async",
            "--threads", W, "--flash-bytes", flash, "--batch", a.batch, per_seq=a.batch)

    print()
    print(f"Model: {os.path.basename(m8)}" + (f" / {os.path.basename(m4)}" if m4 else "")
          + f", {T} tokens per sequence")
    print()
    print("| Setting | Configuration | Time / sequence | Read / sequence | Layers |")
    print("|---|---|---|---|---|")
    for group, label, sec, by, layers in rows:
        print(f"| {group} | {label} | {sec:.2f} s | {by / 1e6:.0f} MB | {layers} |")


if __name__ == "__main__":
    main()
