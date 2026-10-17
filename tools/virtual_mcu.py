"""Virtual microcontroller: what would this model need on each board?

Runs the real C forward pass on the PC with trv_run (which measures RAM,
bytes streamed and multiply-accumulates exactly), then applies each board's
RAM limit and rough speed figures.

    python tools/virtual_mcu.py build/trv_run model.trv [--tokens 8,16,32,64]

RAM and bytes are exact. Times are ESTIMATES from the board profiles below:
plain scalar C, no SIMD, one core. Replace them with numbers you measure on
real hardware.

Copyright 2026 Vadhyar Sai Charan
SPDX-License-Identifier: Apache-2.0
"""

import argparse
import json
import subprocess

# usable_ram: bytes left for Trivikrama after firmware, SD/FAT driver and stack
# macs_per_s: int8 multiply-accumulates per second, plain C, single core
# sd_bytes_per_s, sd_read_overhead_s: sustained read speed and per-read cost
BOARDS = [
    dict(name="NodeMCU (ESP8266)", usable_ram=40_000, tile=2_048,
         macs_per_s=30e6, sd_bytes_per_s=1.5e6, sd_read_overhead_s=0.5e-3),
    dict(name="Shrike-lite (RP2040)", usable_ram=200_000, tile=8_192,
         macs_per_s=40e6, sd_bytes_per_s=2.5e6, sd_read_overhead_s=0.3e-3),
    dict(name="Shrike (RP2350)", usable_ram=440_000, tile=16_384,
         macs_per_s=80e6, sd_bytes_per_s=2.5e6, sd_read_overhead_s=0.3e-3),
    dict(name="ESP32-S3 + 8 MB PSRAM", usable_ram=7_000_000, tile=32_768,
         macs_per_s=120e6, sd_bytes_per_s=10e6, sd_read_overhead_s=0.2e-3),
]


def fmt_bytes(n):
    for unit, k in (("MB", 1e6), ("KB", 1e3)):
        if n >= k:
            return f"{n / k:.1f} {unit}"
    return f"{n} B"


def fmt_time(s):
    if s >= 3600:
        return f"{s / 3600:.1f} h"
    if s >= 60:
        return f"{s / 60:.1f} min"
    return f"{s:.1f} s"


def measure(binary, model, T, tile):
    res = subprocess.run([binary, model, "--tokens", str(T), "--tile", str(tile)],
                         capture_output=True, text=True)
    stats = json.loads(res.stdout)
    if not stats.get("ok"):
        raise SystemExit(f"trv_run failed: {res.stdout} {res.stderr}")
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("binary")
    p.add_argument("model")
    p.add_argument("--tokens", default="8,16,32,64")
    a = p.parse_args()

    rows = []
    for T in [int(x) for x in a.tokens.split(",")]:
        s = measure(a.binary, a.model, T, tile=4096)
        base_ram = s["ram_bytes"] - s["tile_bytes"]  # activations + scratch
        print(f"T={T:4d}: laptop {s['seconds']:.2f} s, {s['macs'] / 1e9:.2f} G MAC, "
              f"streamed {fmt_bytes(s['bytes_streamed'])}, needs {fmt_bytes(base_ram)} + tile")
        for b in BOARDS:
            ram = base_ram + b["tile"]
            fits = ram <= b["usable_ram"]
            reads = s["bytes_streamed"] / b["tile"] * 2  # scales + rows per tile
            est = (s["macs"] / b["macs_per_s"]
                   + s["bytes_streamed"] / b["sd_bytes_per_s"]
                   + reads * b["sd_read_overhead_s"])
            rows.append((b["name"], T, ram, fits, est))

    print()
    print(f"| Board | Tokens | RAM needed | Fits | Est. time / inference |")
    print(f"|---|---|---|---|---|")
    for name, T, ram, fits, est in rows:
        print(f"| {name} | {T} | {fmt_bytes(ram)} | {'yes' if fits else 'NO'} | "
              f"{fmt_time(est) if fits else '-'} |")
    print("\nRAM and bytes are measured. Times are estimates from tools/virtual_mcu.py BOARDS.")


if __name__ == "__main__":
    main()
