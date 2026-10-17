# Trivikrama

**A tiny chip, a giant model.**

Trivikrama runs full-size, pretrained transformer encoders (BERT / ModernBERT family)
on microcontrollers by **streaming weights from an SD card**, one tile at a time.
A board with ~50 KB of free RAM can compute a 400M-parameter model, exactly as released,
if you are willing to wait.

> In the Vamana story, a dwarf asks for three steps of land, then grows into
> Trivikrama and measures the universe in three strides.
> The chip is Vamana. The model it runs is Trivikrama.

## The three strides

```
  1. convert  (PC, Python)   Hugging Face checkpoint  ->  .trv file (int8, per-row scales)
  2. stream   (MCU, C)       .trv on SD card          ->  one weight tile in RAM at a time
  3. infer    (MCU, C)       tile applied to ALL tokens, so every weight is read once
```

## Why it works in tiny RAM

| Trick | Effect |
|---|---|
| Token-batched weight tiles | Each weight is read from storage **once per inference**, not once per token |
| Per-token FFN | The wide FFN intermediate never exists in full |
| Per-head attention with online softmax | No `T x T` score matrix; attention fits in a few KB |
| Sliding-window masks | Matches ModernBERT's local-attention layers natively |
| Pluggable weight source | SD card, SPI flash, a file on a PC, or anything with a `read()` |

## Status

| Part | State |
|---|---|
| `.trv` model format (writer + C reader) | ✅ |
| Int8 tiled streaming linear layer | ✅ verified against NumPy |
| LayerNorm, GELU, RoPE | ✅ verified against NumPy |
| Windowed attention with online softmax | ✅ verified against NumPy |
| ModernBERT encoder forward pass | ✅ matches the NumPy mirror to ~1e-7; Hugging Face check: `make test-modernbert` |
| Virtual-MCU runner (RAM limits, bytes streamed, MACs) | ✅ |
| Laya decision head example | 🚧 next |
| Ports: ESP32-S3, RP2040 / RP2350 | 🚧 |
| ESP8266 (needs a smaller residual stream, e.g. fp16) | 🚧 |
| Resident mode for small distilled models | 🚧 |

## Virtual-MCU results

A model with the exact shape of ModernBERT-large (Laya's encoder: 28 layers,
hidden 1024, 394.8M parameters, 396 MB as int8) run through `make virtual-mcu`.
RAM and bytes are measured by the real C code; times are estimates for plain
scalar C until measured on hardware.

| Tokens | RAM (activations + scratch) | Weights streamed | Multiply-accumulates | Laptop |
|---|---|---|---|---|
| 8  | 84 KB  | 344.6 MB | 2.8 G  | 2.0 s |
| 16 | 160 KB | 344.6 MB | 5.5 G  | 3.8 s |
| 32 | 312 KB | 344.6 MB | 11.0 G | 7.2 s |
| 64 | 617 KB | 344.6 MB | 22.2 G | 14.1 s |

Every weight is read once per inference (the 51 MB embedding table is only
read for the tokens present, which is why 344.6 MB < 396 MB).

| Board | Fits up to | Estimated time |
|---|---|---|
| ESP32-S3 + 8 MB PSRAM | 64+ tokens | ~1 min (8 tokens) to ~4 min (64) |
| Shrike (RP2350, 520 KB) | 32 tokens | ~3 min (8) to ~5 min (32) |
| Shrike-lite (RP2040, 264 KB) | 16 tokens | ~4 to 5 min |
| NodeMCU (ESP8266) | does not fit yet | needs ~86 KB at 8 tokens; ~40 KB is free |

## Quick start (PC)

```bash
pip install -e ".[convert]"     # installs the `trivi` CLI
make test-kernels               # C kernels vs NumPy
make test-modernbert            # C ModernBERT vs Hugging Face (needs: pip install torch transformers)

# a real model: download a ModernBERT checkpoint folder, then
trivi export-modernbert ./ModernBERT-large -o model.trv
make virtual-mcu MODEL=model.trv   # RAM per board, bytes streamed, time estimates
build/trv_run model.trv --ids 101,2023,2003,102 --dump out/   # per-layer hidden states
```

The C core is plain C99 with no allocations and no OS dependencies.
Develop and test on your laptop; the same code compiles for the microcontroller.

## Layout

```
core/include/trivikrama.h   public C API
core/src/trv_kernels.c      linear (int8, streamed), layernorm, gelu, rope, attention
core/src/trv_model.c        .trv reader over any weight source
core/src/trv_source_file.c  stdio weight source (PC builds and tests)
core/src/trv_modernbert.c   ModernBERT encoder forward pass
python/trivikrama/          converter, ModernBERT exporter, NumPy reference, `trivi` CLI
tools/trv_run.c             run a model on the PC as a "virtual MCU" (RAM budget, stats)
tools/virtual_mcu.py        per-board RAM fit and time estimates
tests/                      C vs NumPy, C vs Hugging Face
```

## Prior work

Trivikrama builds on ideas from:

- Miao & Lin, *Enabling Large Neural Networks on Tiny Microcontrollers with Swapping* (out-of-core MCU inference)
- Yang et al., *MCUBERT: Memory-Efficient BERT Inference on Commodity Microcontrollers*
- Jung et al., *Optimizing the Deployment of Tiny Transformers on Low-Power MCUs* (pulp-transformer)
- OnnxStream (weight streaming on single-board computers)
- Dao et al., *FlashAttention* (online softmax)

## Author

Created by **Vadhyar Sai Charan**.

If you use Trivikrama in a project, report, or paper, please credit it. GitHub's
"Cite this repository" button uses [`CITATION.cff`](CITATION.cff).

## License

Apache License 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
Redistributions must keep the NOTICE file.
