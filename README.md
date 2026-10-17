# Trivikrama

**A tiny chip, a giant model.**

Trivikrama runs full-size, pretrained transformer encoders (BERT / ModernBERT family)
on microcontrollers by **streaming weights from an SD card**, one tile at a time.
A board with ~100 KB of free RAM can compute a 400M-parameter model, exactly as released,
if you are willing to wait.

> In the Vamana story, a dwarf asks for three steps of land, then grows into
> Trivikrama and measures the universe in three strides.
> The chip is Vamana. The model it runs is Trivikrama.

## The three strides

```
  1. convert  (PC, Python)   Hugging Face checkpoint  ->  .trv file (int8 or int4)
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
| Streamed linear layer, int8 and int4 weights | ✅ verified against NumPy |
| LayerNorm, GELU, RoPE | ✅ verified against NumPy |
| Windowed attention with online softmax | ✅ verified against NumPy |
| ModernBERT encoder forward pass | ✅ matches Hugging Face (`make test-modernbert`); runs the real Laya encoder |
| Virtual-MCU runner (RAM limits, bytes streamed, MACs) | ✅ |
| Speed features: int4, mixed precision, multi-core, async prefetch, storage tiers, batching, early exit, token pruning, SIMD hook | ✅ bit-exact against the plain path (`make test-features`) |
| CI (GitHub Actions) | ✅ |
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
| 16 | 161 KB | 344.6 MB | 5.5 G  | 3.8 s |
| 32 | 313 KB | 344.6 MB | 11.0 G | 7.2 s |
| 64 | 618 KB | 344.6 MB | 22.2 G | 14.1 s |

Every weight is read once per inference (the 51 MB embedding table is only
read for the tokens present, which is why 344.6 MB < 396 MB).

| Board | Fits up to | Estimated time |
|---|---|---|
| ESP32-S3 + 8 MB PSRAM | 64+ tokens | ~1 min (8 tokens) to ~4 min (64) |
| Shrike (RP2350, 520 KB) | 32 tokens | ~3 min (8) to ~5 min (32) |
| Shrike-lite (RP2040, 264 KB) | 16 tokens | ~4 to 5 min |
| NodeMCU (ESP8266) | does not fit yet | needs ~86 KB at 8 tokens; ~40 KB is free |

These board times are for the plain configuration (int8, one core, no
prefetch). With int4 alone the estimates drop by about a quarter to a third
(ESP32-S3, 8 tokens: ~45 s); the other speed features stack on top.

## Speed features

All optional, all model-agnostic, and all checked to give **bit-identical output**
to the plain path (`make test-features`). Measured on the PC with the same
Laya-sized model, 8 tokens, reads slowed to 100 MB/s to behave like a
microcontroller's SD card (`tools/benchmark.py`):

| Configuration | Time / sequence | Read / sequence | vs. baseline |
|---|---|---|---|
| int8, sync, 1 core (baseline) | 8.18 s | 345 MB | 1.0× |
| int8 + async prefetch | 7.19 s | 345 MB | 1.1× |
| int8 + async + 2 cores | 6.88 s | 345 MB | 1.2× |
| int4, sync, 1 core | 6.10 s | 193 MB | 1.3× |
| int4 + async + 2 cores | 4.05 s | 193 MB | 2.0× |
| int4 + async + 2 cores + 64 MB fast-flash tier | 3.18 s | 193 MB | 2.6× |
| ... + batch of 4 sequences (time per sequence) | 1.58 s | 48 MB | 5.2× |
| ... or early exit after 14 of 28 layers | 1.28 s | 97 MB | 6.4× |

Early exit and int4 change the model's output; on the real Laya encoder
they cost too much accuracy (see below), so the useful rows for Laya are the
exact ones: async, multi-core, flash tier and batching. Measured on Laya
itself (int8, 8 tokens, simulated 100 MB/s card): 9.03 s plain, 7.14 s with
async + 2 cores, about 1.6 s per sequence in a batch of 4.

### Accuracy on the real Laya encoder

Laya's released weights, converted with `trivi export-modernbert`, compared
token by token with Hugging Face's float32 ModernBERT on real sentences:

| Weights | Mean token cosine vs. float32 | Worst token |
|---|---|---|
| int8 | 0.994 – 0.999 | 0.98 |
| `--weights auto` (7 of 112 matrices int4, file 386 MB vs 396 MB) | 0.989 – 0.996 | 0.97 |
| int4 everywhere | 0.94 – 0.97 | 0.87 |
| int8, early exit after layer 14 of 28 | 0.51 – 0.62 | 0.06 |

What this means for Laya:

- **int8 is the right choice.** It is accurate, and the exact speed features
  (async prefetch, multi-core, storage tiers, batching) all apply to it.
- **int4 buys almost nothing here.** Only a handful of matrices tolerate it,
  saving about 3% of the file. Plain group-wise int4 is too coarse for this
  encoder; better int4 methods (activation-aware scaling, GPTQ-style
  rounding) are future work.
- **Early exit is not usable on Laya as released**: half the layers gives a
  very different representation. It would need a head trained on an
  intermediate layer. The early-exit hook stays useful for models built for it.

| Feature | What it does | How to use |
|---|---|---|
| int4 weights | 4-bit, groups of 32, fp16 scales (error-minimizing scale search): ~45% fewer bytes | `trivi export-modernbert --weights int4` |
| Calibrated mixed precision | int4 only for the matrices whose int4 version keeps the model's output on your calibration text within a budget of the int8 model | `--weights auto --max-drop 0.005 --calib-text FILE` |
| Async prefetch | loads the next weight tile, even the next call's, while computing | give `trv_source` a `read_async` + `wait` (DMA, second core) |
| Multi-core | splits each tile's rows across cores | `trv_runtime.parallel` + `n_workers` |
| SIMD hook | swap in a vector int8 dot product (e.g. ESP32-S3 PIE) | `trv_runtime.dot_groups` |
| Storage tiers | first layers from internal flash/PSRAM, the rest from SD | `trv_source_tiered_init()`; the exporter puts layers in forward order |
| Batching | N sequences in one pass, every weight still read once | `trv_mbert_forward_batch()` |
| Early exit | stop after any layer | hook sets `ctl->stop = 1` |
| Token pruning | drop tokens between layers | hook clears `ctl->keep[t]` |

```c
/* plug in a board's SIMD dot product and its second core */
trv_runtime rt = {0};
rt.dot_groups = my_simd_dot;          /* acc[j] = dot of chunk j */
rt.parallel = my_run_on_both_cores;   /* runs fn(arg, 0, 2) and fn(arg, 1, 2) */
rt.n_workers = 2;
trv_set_runtime(&rt);
```

## Quick start (PC)

```bash
pip install -e ".[convert]"     # installs the `trivi` CLI
make test-kernels               # C kernels vs NumPy
make test-features              # every speed feature is bit-exact vs the plain path
make test-export                # exporter: nested configs, int4 scale search, calibration
make test-modernbert            # C ModernBERT vs Hugging Face (needs: pip install torch transformers)

# a real model: download a ModernBERT checkpoint folder, then
trivi export-modernbert ./ModernBERT-large -o model.trv              # int8 (recommended)
trivi export-modernbert ./ModernBERT-large -o mixed.trv --weights auto \
      --max-drop 0.005 --calib-text my_examples.txt                    # int4 only where safe
make virtual-mcu MODEL=model.trv   # RAM per board, bytes streamed, time estimates
build/trv_run model.trv --ids 101,2023,2003,102 --dump out/   # per-layer hidden states
python tools/benchmark.py build/trv_run model.trv model4.trv    # every speed feature
```

The C core is plain C99 with no allocations and no OS dependencies.
Develop and test on your laptop; the same code compiles for the microcontroller.

## Layout

```
core/include/trivikrama.h   public C API
core/src/trv_linear.c       streamed linear layer (int8/int4, multi-core, double buffering)
core/src/trv_kernels.c      quantization, layernorm, gelu, rope, attention
core/src/trv_sources.c      memory source, storage tiers
core/src/trv_model.c        .trv reader over any weight source
core/src/trv_source_file.c  stdio weight source (PC builds and tests)
core/src/trv_posix.c        PC only: thread pool, async + SD-speed-simulating file source
core/src/trv_modernbert.c   ModernBERT encoder: batching, early exit, token pruning
python/trivikrama/          converter, ModernBERT exporter, NumPy reference, `trivi` CLI
tools/trv_run.c             run a model on the PC as a "virtual MCU" (RAM budget, stats)
tools/virtual_mcu.py        per-board RAM fit and time estimates
tools/benchmark.py          measure every speed feature on a model
tests/                      C vs NumPy, features vs plain path, exporter, C vs Hugging Face
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
