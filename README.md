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
| ModernBERT encoder forward pass | 🚧 next |
| Laya decision head example | 🚧 |
| Ports: ESP8266, ESP32-S3, RP2040 / RP2350 | 🚧 |
| Resident mode for small distilled models | 🚧 |

## Quick start (PC)

```bash
pip install -e .            # installs the `trivi` CLI
make test                   # builds the C core with gcc and checks it against NumPy
trivi convert model.safetensors -o model.trv   # needs: pip install safetensors
trivi inspect model.trv
```

The C core is plain C99 with no allocations and no OS dependencies.
Develop and test on your laptop; the same code compiles for the microcontroller.

## Layout

```
core/include/trivikrama.h   public C API
core/src/trv_kernels.c      linear (int8, streamed), layernorm, gelu, rope, attention
core/src/trv_model.c        .trv reader over any weight source
core/src/trv_source_file.c  stdio weight source (PC builds and tests)
python/trivikrama/          converter, format writer, NumPy reference, `trivi` CLI
tests/                      C harness + NumPy comparison
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
