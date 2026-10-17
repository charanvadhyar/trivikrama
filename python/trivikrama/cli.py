"""The `trivi` command line tool.

    trivi convert model.safetensors -o model.trv
    trivi export-modernbert ./laya -o laya.trv
    trivi inspect model.trv

Copyright 2026 Vadhyar Sai Charan
SPDX-License-Identifier: Apache-2.0
"""

import argparse
import fnmatch
import sys

import numpy as np

from . import __version__
from .format import DTYPE_F32, DTYPE_Q8_ROWS, read_trv, write_trv

_DTYPE_NAMES = {DTYPE_F32: "f32", DTYPE_Q8_ROWS: "q8_rows"}


def _load_safetensors(path):
    try:
        from safetensors.numpy import load_file
    except ImportError:
        sys.exit("trivi convert needs safetensors:  pip install 'trivikrama[convert]'")
    try:
        return load_file(path)
    except Exception:
        # bf16 checkpoints cannot be loaded by the numpy backend; go through torch
        try:
            from safetensors.torch import load_file as load_torch
        except ImportError:
            sys.exit("this checkpoint needs torch to read (bf16?):  pip install torch")
        return {k: v.float().numpy() for k, v in load_torch(path).items()}


def cmd_convert(args):
    tensors = _load_safetensors(args.input)
    plan = []
    for name in sorted(tensors):
        arr = np.asarray(tensors[name], dtype=np.float32)
        keep = any(fnmatch.fnmatch(name, pat) for pat in args.keep_f32)
        dtype = DTYPE_Q8_ROWS if arr.ndim == 2 and not keep else DTYPE_F32
        plan.append((name, arr, dtype))
    size = write_trv(args.output, plan)
    n_q8 = sum(1 for _, _, d in plan if d == DTYPE_Q8_ROWS)
    print(f"wrote {args.output}: {len(plan)} tensors ({n_q8} int8), {size / 1e6:.1f} MB")


def cmd_export_modernbert(args):
    import glob
    import json
    import os

    from .modernbert import export_modernbert

    with open(os.path.join(args.model_dir, "config.json")) as f:
        cfg = json.load(f)
    files = sorted(glob.glob(os.path.join(args.model_dir, "*.safetensors")))
    if not files:
        sys.exit(f"no .safetensors files in {args.model_dir}")
    state = {}
    for path in files:
        state.update(_load_safetensors(path))
    size, ignored = export_modernbert(state, cfg, args.output, group=args.group)
    print(f"wrote {args.output}: {size / 1e6:.1f} MB")
    if ignored:
        print(f"not part of the encoder, skipped ({len(ignored)}): " + ", ".join(ignored[:8])
              + (" ..." if len(ignored) > 8 else ""))


def cmd_inspect(args):
    tensors = read_trv(args.input, dequantize=False)
    total = 0
    for name, (dtype, val) in tensors.items():
        shape = val[0].shape if dtype == DTYPE_Q8_ROWS else val.shape
        n = int(np.prod(shape))
        total += n
        print(f"{_DTYPE_NAMES[dtype]:8s} {str(tuple(shape)):>20s}  {name}")
    print(f"{len(tensors)} tensors, {total / 1e6:.2f}M parameters")


def main(argv=None):
    p = argparse.ArgumentParser(prog="trivi", description="Trivikrama model tools")
    p.add_argument("--version", action="version", version=f"trivikrama {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert", help="convert a .safetensors checkpoint to .trv")
    c.add_argument("input")
    c.add_argument("-o", "--output", required=True)
    c.add_argument(
        "--keep-f32",
        action="append",
        default=[],
        metavar="PATTERN",
        help="glob of 2-D tensor names to keep in float32 (repeatable)",
    )
    c.set_defaults(func=cmd_convert)

    e = sub.add_parser(
        "export-modernbert",
        help="export a ModernBERT encoder (local Hugging Face folder) for the C runtime",
    )
    e.add_argument("model_dir", help="folder with config.json and *.safetensors")
    e.add_argument("-o", "--output", required=True)
    e.add_argument("--group", type=int, default=None,
                   help="activation quantization group (default: gcd(head_dim, intermediate))")
    e.set_defaults(func=cmd_export_modernbert)

    i = sub.add_parser("inspect", help="list the tensors in a .trv file")
    i.add_argument("input")
    i.set_defaults(func=cmd_inspect)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
