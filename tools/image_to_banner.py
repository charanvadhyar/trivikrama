"""Turn a picture into a terminal banner for laya_cli (Unicode blocks or Braille).

    pip install pillow
    python tools/image_to_banner.py painting.jpg --style blocks --cols 100 --preview
    python tools/image_to_banner.py painting.jpg --style braille --cols 100 \\
        -o examples/laya_cli/trivi_picture.h

Styles (all print in any modern terminal: Windows Terminal, VS Code, PuTTY,
idf.py monitor, Linux/Mac terminals; not the Arduino IDE monitor):
  blocks   each character is 2 pixels stacked (the top half block, foreground
           = upper pixel, background = lower pixel), in colour: closest to
           the photo. --color 256 (default) or truecolor
  braille  each character is a 2 x 4 dot grid, one colour: 8 dots per
           character, the finest detail, small output
  ascii    classic density characters, like image-to-ascii sites

Copyright 2026 Vadhyar Sai Charan
SPDX-License-Identifier: Apache-2.0
"""

import argparse
import sys

ASCII_RAMP = " .:-=+*#%@"


def load(path, cols, rows_per_char):
    """Resize so pixels come out square: a character cell is ~twice as tall as wide."""
    from PIL import Image, ImageEnhance, ImageOps

    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    im = ImageEnhance.Contrast(im).enhance(1.15)
    w = cols * (2 if rows_per_char == 4 else 1)              # braille: 2 dots per column
    h = w * im.height / im.width * (0.5 if rows_per_char == 1 else 1.0)
    h = max(rows_per_char, int(round(h / rows_per_char)) * rows_per_char)
    return im.resize((w, h), Image.LANCZOS)


def rgb_to_256(r, g, b):
    def q(v):
        return 0 if v < 48 else 1 if v < 115 else (v - 35) // 40
    cube = 16 + 36 * q(r) + 6 * q(g) + q(b)
    gray = sum((r, g, b)) / 3
    gi = 232 + min(23, max(0, int(round((gray - 8) / 10))))
    def back(i):
        if i >= 232:
            v = 8 + 10 * (i - 232)
            return v, v, v
        i -= 16
        lv = [0, 95, 135, 175, 215, 255]
        return lv[i // 36], lv[(i // 6) % 6], lv[i % 6]
    d = lambda i: sum((a - b) ** 2 for a, b in zip(back(i), (r, g, b)))  # noqa: E731
    return cube if d(cube) <= d(gi) else gi


# 256-colour greens from black to pale green, for --tint green
GREEN_256 = [16, 22, 28, 34, 40, 46, 83, 120, 157, 194, 231]


def _xterm_rgb(i):
    if i >= 232:
        v = 8 + 10 * (i - 232)
        return v, v, v
    i -= 16
    lv = [0, 95, 135, 175, 215, 255]
    return lv[i // 36], lv[(i // 6) % 6], lv[i % 6]


def green_blocks(im, color):
    """Brightness of the picture in shades of green, Floyd-Steinberg dithered."""
    import numpy as np

    g = np.asarray(im.convert("L"), dtype=np.float64)
    lum = [0.2126 * r + 0.7152 * gg + 0.0722 * b for r, gg, b in map(_xterm_rgb, GREEN_256)]
    # stretch the picture onto the ramp's luminance range
    g = (g - g.min()) / max(1e-6, g.max() - g.min()) * lum[-1]
    idx = np.zeros(g.shape, dtype=int)
    if color == "truecolor":
        idx = None
    else:
        lum_a = np.array(lum)
        for y in range(g.shape[0]):
            for x in range(g.shape[1]):
                k = int(np.abs(lum_a - g[y, x]).argmin())
                idx[y, x] = k
                err = g[y, x] - lum_a[k]
                if x + 1 < g.shape[1]:
                    g[y, x + 1] += err * 7 / 16
                if y + 1 < g.shape[0]:
                    if x:
                        g[y + 1, x - 1] += err * 3 / 16
                    g[y + 1, x] += err * 5 / 16
                    if x + 1 < g.shape[1]:
                        g[y + 1, x + 1] += err * 1 / 16
    lines = []
    for y in range(0, g.shape[0] - 1, 2):
        out, last = [], None
        for x in range(g.shape[1]):
            if idx is None:
                t, b = (min(255, max(0, v)) / lum[-1] for v in (g[y, x], g[y + 1, x]))
                code = "\033[38;2;%d;%d;%dm\033[48;2;%d;%d;%dm" % (
                    40 * t, 255 * t, 90 * t, 40 * b, 255 * b, 90 * b)
            else:
                code = "\033[38;5;%dm\033[48;5;%dm" % (GREEN_256[idx[y, x]], GREEN_256[idx[y + 1, x]])
            if code != last:
                out.append(code)
                last = code
            out.append("\u2580")
        lines.append("".join(out) + "\033[0m")
    return lines


def blocks(im, color):
    px = im.load()
    lines = []
    for y in range(0, im.height, 2):
        out, last = [], None
        for x in range(im.width):
            top, bot = px[x, y], px[x, y + 1]
            if color == "truecolor":
                code = "\033[38;2;%d;%d;%dm\033[48;2;%d;%d;%dm" % (*top, *bot)
            else:
                code = "\033[38;5;%dm\033[48;5;%dm" % (rgb_to_256(*top), rgb_to_256(*bot))
            if code != last:
                out.append(code)
                last = code
            out.append("▀")
        lines.append("".join(out) + "\033[0m")
    return lines


def braille(im, invert=False):
    from PIL import Image

    g = im.convert("L").convert("1", dither=Image.FLOYDSTEINBERG)
    px = g.load()
    bits = [(0, 0, 0x01), (0, 1, 0x02), (0, 2, 0x04), (1, 0, 0x08), (1, 1, 0x10), (1, 2, 0x20),
            (0, 3, 0x40), (1, 3, 0x80)]
    lines = []
    for y in range(0, g.height, 4):
        row = []
        for x in range(0, g.width, 2):
            v = 0
            for dx, dy, b in bits:
                on = px[x + dx, y + dy] != 0
                if on != invert:
                    v |= b
            row.append(chr(0x2800 + v))
        lines.append("".join(row).rstrip("⠀"))
    return lines


def ascii_art(im):
    g = im.convert("L")
    px = g.load()
    return ["".join(ASCII_RAMP[min(9, px[x, y] * 10 // 256)] for x in range(g.width)).rstrip()
            for y in range(g.height)]


def c_header(lines, name):
    def lit(s):
        return '"' + "".join(c if 32 <= ord(c) < 127 and c not in '"\\' else
                             "".join("\\%03o" % b for b in c.encode("utf-8")) for c in s) + '"'
    body = ",\n".join("    " + lit(l) for l in lines)
    return (f"/* generated by tools/image_to_banner.py */\n"
            f"static const char *const {name}[{len(lines)}] = {{\n{body}\n}};\n")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("image")
    p.add_argument("--style", choices=["blocks", "braille", "ascii"], default="blocks")
    p.add_argument("--cols", type=int, default=100, help="width in characters")
    p.add_argument("--color", choices=["256", "truecolor"], default="256",
                   help="blocks only: 256 colours (works everywhere) or 24-bit")
    p.add_argument("--tint", choices=["none", "green"], default="none",
                   help="blocks: green shades instead of the picture's colours")
    p.add_argument("--invert", action="store_true", help="braille: dots for dark parts")
    p.add_argument("--name", default="trivi_picture_custom", help="C array name")
    p.add_argument("-o", "--output", help="write a C header for laya_cli")
    p.add_argument("--preview", action="store_true", help="print it here")
    a = p.parse_args()

    if a.style == "blocks":
        im = load(a.image, a.cols, 2)
        lines = green_blocks(im, a.color) if a.tint == "green" else blocks(im, a.color)
    elif a.style == "braille":
        lines = braille(load(a.image, a.cols, 4), a.invert)
    else:
        lines = ascii_art(load(a.image, a.cols, 1))
    if a.preview or not a.output:
        green = a.style != "blocks"
        print("\n".join(("\033[92m" + l + "\033[0m") if green else l for l in lines))
    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            f.write(c_header(lines, a.name))
        size = sum(len(l.encode("utf-8")) + 1 for l in lines)
        print(f"wrote {a.output}: {len(lines)} lines, {size / 1024:.1f} KB "
              f"({size * 10 / 115200:.1f} s over a 115200-baud UART)", file=sys.stderr)


if __name__ == "__main__":
    main()
