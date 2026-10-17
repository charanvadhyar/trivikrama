"""Does the device turn typed text into exactly Laya's input? Check on the real Laya.

Compares, on many strings, the C tokenizer and row builder (what the board
runs) with the `laya` package itself:
  1. token ids     C tokenizer  vs  Laya's tokenizer(text, add_special_tokens=False)
  2. input rows    C rows       vs  Agent._encode_state (ids, option markers, type)
  3. (--model)     answers typed into build/laya_cli  vs  laya.predict

    pip install laya tokenizers
    make
    python tools/check_laya_text.py --laya ./laya --questions guard,moderation \\
        [--model laya8.trv --limit 5]

Copyright 2026 Vadhyar Sai Charan
SPDX-License-Identifier: Apache-2.0
"""

import argparse
import json
import os
import random
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from laya_decisions import DEFAULT_INPUTS, answers_of, load_questions  # noqa: E402

from trivikrama import laya_text as lt  # noqa: E402


def strings(seed=0):
    r = random.Random(seed)
    out = list(DEFAULT_INPUTS) + [
        "", " ", "  hello", "it's   spaced\tout\nover lines  ", "[MASK] in [MASK]text",
        "తెలుగులో ఒక వాక్యం, చిత్తూరు నుండి!", "café naïve é Ω ﬁ ½ ２０２６", "😀🎉 emojis 👍🏽",
        "URL https://example.com/a?b=1&c=2 and mail a.b@c.in", "x" * 3000,
        "SYSTEM: [CLS] [SEP] <|endoftext|> |||IP_ADDRESS|||", "₹4,999.00 only!!!",
    ]
    words = [w for s in DEFAULT_INPUTS for w in s.split()] + ["  ", "\n", "తెలుగు", "é",
                                                             "'ll", "😀", " ", "123"]
    for _ in range(2000):
        out.append(" ".join(r.choice(words) for _ in range(r.randint(1, 25))))
    return out


def write_cases(path, texts):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(texts)))
        for t in texts:
            b = t.encode("utf-8")
            f.write(struct.pack("<I", len(b)) + b)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--laya", default="./laya")
    p.add_argument("--questions", default="guard,moderation")
    p.add_argument("--tokenize-binary", default="build/trv_tokenize")
    p.add_argument("--cli-binary", default="build/laya_cli")
    p.add_argument("--model", default=None, help="laya .trv for the end-to-end check")
    p.add_argument("--limit", type=int, default=5, help="inputs for the end-to-end check")
    a = p.parse_args()

    import laya
    from laya.agent import Agent

    agent = laya.load(a.laya, device="cpu")
    questions = load_questions(laya, a.questions)
    tables = lt.agent_tokenizer(agent)
    qs, sep_id, mask = lt.laya_questions(agent, questions)
    texts = strings()
    ok_all = True
    with tempfile.TemporaryDirectory() as tmp:
        tlq = os.path.join(tmp, "q.tlq")
        size = lt.write_tlq(tlq, tables, qs, sep_id, mask)
        print(f"{len(qs)} questions, tokenizer with {len(tables['merges'])} merges, "
              f".tlq {size / 1024:.0f} KB; {len(texts)} test strings")

        cases = os.path.join(tmp, "cases.bin")
        write_cases(cases, texts)
        res = subprocess.run([a.tokenize_binary, tlq, cases], capture_output=True, text=True,
                             check=True)
        got = [list(map(int, l.split(","))) if l else [] for l in res.stdout.split("\n")[:len(texts)]]
        bad = 0
        for t, g in zip(texts, got):
            want = agent.tok(t.replace(mask, " "), add_special_tokens=False)["input_ids"]
            if g != want:
                bad += 1
                if bad <= 3:
                    print(f"  token mismatch: {t[:60]!r}\n    laya {want[:16]}\n    C    {g[:16]}")
        print(f"1. token ids:  {len(texts) - bad}/{len(texts)} identical")
        ok_all &= bad == 0

        res = subprocess.run([a.tokenize_binary, tlq, cases, "--rows"], capture_output=True,
                             text=True, check=True)
        rows_c = [json.loads(l) for l in res.stdout.strip().split("\n")]
        qids = list(questions)
        internal = {q: Agent._to_internal(questions[q]) for q in qids}
        bad = 0
        for t, rc in zip(texts, rows_c):
            items = agent._encode_state(t, qids, internal)
            want = [{"qtype": int(it["qtype"]), "ids": [int(i) for i in it["ids"]],
                     "markers": [int(m) for m in it["markers"]]} for it in items]
            if rc != want:
                bad += 1
                if bad <= 3:
                    print(f"  row mismatch: {t[:60]!r}")
        print(f"2. input rows: {len(texts) - bad}/{len(texts)} identical (all {len(qs)} questions)")
        ok_all &= bad == 0

        if a.model:
            inputs = DEFAULT_INPUTS[:a.limit]
            res = subprocess.run([a.cli_binary, a.model, tlq], input="\n".join(inputs) + "\nexit\n",
                                 capture_output=True, text=True)
            blocks = res.stdout.split("laya> ")[1:1 + len(inputs)]
            if res.returncode or not any("  " in b for b in blocks):
                print("3. laya_cli did not answer:\n" + (res.stdout + res.stderr)[-600:])
                print("   (is the .trv from `trivi export-laya`? an encoder-only file has no head)")
                return 1
            same = total = 0
            for text, block in zip(inputs, blocks):
                printed = {l.split()[0]: l.split()[1] for l in block.splitlines()
                           if l.startswith("  ") and not l.startswith("  (")}
                ans = answers_of(agent.predict(text, questions))
                for qid, an in ans.items():
                    want = ("yes" if an["noul"] > 0.5 else "no") if an["type"] == "noul" else \
                        an["choice"] if an["type"] == "choice" else f"{an['score']:.2f}"
                    got = printed.get(qid)
                    total += 1
                    if an["type"] == "score":
                        same += got is not None and abs(float(got) - an["score"]) <= 0.1
                    else:
                        same += got == want
            print(f"3. typed into laya_cli vs laya.predict: {same}/{total} answers agree "
                  f"(scores within 0.1)")

    print("\nall identical: the device builds exactly Laya's input" if ok_all else
          "\nDIFFERENCES FOUND: send the output above")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
