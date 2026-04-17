#!/usr/bin/env python3
"""
flatten_corpus.py — Convert corpus_dialogue.jsonl to byte-level training text

Split by project: one project held out as val, rest goes to train.
Output format uses ISAAC: / CC: role markers.

Usage:
  python flatten_corpus.py --val E--dev-tools-two3 --out ./corpus/
"""

import argparse
import json
from pathlib import Path


def flatten_turn(user, assistant):
    """Format one turn as bytes-ready text with ISAAC:/CC: markers."""
    # Normalize whitespace, strip leading/trailing
    u = user.strip()
    a = assistant.strip()
    return f"ISAAC: {u}\nCC: {a}\n\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input", default="./corpus/corpus_dialogue.jsonl")
    ap.add_argument("--val", default="E--dev-tools-two3",
                    help="Project name to hold out as val")
    ap.add_argument("--out", default="./corpus/")
    ap.add_argument("--tiny", type=int, default=200,
                    help="Also write a tiny overfit subset of this many turns")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    with open(in_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    train = [s for s in samples if s["project"] != args.val]
    val = [s for s in samples if s["project"] == args.val]

    print(f"Train: {len(train)} turns")
    print(f"Val:   {len(val)} turns ({args.val})")

    # Write train corpus
    train_path = out_dir / "corpus_train.txt"
    with open(train_path, "w", encoding="utf-8") as f:
        for s in train:
            f.write(flatten_turn(s["user"], s["assistant"]))
    train_bytes = train_path.stat().st_size
    print(f"  wrote {train_path} ({train_bytes/1024:.1f} KB)")

    # Write val corpus
    val_path = out_dir / "corpus_val.txt"
    with open(val_path, "w", encoding="utf-8") as f:
        for s in val:
            f.write(flatten_turn(s["user"], s["assistant"]))
    val_bytes = val_path.stat().st_size
    print(f"  wrote {val_path} ({val_bytes/1024:.1f} KB)")

    # Write tiny overfit subset — first N turns from train
    tiny = train[:args.tiny]
    tiny_path = out_dir / f"corpus_tiny{args.tiny}.txt"
    with open(tiny_path, "w", encoding="utf-8") as f:
        for s in tiny:
            f.write(flatten_turn(s["user"], s["assistant"]))
    tiny_bytes = tiny_path.stat().st_size
    print(f"  wrote {tiny_path} ({tiny_bytes/1024:.1f} KB, {args.tiny} turns)")

    # Write a small sample for eyeball inspection
    sample_path = out_dir / "corpus_sample.txt"
    with open(sample_path, "w", encoding="utf-8") as f:
        for s in train[:5]:
            f.write(flatten_turn(s["user"], s["assistant"]))
    print(f"  wrote {sample_path} (eyeball sample)")


if __name__ == "__main__":
    main()
