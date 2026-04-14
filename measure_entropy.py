"""Measure per-tier byte entropy in a code corpus.

Classifies bytes into structure/keyword/name tiers and computes
conditional entropy H(next_byte | current_byte) per tier.
If the three tiers show clearly separated entropy, the ternary
experiment is well-posed.

Usage: python measure_entropy.py <corpus_file>
       python measure_entropy.py --build <out_file> <dir1> [dir2] ...
"""

import sys
import os
import math
import numpy as np
from collections import defaultdict

# --- Byte classification ---

STRUCTURE_BYTES = set(ord(c) for c in '{}()[];,\n\t\r <>:')
SPACE_BYTE = ord(' ')

# C keywords — we tag bytes that appear WITHIN these tokens
C_KEYWORDS = {
    'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do',
    'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if',
    'inline', 'int', 'long', 'register', 'return', 'short', 'signed',
    'sizeof', 'static', 'struct', 'switch', 'typedef', 'union', 'unsigned',
    'void', 'volatile', 'while',
    # CUDA/common extensions
    '__global__', '__device__', '__host__', '__shared__', '#include',
    '#define', '#ifdef', '#ifndef', '#endif', '#else', '#pragma',
    'NULL', 'true', 'false', 'bool', 'size_t', 'uint8_t', 'int8_t',
    'uint16_t', 'int16_t', 'uint32_t', 'int32_t', 'uint64_t', 'int64_t',
}

def classify_tokens(data):
    """For each byte position, assign: 0=structure, 1=keyword, 2=name. Vectorized."""
    arr = np.frombuffer(data, dtype=np.uint8)
    n = len(arr)
    labels = np.zeros(n, dtype=np.uint8)  # default 0 = structure

    # Build lookup tables
    is_struct = np.zeros(256, dtype=bool)
    for b in STRUCTURE_BYTES:
        is_struct[b] = True
    is_struct[SPACE_BYTE] = True
    for b in (ord('='), ord('+'), ord('-'), ord('*'), ord('/'),
              ord('%'), ord('&'), ord('|'), ord('^'), ord('~'),
              ord('!'), ord('?'), ord('.'), ord('"'), ord("'"), ord('\\')):
        is_struct[b] = True

    is_alpha_under = np.zeros(256, dtype=bool)
    for c in range(ord('a'), ord('z')+1): is_alpha_under[c] = True
    for c in range(ord('A'), ord('Z')+1): is_alpha_under[c] = True
    is_alpha_under[ord('_')] = True

    is_alnum_under = is_alpha_under.copy()
    for c in range(ord('0'), ord('9')+1): is_alnum_under[c] = True

    is_digit = np.zeros(256, dtype=bool)
    for c in range(ord('0'), ord('9')+1): is_digit[c] = True

    # Mark structure bytes (fast vectorized)
    struct_mask = is_struct[arr]
    labels[struct_mask] = 0  # already 0, but explicit

    # Mark digits as name
    labels[is_digit[arr]] = 2

    # Now handle identifiers/keywords — must scan sequentially for token boundaries
    # but only over non-struct, non-digit runs
    i = 0
    while i < n:
        if struct_mask[i] or is_digit[arr[i]]:
            i += 1
            continue
        if is_alpha_under[arr[i]] or arr[i] == ord('#'):
            start = i
            i += 1
            while i < n and is_alnum_under[arr[i]]:
                i += 1
            token = data[start:i].decode('ascii', errors='replace')
            tier = 1 if token in C_KEYWORDS else 2
            labels[start:i] = tier
        else:
            labels[i] = 2  # unknown non-struct -> name
            i += 1

    return labels


def compute_conditional_entropy(data, labels, tier):
    """H(next_byte | current_byte) for positions labeled as `tier`. Vectorized."""
    arr = np.frombuffer(data, dtype=np.uint8)
    lab = labels
    mask = (lab[:-1] == tier)
    if not np.any(mask):
        return 0.0, 0

    cur = arr[:-1][mask]
    nxt = arr[1:][mask]
    total = len(cur)

    # Build 256x256 transition matrix
    trans = np.zeros((256, 256), dtype=np.int64)
    np.add.at(trans, (cur, nxt), 1)

    row_sums = trans.sum(axis=1)
    nonzero = row_sums > 0

    # Per-row entropy, weighted by row frequency
    h = 0.0
    for bval in np.where(nonzero)[0]:
        row = trans[bval]
        rs = row_sums[bval]
        probs = row[row > 0] / rs
        h_row = -np.sum(probs * np.log2(probs))
        h += (rs / total) * h_row

    return h, total


def compute_marginal_entropy(data, labels, tier):
    """H(byte) for positions labeled as `tier`. Vectorized."""
    arr = np.frombuffer(data, dtype=np.uint8)
    mask = (labels == tier)
    subset = arr[mask]
    total = len(subset)
    if total == 0:
        return 0.0, 0, 0

    hist = np.bincount(subset, minlength=256)
    nonzero = hist[hist > 0]
    probs = nonzero / total
    h = -np.sum(probs * np.log2(probs))

    return h, total, int(np.sum(hist > 0))


def build_corpus(out_path, dirs):
    """Concatenate all .c/.cu/.h files from dirs into one file, 0xFF separated."""
    SEP = b'\xff'
    count = 0
    total = 0
    with open(out_path, 'wb') as f:
        for d in dirs:
            for root, _, files in os.walk(d):
                for fn in sorted(files):
                    if fn.endswith(('.c', '.cu', '.h')):
                        path = os.path.join(root, fn)
                        try:
                            content = open(path, 'rb').read()
                        except (OSError, PermissionError):
                            continue
                        if count > 0:
                            f.write(SEP)
                        f.write(content)
                        total += len(content)
                        count += 1
    print(f"Built corpus: {count} files, {total:,} bytes -> {out_path}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    if sys.argv[1] == '--build':
        if len(sys.argv) < 4:
            print("Usage: python measure_entropy.py --build <out_file> <dir1> [dir2] ...")
            sys.exit(1)
        build_corpus(sys.argv[2], sys.argv[3:])
        sys.exit(0)

    path = sys.argv[1]
    data = open(path, 'rb').read()
    print(f"Corpus: {path} ({len(data):,} bytes)")

    labels = classify_tokens(data)

    tier_names = ['structure', 'keyword', 'name']
    print(f"\n{'Tier':<12} {'Count':>10} {'Frac':>7} {'Unique':>7} {'H(byte)':>9} {'H(next|cur)':>12}")
    print('-' * 62)

    for tier in range(3):
        h_marginal, count, unique = compute_marginal_entropy(data, labels, tier)
        h_cond, _ = compute_conditional_entropy(data, labels, tier)
        frac = count / len(data) if len(data) > 0 else 0
        print(f"{tier_names[tier]:<12} {count:>10,} {frac:>7.1%} {unique:>7} {h_marginal:>9.3f} {h_cond:>12.3f}")

    print(f"\nTarget: structure < 2 bits, keyword 3-4 bits, name > 5 bits (marginal)")
    print(f"If tiers overlap, name entropy is too low — need foreign code mixed in.")


if __name__ == '__main__':
    main()
