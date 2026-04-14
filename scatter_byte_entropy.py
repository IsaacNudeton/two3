"""Scatter plot: zero_fraction vs byte entropy, mapped through embedding.

Loads the checkpoint embedding matrix (256 × dim) and weight_dump.bin.
For each byte value b (0-255):
  1. Get its embedding vector e_b = embed[b, :]  (dim-dimensional)
  2. Compute "effective zero fraction" = dot(|e_b|, zero_fraction) / sum(|e_b|)
     This weights each dimension's zero_fraction by how active that dimension
     is for byte b.
  3. Compute conditional entropy H(next | current=b) from corpus statistics.

Then scatter: x = byte entropy, y = effective zero fraction.
If zeros cluster on high-entropy bytes, positive correlation.

Usage: python scatter_byte_entropy.py final.t2l4 weight_dump.bin corpus/corpus_code_isaac.bin
"""

import sys
import struct
import numpy as np


def load_checkpoint_embedding(ckpt_path):
    """Load embedding matrix from .t2l4 checkpoint."""
    with open(ckpt_path, 'rb') as f:
        magic = struct.unpack('I', f.read(4))[0]
        assert magic == 0x54324C34, f"Bad magic: {hex(magic)}"

        # ModelConfig: 7 ints + 1 float = 32 bytes
        cfg = struct.unpack('iiiiiiif', f.read(32))
        dim, n_heads, n_kv_heads, head_dim, intermediate, n_layers, max_seq, rope_theta = cfg
        step = struct.unpack('i', f.read(4))[0]

        print(f"Checkpoint: dim={dim}, layers={n_layers}, step={step}")

        # Embedding: 256 × dim floats
        embed = np.frombuffer(f.read(256 * dim * 4), dtype=np.float32).reshape(256, dim).copy()

    return embed, dim


def load_weight_dump(path):
    """Load zero_fraction vectors from weight_dump.bin."""
    with open(path, 'rb') as f:
        D, KV, INTER, L = struct.unpack('iiii', f.read(16))
        proj_names = ['W_q', 'W_k', 'W_v', 'W_o', 'W_gate', 'W_up', 'W_down']
        layers = []
        for l in range(L):
            projs = {}
            for name in proj_names:
                K = struct.unpack('i', f.read(4))[0]
                zf = np.frombuffer(f.read(K * 4), dtype=np.float32).copy()
                projs[name] = zf
            layers.append(projs)
    return D, KV, INTER, L, layers


def compute_byte_entropy(corpus_path):
    """Per-byte-value conditional entropy H(next | current=b)."""
    data = np.frombuffer(open(corpus_path, 'rb').read(), dtype=np.uint8)

    trans = np.zeros((256, 256), dtype=np.int64)
    np.add.at(trans, (data[:-1], data[1:]), 1)

    row_sums = trans.sum(axis=1)
    byte_freq = np.bincount(data, minlength=256).astype(np.float64)

    entropy = np.zeros(256, dtype=np.float64)
    for b in range(256):
        if row_sums[b] == 0:
            continue
        probs = trans[b][trans[b] > 0] / row_sums[b]
        entropy[b] = -np.sum(probs * np.log2(probs))

    return entropy, byte_freq


def classify_byte(b):
    """Classify a byte as structure(0), keyword-likely(1), or name(2)."""
    c = chr(b) if b < 128 else ''
    if c in '{}()[];,\n\t\r <>:=+-*/%&|^~!?.#"\'\\ ':
        return 0  # structure
    if c.isalpha() or c == '_':
        return 2  # could be keyword or name — mark as name, refine below
    if c.isdigit():
        return 2  # literal
    return 2  # other


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    ckpt_path = sys.argv[1]
    weight_path = sys.argv[2]
    corpus_path = sys.argv[3]

    embed, dim = load_checkpoint_embedding(ckpt_path)
    D, KV, INTER, L, layers = load_weight_dump(weight_path)
    entropy, byte_freq = compute_byte_entropy(corpus_path)

    assert dim == D, f"dim mismatch: checkpoint={dim}, weights={D}"

    # Average zero_fraction across all D-input projections at layer 0
    dim_projs = ['W_q', 'W_k', 'W_v', 'W_o', 'W_gate', 'W_up']
    zf_stack = np.stack([layers[0][p] for p in dim_projs])
    zf_avg = zf_stack.mean(axis=0)  # [dim]

    # Map through embedding: for each byte b, compute weighted zero_fraction
    # effective_zf[b] = sum(|embed[b,k]| * zf[k]) / sum(|embed[b,k]|)
    abs_embed = np.abs(embed)  # [256, dim]
    embed_norm = abs_embed.sum(axis=1, keepdims=True)  # [256, 1]
    embed_norm = np.maximum(embed_norm, 1e-8)  # avoid div-by-zero
    effective_zf = (abs_embed @ zf_avg) / embed_norm.squeeze()  # [256]

    # Filter to bytes that actually appear in corpus
    active = byte_freq > 100  # at least 100 occurrences
    n_active = active.sum()

    print(f"\nActive bytes (freq > 100): {n_active}")
    print(f"Byte entropy range: [{entropy[active].min():.3f}, {entropy[active].max():.3f}]")
    print(f"Effective ZF range: [{effective_zf[active].min():.4f}, {effective_zf[active].max():.4f}]")

    # Correlation
    from numpy import corrcoef
    r = corrcoef(entropy[active], effective_zf[active])[0, 1]
    print(f"\n*** Pearson correlation (byte_entropy vs effective_zero_fraction): r = {r:+.4f} ***")
    if r > 0.3:
        print("    POSITIVE — zeros cluster on high-entropy (name) bytes. Architecture works.")
    elif r < -0.3:
        print("    NEGATIVE — zeros cluster on low-entropy (structure) bytes. Inverted from prediction.")
    else:
        print("    WEAK — no clear relationship. Zeros distributed randomly across byte types.")

    # ASCII scatter plot
    print(f"\n--- Scatter: byte entropy (x) vs effective zero_fraction (y) ---")
    print(f"    Each point is one byte value. Labels: S=structure, N=name/keyword")

    # Bin into grid
    W, H = 60, 20
    x_min, x_max = 0.0, entropy[active].max() * 1.05
    y_min, y_max = effective_zf[active].min() * 0.95, effective_zf[active].max() * 1.05
    grid = [[' '] * W for _ in range(H)]

    for b in range(256):
        if not active[b]:
            continue
        xi = int((entropy[b] - x_min) / (x_max - x_min) * (W - 1))
        yi = int((effective_zf[b] - y_min) / (y_max - y_min) * (H - 1))
        xi = max(0, min(W - 1, xi))
        yi = max(0, min(H - 1, yi))
        tier = classify_byte(b)
        ch = 'S' if tier == 0 else 'N'
        grid[H - 1 - yi][xi] = ch

    # Print with axes
    print(f"  ZF {effective_zf[active].max():.3f} |", end='')
    for row in grid[:1]:
        print(''.join(row), '|')
    for row in grid[1:-1]:
        print('         |' + ''.join(row) + '|')
    print(f"  ZF {effective_zf[active].min():.3f} |", end='')
    for row in grid[-1:]:
        print(''.join(row), '|')
    print(f"         0.0{'=' * (W - 20)}entropy{'=' * 8}{entropy[active].max():.1f}")

    # Top-10 highest and lowest effective ZF bytes
    print(f"\n--- Top 10 HIGHEST effective zero_fraction (most uncertain) ---")
    sorted_idx = np.argsort(effective_zf)[::-1]
    count = 0
    for b in sorted_idx:
        if not active[b]:
            continue
        c = chr(b) if 32 <= b < 127 else f'0x{b:02x}'
        tier = ['struct', 'kw', 'name'][classify_byte(b)]
        print(f"  byte={c:>6s}  entropy={entropy[b]:.3f}  eff_zf={effective_zf[b]:.4f}  freq={byte_freq[b]:.0f}  tier={tier}")
        count += 1
        if count >= 10:
            break

    print(f"\n--- Top 10 LOWEST effective zero_fraction (most committed) ---")
    sorted_idx = np.argsort(effective_zf)
    count = 0
    for b in sorted_idx:
        if not active[b]:
            continue
        c = chr(b) if 32 <= b < 127 else f'0x{b:02x}'
        tier = ['struct', 'kw', 'name'][classify_byte(b)]
        print(f"  byte={c:>6s}  entropy={entropy[b]:.3f}  eff_zf={effective_zf[b]:.4f}  freq={byte_freq[b]:.0f}  tier={tier}")
        count += 1
        if count >= 10:
            break

    # Per-tier summary
    print(f"\n--- Per-tier effective zero_fraction ---")
    for tier_name, tier_id in [('structure', 0), ('name', 2)]:
        mask = active.copy()
        for b in range(256):
            if classify_byte(b) != tier_id:
                mask[b] = False
        if mask.sum() == 0:
            continue
        mean_e = entropy[mask].mean()
        mean_zf = effective_zf[mask].mean()
        print(f"  {tier_name:>12s}: n={mask.sum():>3d}  mean_entropy={mean_e:.3f}  mean_eff_zf={mean_zf:.4f}")


if __name__ == '__main__':
    main()
