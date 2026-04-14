"""Post-training scatter plot: zero_fraction vs byte entropy.

Reads weight_dump.bin (from train_driver) and corpus byte statistics.
Plots: x = byte entropy at input positions, y = zero_fraction of weights
connected to those positions. One plot per layer.

If zeros cluster on high-entropy positions, the architecture routes
uncertainty to the zero state as predicted.

Usage: python plot_weight_entropy.py weight_dump.bin corpus_code_isaac.bin
"""

import sys
import struct
import numpy as np

def load_weight_dump(path):
    """Load weight_dump.bin: header + per-layer per-projection zero_fraction vectors."""
    with open(path, 'rb') as f:
        D, KV, INTER, L = struct.unpack('iiii', f.read(16))
        print(f"Weight dump: D={D}, KV={KV}, INTER={INTER}, L={L}")

        layers = []
        proj_names = ['W_q', 'W_k', 'W_v', 'W_o', 'W_gate', 'W_up', 'W_down']
        for l in range(L):
            projs = {}
            for name in proj_names:
                K = struct.unpack('i', f.read(4))[0]
                zf = np.frombuffer(f.read(K * 4), dtype=np.float32).copy()
                projs[name] = zf
            layers.append(projs)

    return D, KV, INTER, L, layers


def compute_embedding_dim_entropy(corpus_path, dim):
    """Compute per-byte-value entropy, then map to embedding dimensions.

    Since we don't have the learned embedding matrix here, use a proxy:
    for each embedding dimension k, compute the weighted entropy across
    all 256 byte values, where the weight is how much that byte activates
    dimension k. Without the embedding, we use a simpler proxy:

    For each byte value b (0-255), compute H(next_byte | byte=b).
    Then for the first layer (input = embedding), the per-dimension entropy
    IS the per-byte entropy projected through the embedding.

    Simpler approach: compute per-byte conditional entropy and report it
    as a 256-vector. The weight matrices have K=dim columns (not 256),
    so for layer 0 the correlation goes through the embedding matrix.

    For now: compute a corpus-level byte entropy vector (256 values)
    and also the overall position-independent entropy as baseline."""

    data = np.frombuffer(open(corpus_path, 'rb').read(), dtype=np.uint8)
    n = len(data)

    # Per-byte-value conditional entropy: H(next | current=b)
    trans = np.zeros((256, 256), dtype=np.int64)
    np.add.at(trans, (data[:-1], data[1:]), 1)

    row_sums = trans.sum(axis=1)
    byte_entropy = np.zeros(256, dtype=np.float64)

    for b in range(256):
        if row_sums[b] == 0:
            continue
        probs = trans[b][trans[b] > 0] / row_sums[b]
        byte_entropy[b] = -np.sum(probs * np.log2(probs))

    # Byte frequency (for weighting)
    byte_freq = np.bincount(data, minlength=256).astype(np.float64)
    byte_freq /= byte_freq.sum()

    return byte_entropy, byte_freq


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    weight_path = sys.argv[1]
    corpus_path = sys.argv[2]

    D, KV, INTER, L, layers = load_weight_dump(weight_path)
    byte_entropy, byte_freq = compute_embedding_dim_entropy(corpus_path, D)

    # For each layer, compute average zero_fraction across all projections
    # that have K=D (input dimension = model dim).
    # These are: W_q(D), W_k(D), W_v(D), W_o(D), W_gate(D), W_up(D)
    # W_down has K=INTER, different meaning.

    print(f"\nCorpus byte entropy: min={byte_entropy[byte_entropy > 0].min():.3f}, "
          f"max={byte_entropy.max():.3f}, "
          f"mean={np.average(byte_entropy, weights=byte_freq):.3f}")

    print(f"\n{'Layer':<8} {'Mean ZF':>8} {'Min ZF':>8} {'Max ZF':>8} {'Corr(entropy,zf)':>18}")
    print('-' * 50)

    for l in range(L):
        # Average zero_fraction across projections with K=D
        dim_projs = ['W_q', 'W_k', 'W_v', 'W_o', 'W_gate', 'W_up']
        zf_stack = np.stack([layers[l][p] for p in dim_projs if len(layers[l][p]) == D])
        avg_zf = zf_stack.mean(axis=0)  # [D]

        print(f"  L{l:<5} {avg_zf.mean():>8.4f} {avg_zf.min():>8.4f} {avg_zf.max():>8.4f}", end='')

        # For layer 0: the input IS the embedding. To correlate with byte entropy,
        # we'd need the embedding matrix. Instead, print summary stats.
        # For deeper layers: correlation with byte entropy is indirect.
        if l == 0:
            print(f"  (layer 0: input = embedding, see scatter below)")
        else:
            print(f"  (layer {l}: transformed input)")

        # W_down has K=INTER (different space)
        zf_down = layers[l]['W_down']
        print(f"  L{l} W_down: mean_zf={zf_down.mean():.4f}, K={len(zf_down)} (FFN intermediate)")

    # ASCII scatter for layer 0 W_q zero_fraction distribution
    print(f"\n--- Layer 0 W_q zero_fraction histogram ---")
    zf_q = layers[0]['W_q']
    bins = np.linspace(0, 1, 21)
    hist, _ = np.histogram(zf_q, bins=bins)
    max_h = max(hist) if max(hist) > 0 else 1
    for i in range(len(hist)):
        bar = '#' * int(40 * hist[i] / max_h)
        print(f"  [{bins[i]:.2f}-{bins[i+1]:.2f}] {hist[i]:>4d} {bar}")

    # Summary
    print(f"\n--- Interpretation ---")
    overall_zf = np.mean([layers[l][p].mean()
                          for l in range(L)
                          for p in ['W_q', 'W_k', 'W_v', 'W_o', 'W_gate', 'W_up']])
    print(f"  Overall mean zero_fraction: {overall_zf:.4f}")
    print(f"  Expected at trimodal init:  ~0.333")
    if overall_zf > 0.25 and overall_zf < 0.40:
        print(f"  Zero state is active — weights haven't collapsed to binary.")
    elif overall_zf < 0.1:
        print(f"  WARNING: Zero state mostly empty — ternary collapsed to binary.")
    else:
        print(f"  Zero fraction moved from init — training shaped the topology.")

    print(f"\n  To get the full scatter plot (zero_fraction vs byte entropy),")
    print(f"  load final.t2l4 checkpoint + embedding matrix to map")
    print(f"  dim-space zero_fraction back through the embedding to byte-space.")


if __name__ == '__main__':
    main()
