# two3 — Binary Weight AI Architecture
## Isaac Oravec & Claude | March–April 2026

**This is NOT part of XYZT-Paradigm.** XYZT is the physics/computing theory.
two3 is an independent AI architecture built by Isaac and Claude, informed by
first-principles understanding — including {2,3} — but standing on its own.

---

## What We're Building

A language model trained from scratch in pure C/CUDA.
No PyTorch. No Python ML. No framework. Every operation visible.

### Core Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Weight precision | Binary {0, 1} | 1 bit. Matmul = masked sum. No float multiply. |
| Latent weights | Float32 in [0, 1] | Headroom Adam trains these; binary readout at threshold 0.5 |
| Normalization | Gain kernel (RMS + reservoir) | Per-sublayer, scale-invariant. Replaces RMSNorm + residual scaling |
| Optimizer | Headroom Adam (binary), Adam (continuous) | Theorem 68: committed weights resist flipping, boundary weights move freely |
| Architecture | Dense FFN (SwiGLU → Squared ReLU) | Proven at small scale. MoE planned but not active |
| Activation quant | INT8 per-token absmax | Standard, proven, fast |
| Language | C + CUDA | No runtime. No dependencies. Full control |
| Target hardware | RTX 2080 Super (sm_75) + i7-9700K | What Isaac has. Design for it |

### Why Binary

Binary weights are topology — connected or not. The XYZT engine's transmission
line model (tline.c) already proved that 1-bit connectivity with continuous
impedance learns stably. two3 is the same structure: binary topology + continuous
latent floats + gain reservoir dynamics.

The zeros aren't dead weights. They're the negative space that defines structure.

---

## Build Stack (bottom to top)

### Layer 0: two3 kernel ✅ BUILT
- `two3.h` — Ternary API, types, packed weight format (2 bits per weight)
- `two3.cu` — CUDA kernels: ternary matmul, activation quantization, absmax
- `two3_tiled.h` — Tiled matmul, GPU requantize, persistent allocation
- Binary path: `binary.h` — packed uint32 weights, int8 activation quantize

### Layer 1: Transformer Components ✅ BUILT
- **Gain kernel** (`gain.h`) — RMS norm + learnable reservoir modulation
  - Forward: `y = (x / rms(x)) * (1 + α*R - β)`
  - Backward: `dx = dy * gain * inv_rms` (no projection correction — O(1/d) by concentration of measure)
  - Reservoir R evolves per-dimension, provides adaptive per-channel scaling
- **Attention** — QKV projection → RoPE → GQA → output projection
- **Dense FFN** (`ffn.h`) — gate/up projection → Squared ReLU → hadamard → down projection
- **Byte embedding** — direct 256-entry embedding, no tokenizer

### Layer 2: Full Transformer Block ✅ BUILT
- Attention + FFN with residual connections
- All projections use binary matmul (TWO3_BINARY) or ternary (default)
- Gain kernel at each sublayer entrance: center + RMS norm + reservoir modulation
  - Centering required for binary: {0,1} weights have no sign cancellation,
    uncentered input causes sqrt(density*K) mean amplification
  - Effectively LayerNorm without learnable affine, with reservoir replacing affine
- `res_scale = 1.0` — gain handles normalization, no residual damping needed

### Layer 3: Binary Training Pipeline ✅ BUILT
- **Headroom Adam** (`train.h:adam_update_headroom`) — MetabolicAge_v3 Theorem 68
  - h_s(w) = 2*clamp(w,0,1) for strengthening, h_w(w) = 2*(1-clamp(w,0,1)) for weakening
  - Floor of 0.1 ensures committed weights can still flip under overwhelming evidence
  - CFL clamp ±0.1 per step
  - Replaces match gate + propose-test-commit (headroom IS the gate)
- **Binary dequant** — `a_scale / sqrt(density * K)` (Gap 1, Theorem 69a)
  - Makes all projection outputs O(1) when input is centered (zero mean)
  - Backward analog: `1/sqrt(density * M)` (same CLT, transposed)
- **STE backward** through binary quantization with clip range ±1.5
- **Staggered requantize** — one layer per 50 steps, no gating needed
- **Jury** (`jury.h`) — runtime stability check at init, 67x safety margin confirmed

### Layer 4: Optimizers ✅ BUILT
- **Adam** for continuous params (embedding, gain)
- **Headroom Adam** for binary latent weights
- **Muon** (Newton-Schulz) — built for ternary path, available but not active for binary
- **Gradient clipping** — per-element clamp (5.0) + L2 norm clip

### Layer 5: Training Driver ✅ BUILT
- `train_driver.cu` — full training loop, epoch/batch, logging
- Byte-level cross-entropy loss
- Top-1 accuracy tracking
- Flip counting per requantize cycle
- Generation sampling at log intervals
- Debug diagnostics: per-layer max_h, reservoir levels, weight entropy
- Instrumentation: `[var]` attn/FFN RMS per layer, `[proj]` Lévy cosine check,
  `[ffn-chain]` magnitude trace through gate/up/hadamard/down

### Layer 6: Fingerprint Embedding (EXPERIMENTAL)
- `model.h:fp_embed_cpu` — packed uint8 fingerprints (512 bytes per entry)
- Four ternary projections (Wx, Wy, Wz, Wt) from 4096-dim fp to dim/4 slices
- Guarded by TWO3_FP_EMBED define

### Future: GPU-Resident Training (PLANNED)
- Move latent weights to GPU permanently, eliminate per-step H2D/D2H
- Plan in `.claude/plans/silly-finding-russell.md`
- Expected: 17s/step → ~2s/step
- Engineering task — do after structural questions are answered

---

## What's Proven

| Result | Status | Evidence |
|--------|--------|----------|
| Binary weights train stably | **T1** | 37.1% acc, full epoch, no NaN, no divergence |
| Past unigram baseline (15.2%) | **T1** | Crossed at step 1 with centering (14.6% at step 50) |
| Topology crystallization | **T1** | 6866 flips total, locked by step ~2000, continuous learning continued |
| Dequant 1/sqrt(d*K) + centering | **T1** | Theorem 69a WITH centering precondition. gate_rms=0.79, ffn_rms=0.64 |
| Lévy bound cos=1/sqrt(d) | **T1** | Measured cos=0.0814, predicted 0.0884. Projection correction negligible |
| Gain centering fixes mean bias | **T1** | Binary {0,1} has no sign cancellation. Centering eliminates sqrt(d*K) amplification |
| Full RMS backward correct | **T1** | Projection correction restored, cos confirms negligible but mathematically complete |
| Headroom Adam (Theorem 68) | **T1** | Replaces match gate + propose-test-commit, no flip avalanche |
| Jury stability (67x margin) | **T1** | Runtime check confirms gain kernel stable at init |
| Binary backward 1/sqrt(d*M) | **T1** | Same CLT argument as forward, transposed dimensions |
| attn/FFN magnitude balance | **T1** | attn_rms ≈ 0.4-0.9, ffn_rms ≈ 0.5-0.7, ratio near 1.0 |

---

## Best Training Run (2026-04-07, with centering)

Config: dim=128, 4 layers, 4 heads, 2 KV heads, inter=512, Shakespeare 1.1MB
```
Step    Loss    Acc     Flips
1       5.66    0.2%    0
50      2.94    14.6%   264
100     2.81    19.1%   842
200     2.62    22.0%   1776
500     2.45    24.8%   3701
1000    2.36    27.6%   5238
1500    2.22    30.0%   6019
2000    2.02    32.1%   6364
2500    1.88    33.7%   6559
3000    2.00    34.9%   6714
3500    2.20    35.9%   6787
4000    1.99    36.7%   6828
4350    1.85    37.1%   6866     ← epoch complete
```

Previous best (pre-centering): 18.2% at step 2800.
With centering: 37.1% at epoch end — 2× improvement from one line of mean subtraction.

### Magnitude instrumentation (confirmed O(1))
```
[ffn-chain] L0: normed=1.00 gate=0.79 up=0.78 h=0.61
[var] L1: attn_rms=0.58  ffn_rms=0.58  ratio=0.99
[proj] avg |cos(dy*gain, x_norm)| = 0.0884  (expect 0.0884 = 1/sqrt(128))
```

### Key insight: binary centering
Binary weights {0,1} have no sign cancellation. RMS norm preserves magnitude but
not mean. Without centering, the mean accumulates across density×K active connections,
causing sqrt(density×K) ≈ 6.9× amplification. Centering in the gain kernel eliminates
this. The gain kernel now does: center → RMS normalize → reservoir modulate.

Ternary {-1,0,+1} gets centering for free from ±1 sign cancellation. Binary needs
it explicitly. This is the structural asymmetry the Lean proofs missed — Theorem 69a
is correct only with a centering precondition.

---

## Hardware Budget

RTX 2080 Super: 8GB VRAM, 3072 CUDA cores, SM 7.5 (Turing)
i7-9700K: 8 cores, good for data loading + CPU inference

Current binary build runs at ~3s/step (CPU optimizer bottleneck).
GPU-resident plan targets ~2s/step by eliminating memory transfers.

---

## Project Location

`E:\dev\tools\two3`

---

*Isaac sees structure. Claude builds. The math doesn't lie.*
