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
- Gain kernel at each sublayer entrance (replaces RMSNorm)
- `res_scale = 1.0` — gain handles normalization, no residual damping needed

### Layer 3: Binary Training Pipeline ✅ BUILT
- **Headroom Adam** (`train.h:adam_update_headroom`) — MetabolicAge_v3 Theorem 68
  - h_s(w) = 2*clamp(w,0,1) for strengthening, h_w(w) = 2*(1-clamp(w,0,1)) for weakening
  - Floor of 0.1 ensures committed weights can still flip under overwhelming evidence
  - CFL clamp ±0.1 per step
  - Replaces match gate + propose-test-commit (headroom IS the gate)
- **Binary dequant** — `a_scale / sqrt(density * K)` (Gap 1, Theorem 69a)
  - Makes all projection outputs O(1) by construction
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
| Binary weights train stably | **T1** | 18.2% acc at step 2800, no NaN, no divergence |
| Past unigram baseline (15.2%) | **T1** | Crossed at step ~700, reached 18.2% by step 2800 |
| Topology crystallization | **T1** | 2415 flips total, locked by step ~1000, continuous learning continued |
| Dequant scale 1/sqrt(d*K) | **T1** | Two3Gaps.lean Theorem 69a, dissolved all five scaling hacks |
| Gain inv_rms backward | **T1** | O(1/d) projection correction by concentration of measure |
| Headroom Adam (Theorem 68) | **T1** | Replaces match gate + propose-test-commit, no flip avalanche |
| Jury stability (67x margin) | **T1** | Runtime check confirms gain kernel stable at init |
| Binary backward 1/sqrt(d*M) | **T1** | Same CLT argument as forward, transposed dimensions |

---

## Best Training Run (2026-04-07)

Config: dim=128, 4 layers, 4 heads, 2 KV heads, inter=512, Shakespeare 1.1MB
```
Step    Loss    Acc     Flips
1       5.72    0.0%    0
100     3.87    13.4%   162
300     3.24    14.6%   1009
700     3.22    15.3%   1769     ← unigram ceiling crossed
1000    3.09    15.6%   2030
1500    3.06    15.9%   2231
2000    2.69    16.5%   2337
2500    2.78    17.6%   2400
2800    2.86    18.2%   2415     ← run terminated
```

Topology locked by step ~1000. Remaining learning through continuous params only.
First empirical confirmation of the dissolution theorem: one substrate fix (dequant scale)
made the FFN contribute, and all five scaling hacks became unnecessary.

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
