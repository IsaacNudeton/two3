# two3 — Ternary Weight Neural Network

Ternary `{-1, 0, +1}` weight neural network built from scratch in CUDA C.
Runs on a single RTX 2080 Super. No frameworks, no tokenizer — raw bytes in, loss out.

## Architecture

- **Byte-level**: 256-entry identity embedding, no tokenizer
- **Ternary weights**: quantized at cos²θ thresholds 1/3 and 2/3
- **Gain kernel**: per-dimension learned scaling (C), frozen after init for reservoir stability
- **Headroom**: 3/2 from impedance physics, gain `C = sqrt(dim/3)`
- **Trimodal init**: weights start at `{1/6, 1/2, 5/6}` for maximum entropy
- **Tensor Cores**: INT8 WMMA forward matmul on sm_75 (RTX 20-series)

## Timeline

### Phase 1: Foundation (early April 2026)
- Built from-scratch training loop: `train_driver.cu`, `two3.cu`, `model.h`, `train.h`
- Shakespeare corpus, SGD optimizer
- Fixed three root-cause bugs: gain backward missing RMS norm, dequant scale, headroom in Adam
- Binary `{0,1}` → ternary `{-1, 0, +1}` restoration via cos²θ thresholds
- 37.1% accuracy on Shakespeare (full epoch)

### Phase 2: Performance (April 10-12)
- Device-resident path: all projections stay on GPU, 59% step speedup
- Tensor Core forward matmul via WMMA intrinsics (gated, bit-exact parity verified)
- Trimodal init at maximum entropy
- Headroom = 3/2 derived from impedance physics

### Phase 3: Code corpus + lexer experiments (April 14-16)
- Switched from Shakespeare to 4.3MB code corpus (267 C/CUDA files)
- Code corpus baseline: 43.66% (dim=128, 3 layers, 1 epoch)
- Structural lexer front-end (`lexer.h`, `lex_precompute.c`):
  - lex-class (16-class character type): 45.90% (+2.24)
  - lex-full (class + brace depth + paren depth + mode): 46.16% (+2.50)
  - dim=256 baseline: 46.04% (2x width matched lex-full at half the wall time)
  - Ternary codebook: 15.72% (falsified — learned embed carries real info)
- 3-epoch lex-full run: peaked 51.48% at step 16,875 (mid epoch 2)
  - Confirmed 46% was a 1-epoch stopping artifact, not a ceiling
  - Degraded to 40.3% by epoch 3 end — no LR decay, model overshoots

### Phase 4: Crystallization proofs (April 17)
- Direction shift: from code-domain experiments to fundamental model mechanism
- Per-weight crystallization: weights that stop moving provably lose plasticity
- Lean 4 + Mathlib proofs in `proofs/` — zero sorry, `lake build` clean
- Deep composability: crystallization commutes with gradient clamp, doesn't touch gain C, doesn't change forward pass

## Build

Requires: CUDA Toolkit, Visual Studio Build Tools, Developer Command Prompt.

```
build_driver.bat              # default SGD build
build_driver.bat binary       # device-resident binary weights
build_driver.bat binary-tc    # + Tensor Core forward matmul
build_driver.bat lex-class    # + character class embedding
build_driver.bat lex-full     # + class + depth + mode embedding
```

Run:
```
train_driver.exe corpus/corpus_code_isaac.bin --medium --layers 3 --seq-len 128 --batch 8 --log-every 50
```

## Proofs

Lean 4 + Mathlib project in `proofs/`. Requires elan.

```
cd proofs
lake exe cache get    # pull Mathlib oleans
lake build            # build all proofs
```

**GainKernel.lean** — Discrete gain kernel stability:
- Fixed-point existence and positivity
- Jury stability conditions (spectral radius < 1)
- Metabolic CFL bound
- Reservoir contraction to capacity

**Crystallization.lean** — Per-weight crystallization:
- Plasticity contraction theorem (factor = e-2 when variance = 0)
- Geometric crystallization under sustained stability
- K.I.D decision matrix (crystallize / prune / keep learning)
- Deep composability with gradient clamp, frozen C, ternary readout

Root-level `.lean` files (`GainKernel.lean`, `Two3Gaps.lean`, `SignalProtocol.lean`) are earlier proof sketches written before the Lean toolchain was set up locally. The compiled proofs are in `proofs/`.

## What's next

1. Evaluate peak checkpoint (step 16,800-16,900) vs final on held-out data
2. Add save-best checkpoint logic and LR decay
3. Crystallization C prototype on parity-N (toy validation per HANDOFF.md)
4. Port crystallization into two3 training loop if prototype validates

## Files

| File | Purpose |
|------|---------|
| `two3.cu` | Core model: attention, FFN, gain kernel, quantization |
| `model.h` | Model struct, forward pass, embedding, init |
| `train.h` | Training state, backward pass, optimizer, save/load |
| `train_driver.cu` | Training loop CLI |
| `build_driver.bat` | Multi-config build system |
| `lexer.h` | Structural lexer (character class, depth, mode) |
| `lex_precompute.c` | Preprocessor: corpus → .lex annotations |
| `infer_duo.cu` | Prompt-based inference tool |
| `proofs/` | Lean 4 + Mathlib formal proofs |
