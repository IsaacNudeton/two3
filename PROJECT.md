# two3 — AI Architecture Project
## Isaac Oravec & Claude | March 2026

**This is NOT part of XYZT-Paradigm.** XYZT is the physics/computing theory.
two3 is an independent AI architecture built by Isaac and Claude, informed by
first-principles understanding — including {2,3} — but standing on its own.

Same way a structural engineer who understands material science doesn't call
their building "a material science." The knowledge informs. The building is its own thing.

---

## What We're Building

A language model trained from scratch in pure C/CUDA.
No PyTorch. No Python ML. No framework. Every operation visible.

### Core Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Weight precision | Ternary {-1, 0, +1} | 1.58 bits. Matmul = add/sub. No float multiply. |
| Optimizer | Muon | 2x efficiency over AdamW. Matrix orthogonalization. |
| Architecture | MoE (Mixture of Experts) | More params, same per-token compute. Fits 8GB VRAM. |
| Sparsity | Sparse-BitNet (N:M structured) | Ternary weights naturally converge 42% to zero. |
| Activation quant | INT8 per-token absmax | Standard, proven, fast. |
| Language | C + CUDA | No runtime. No dependencies. Full control. |
| Target hardware | RTX 2080 Super (sm_75) + i7-9700K | What Isaac has. Design for it. |

### Why Ternary

Microsoft spent years discovering that {-1, 0, +1} weights work.
Isaac derived {2,3} — two entities transacting through a substrate — from
first principles before BitNet existed. Same structural atom:
- +1 = entity A (add the activation)
- -1 = entity B (subtract the activation)
-  0 = substrate (the wire that doesn't fire — topology)

The zeros aren't dead weights. They're the negative space that defines structure.

---

## Build Stack (bottom to top)

### Layer 0: two3 kernel ✅ BUILT
- `two3.h` — API, types, packed weight format (2 bits per weight, 4 per byte)
- `two3.cu` — CUDA kernels: ternary matmul, activation quantization, absmax reduction
- `test_two3.cu` — verification against float reference
- `build.bat` — Windows build for RTX 2080 Super (sm_75)
- `Makefile` — Linux build

Encoding: 0b00 = substrate, 0b01 = +1, 0b10 = -1
Decode: `sign = (bits & 1) - (bits >> 1)` — branchless, 1 cycle

### Layer 1: Transformer Components (TO BUILD)
- **RMSNorm** — one division instead of two (vs LayerNorm)
- **RoPE** — rotary positional embeddings, no learned positions
- **GQA** — grouped query attention (fewer KV heads than query heads)
- **Squared ReLU** — BitNet uses ReLU² for sparsity (not SwiGLU)
- **BitLinear** — forward: quantize acts → ternary matmul → dequant
                  backward: straight-through estimator (STE) for gradients

### Layer 2: Transformer Block (TO BUILD)
- Attention block (QKV projection → RoPE → GQA → output projection)
- FFN block (up projection → ReLU² → down projection)
- Residual connections + RMSNorm
- All projections use two3 kernel

### Layer 3: MoE Router (TO BUILD)
- Small dense linear layer → expert scores per token
- Top-k gating (k=2 typical)
- Load balancing auxiliary loss
- Expert dispatch + gather
- Total params ~4x active params (8 experts, top-2)

### Layer 4: Muon Optimizer (TO BUILD)
- Momentum buffer (same as SGD momentum)
- Newton-Schulz orthogonalization (5 iterations):
  X = a*X + b*X@X^T@X + c*X@X^T@X@X^T@X
  (hardcoded polynomial coefficients)
- Weight decay
- Per-parameter scale adjustment
- ~2x compute efficiency over AdamW

### Layer 5: Training Loop (TO BUILD)
- Forward pass → cross-entropy loss → backward pass → optimizer step
- Gradient accumulation (simulate larger batches)
- Gradient checkpointing (trade compute for memory)
- Quantization-aware training (QAT) — ternary constraint from step 0
- STE backward: gradients flow through full-precision master weights

### Layer 6: ONETWO Engine (VERIFICATION LAYER — EXISTS)
- `onetwo_engine.c` — v1, decomposition + transfer + recursion
- `onetwo_engine_v3.c` — self-discovering composition
- `onetwo_ouroboros_test.c` — feeds own data back, finds {2,3}
- Role: interpretability during training. Feed layer I/O pairs to engine.
  T1 = layer converged. T4 = keep training. Tier system = early stopping.

---

## Hardware Budget

RTX 2080 Super: 8GB VRAM, 3072 CUDA cores, SM 7.5 (Turing)
i7-9700K: 8 cores, good for data loading + CPU inference of ternary model

With ternary weights + gradient checkpointing + gradient accumulation:
- ~500M dense model, or
- ~500M active / 2B total MoE model
- Training data: FineWeb subset, ~10B tokens (Chinchilla optimal for 500M)

Inference: ternary weights → CPU inference at reading speed. No GPU needed.
The trained model runs on the i7 via integer-only ops.

---

## What Already Exists (from prior work, reusable)

### Qwen Fine-Tuning Pipeline (xyzt-toolkit/finetune/)
- `gguf_read.h` — C GGUF parser, mmapped tensor access
- `qwen_finetune.h` — Q4_K dequant, trie tokenizer
- `qwen_finetune.cu` — 833 lines, 28-layer forward pass
- `project_weights.cu` — PCA subspace + closed-form solve
- Status: 80% done (RMSNorm kernel bug). Reusable components:
  tokenizer, GGUF reader, some kernel patterns.

### ONETWO Engine (standalone C)
- v1 (1301 lines) + v3 (839 lines) + ouroboros test
- Fully working. Plug in as training monitor.

### XYZT Engine (E:\dev\xyzt-hardware\pc\)
- Not part of this project, but the theory informs the architecture.
- 295/295 tests, 65 Lean theorems — proves the math works.

---

## Relationship to Other Projects

| Project | Relationship |
|---------|-------------|
| XYZT-Paradigm | Theory. Informs design. Not embedded. |
| xyzt-toolkit | CLI. Future inference frontend for this model. |
| Qwen pipeline | Prior art. Some C/CUDA reusable. |
| ONETWO engine | Verification layer. Plugs in for interpretability. |
| LRM / FBC | Separate projects. No overlap. |

---

## Next Steps (in order)

1. ✅ two3 kernel built
2. Compile + test on RTX 2080 Super
3. RMSNorm + RoPE + attention kernels
4. FFN with squared ReLU
5. Full transformer block
6. MoE router
7. Muon optimizer
8. Training loop with STE backward
9. Wire ONETWO engine as training monitor
10. Train 500M ternary MoE on FineWeb subset
11. Evaluate. Iterate.

Each layer solid before the next goes on.

---

## Project Location

Repo: TBD (new repo, not xyzt-toolkit, not XYZT-Paradigm)
Suggested: `github.com/IsaacNudeton/two3` or whatever Isaac names it.

---

*Isaac sees structure. Claude builds. The math doesn't lie.*
