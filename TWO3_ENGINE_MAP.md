# two3 × XYZT Engine: What's Wired, What's Proven, What's Open

**Purpose:** Map between XYZT engine patterns and two3 implementation.
Updated 2026-04-07 after Opus diagnostic session + Gap 1 fix.

---

## Pattern Map: Engine → two3

### 1. MATCH-GATED LEARNING (Thermodynamic Clutch) — ✅ SUPERSEDED

**Engine:** `pc/engine.c:612-656` — `graph_learn()`
Structural match controls learning rate. Low match → freeze. High match → crystallize.

**two3 history:** Was implemented as loss-ring match gate in train_driver.cu.
Gate opened only when loss trending down, froze topology otherwise.

**Status:** **REMOVED.** Headroom Adam (Theorem 68) replaces the match gate entirely.
Committed weights resist flipping by construction — no external gate needed.
The gate was a hack compensating for Adam's lack of basin structure.
Removed in commit cfa6961.

### 2. MULTIPLICATIVE STRENGTHEN/WEAKEN — ✅ WIRED (as Headroom)

**Engine:** `pc/tline.c:126-137` — multiplicative updates on impedance (Lc).
High Lc → disconnected. Low Lc → connected. Exponential divergence from boundary.

**two3 equivalent:** Headroom-modulated Adam (`train.h:adam_update_headroom`).
- h_s(w) = 2*clamp(w, 0, 1) — strengthening headroom
- h_w(w) = 2*(1 - clamp(w, 0, 1)) — weakening headroom
- Floor 0.1 — committed weights can still flip under overwhelming evidence
- CFL clamp ±0.1 per step

Not literally multiplicative on the weight, but achieves the same effect:
weights near 0 or 1 have near-zero effective learning rate, weights near 0.5
have full learning rate. Same basin dynamics as the engine's Lc field.

**Proven:** MetabolicAge_v3.lean Theorem 68. Equilibrium L*(p) is unique and stable.

### 3. ASYMMETRIC REINFORCE/ERODE — ⚠️ PARTIAL

**Engine:** `pc/engine.h:410-411` — learn_strengthen=65, learn_weaken=40.
`pc/infer.c:260-267` — +2 strengthen, -1 weaken.

**two3 status:** The headroom function is symmetric (h_s and h_w are mirror images).
The asymmetry from the engine (reinforce faster than erode) is NOT currently
wired. The adaptive K (+2/-1) from commit 6d291e0 was on flip caps, not weight
updates, and is now irrelevant since flips are controlled by headroom.

**Open question:** Should headroom have asymmetric floors? E.g., floor=0.15 for
strengthening, floor=0.05 for weakening? The engine says reinforcement should
dominate erosion. Currently symmetric at floor=0.1.

### 4. HEBBIAN CO-ACTIVATION — ⚠️ IMPLICIT

**Engine:** `pc/engine.c:895-914` — both sources active → strengthen, one silent → weaken.

**two3 status:** The STE gradient `dW = dY[m] * X[k]` IS Hebbian co-activation
interpreted through the gradient lens. Large input AND large gradient → strengthen.
But it goes through Adam's momentum averaging, losing the immediacy of the engine's
local Hebbian signal.

**Open question:** Would a direct Hebbian term (bypass Adam, small magnitude)
accelerate topology discovery? The engine uses it for initial structure; Adam
refines within the structure. Currently, Adam does both jobs.

### 5. HYPOTHESIS TESTING (0.3x Injection / Sponge) — ✅ SUPERSEDED

**Engine:** `pc/infer.c:210-230` — re-inject predictions at 0.3x amplitude.
Only predictions matching carved topology resonate. Sponge filters bad predictions.

**two3 history:** Implemented as propose-test-commit in train_driver.cu.
Save weights → requantize → forward pass → compare loss → commit or revert.

**Status:** **REMOVED.** Headroom Adam makes this unnecessary. Committed weights
(near 0 or 1) resist flipping — only well-supported flips happen. The headroom
IS the sponge. No explicit hypothesis test needed.
Removed in commit cfa6961.

### 6. PER-WEIGHT PLASTICITY — ⚠️ IMPLICIT

**Engine:** `pc/engine.h:91-95` — plasticity field per node.
Frustrated weights heat up (more plastic), stable weights cool down (more rigid).

**two3 status:** Adam's second moment v IS a plasticity proxy. High v = oscillating
gradient = hot = high plasticity. Low v = stable = cold = low plasticity.
But Adam recomputes v each step as exponential moving average. The engine's
plasticity is a persistent STATE that decays independently.

The headroom function adds another plasticity layer: weights near 0.5 are
inherently more plastic (higher effective learning rate) than committed weights.
This is closer to the engine's behavior than Adam's v alone.

**Assessment:** Likely sufficient. The combination of Adam's v + headroom gives
two plasticity signals. Adding a third explicit plasticity field is probably
over-engineering at this stage.

### 7. CRYSTAL UPDATE (Commitment Measurement) — ✅ WIRED (as diagnostics)

**Engine:** `pc/engine.c:579-590` — histogram edge weights per node.
Bimodal = crystallized. Uniform = liquid.

**two3 status:** Weight entropy logged per layer at diagnostic intervals:
`L0 W_q entropy: 0.9563 (max=1.5850)  [-1]=0.0% [0]=62.2% [+1]=37.8%`
Flip counting tracks topology changes over time.

The jury (`jury.h`) checks stability at init. The diagnostic logging tracks
crystallization during training. Between them, we have visibility.

---

## What's Proven (Lean theorems)

| Theorem | What | Status |
|---------|------|--------|
| 68 | Headroom Adam equilibrium L*(p) unique, stable | **Proven** (MetabolicAge_v3.lean) |
| 69a | Dequant scale 1/sqrt(d*K) makes output O(1) | **Proven** (Two3Gaps.lean) |
| 69b-69j | Full signal chain T1 | **Proven** (Two3Gaps.lean) |

---

## What's Open

1. **Grad spikes** — intermittent spikes to 30-72 in dW_q. Not growing, not causing
   instability, but not understood. Likely attention softmax sharpening. Investigate
   at dim=128 before scaling.

2. **Asymmetric headroom** — engine uses 2:1 reinforce/erode ratio. Current headroom
   is symmetric. May matter at larger scale where topology decisions are more costly.

3. **GPU-resident training** — plan exists (`.claude/plans/silly-finding-russell.md`).
   Pure engineering: move latent weights to GPU, stream optimizer per-layer.
   17s → 2s/step. Do after structural questions answered.

4. **Scale test** — dim=128 works. dim=256, dim=512, dim=1024 untested. Each scale
   transition may surface new structural issues (the lesson of this project).

5. **Generation quality** — 18.2% accuracy should produce recognizable English
   fragments. Not yet checked qualitatively.

---

## Key Commits

| Commit | What |
|--------|------|
| `6d73ceb` | Fix gain_backward_cpu: missing RMS norm backward |
| `9ae3b60` | Gate fp projection requantize — fix gradient explosion |
| `cfa6961` | Headroom Adam + gain inv_rms + FFN scaling + jury.h + binary init |
| `195e056` | Gap 1 fix: binary dequant a_scale/sqrt(d*K), delete scaling hacks |

---

*The engine already solved every problem two3 encountered. The work is porting
the solutions correctly — with mathematical proof, not hand-tuning.*
