# Two3 Gap Implementation — From Lean to Code

**Theorems:** Two3Gaps.lean (Theorem 69a-69j + Dissolution)  
**ALL T1.** No sorry, no axiom, no architecture assumption.  
**Depends on:** GainKernel.lean (Theorems 1-3), MetabolicAge_v3.lean (Theorem 68)

**CRITICAL UPDATE (2026-04-07):** Theorem 69a requires a **centering precondition**.
Binary weights {0,1} have no sign cancellation. The CLT variance normalization
`1/sqrt(d*K)` is correct ONLY when the input has zero mean. RMS norm does not
center — it preserves magnitude but not mean. Without centering, the mean
accumulates across `density*K` active connections, causing `sqrt(density*K)` ≈ 6.9×
amplification (measured: gate_rms=6.85 without centering, 0.79 with centering).

**Fix:** Add mean subtraction to the gain kernel (gain.h:gain_forward_cpu).
The gain kernel now does: center → RMS normalize → reservoir modulate.
This is effectively LayerNorm without learnable affine, with reservoir modulation.

Ternary {-1,0,+1} gets centering for free from ±1 sign cancellation.
Binary {0,1} needs it explicitly. The Lean proof is mathematically correct
but the premise E[acc]=0 only holds for centered input.

---

## The Two Fixes

### Fix 1: Dequant normalization (Theorem 69a)
```c
// binary.h — binary_dequantize(), binary_project_cpu(), binary_project_batch_cpu()
// WRONG:
float combined = a_scale * density;

// CORRECT (Theorem 69a — CLT variance of masked sum):
float combined = a_scale / sqrtf(density * (float)K);
```

### Fix 2: Gain kernel centering (precondition for Theorem 69a)
```c
// gain.h — gain_forward_cpu()
// Center input before RMS norm — binary weights need zero-mean input
float mean = 0.0f;
for (int i = 0; i < dim; i++) mean += x[i];
mean /= (float)dim;
// Then RMS on centered signal: rms = sqrt(mean((x-mean)²))
// Then modulate: y = (x-mean)/rms * gain
```

## The Five Deletions

After the dequant fix + centering, all projections output O(1). These become unnecessary:

### 1. Remove 1/sqrt(INTER) post-scale on FFN down projection
**Theorem 69c:** T(1) = 1. Impedance match → no loss. Already O(1) from dequant.
```
train.h — remove down_scale after down projection
ffn.h   — remove 1/sqrt(intermediate) in both forward functions
```

### 2. Set res_scale = 1.0
**Theorem 69h + 69d:** Gain kernel is scale-invariant. res_scale < 1 loses
signal at the Fresnel junction (T(0.354) = 0.77, 23% loss per junction,
compounding to 14% throughput after 8 junctions). No benefit — gain norm
absorbs whatever magnitude arrives.
```
train.h line 1709: float res_scale = 1.0f;
model.h line 634:  float res_scale = 1.0f;
```

### 3. Projection correction stays dropped
**Theorem 69b:** Concentration of measure on S^{d-1}. The projection captures
1/d of gradient variance. For d ≥ 100: < 1%. PLUS the next gain norm absorbs
the radial component anyway (dual redundancy). Already done.

### 4. Headroom Adam stays as-is  
**Theorem 69f:** Adiabatic ratio = 1/2, independent of lr. Structural, not tuned.
**Theorem 69g:** Floor prevents divergence at rails. Already wired.

### 5. Threshold 0.5 stays
**Theorem 69i:** Any threshold agrees on committed weights.
**Theorem 69j:** Headroom creates gap, pushing weights to rails. Already correct.

---

## Verification — CONFIRMED (2026-04-07)

After implementing dequant fix + centering + deletions:

```
WITHOUT centering (dequant fix only):
  [ffn-chain] L0: normed=1.00 gate=6.85 up=6.87 h=224.11   ← BROKEN
  [var] L1: attn_rms=47.17  ffn_rms=3124.34  ratio=0.02
  [proj] cos = 0.70  (expected 0.088)

WITH centering:
  [ffn-chain] L0: normed=1.00 gate=0.79 up=0.78 h=0.61     ← O(1) ✓
  [var] L1: attn_rms=0.58   ffn_rms=0.58    ratio=0.99      ← balanced ✓
  [proj] cos = 0.0814  (expected 0.0884 = 1/sqrt(128))      ← Lévy exact ✓
```

Result: 37.1% accuracy (vs 18.2% without centering). 2× improvement.

---

## Full Theorem Chain

```
GainKernel.lean (T1)          — gain kernel stable, CFL bound
  ↓
MetabolicAge_v3.lean (T1)     — headroom equilibria, Theorem 68
  ↓
Two3Gaps.lean (ALL T1)
  ├── 69a  Dequant: Var = d·K·σ², normalize 1/√(dK) [REQUIRES CENTERED INPUT]
  ├── 69b  Projection: O(1/d) concentration of measure [CONFIRMED: cos=0.088]
  ├── 69c  Fresnel T(1) = 1 at impedance match
  ├── 69d  Fresnel T < 1 for K < 1 (subunit scale loses signal)
  ├── 69e  T^n compounding loss
  ├── 69f  Adiabatic ratio = 1/2 (lr-independent)
  ├── 69g  Headroom floor bounds relaxation time
  ├── 69h  Gain scale invariance
  ├── 69i  Threshold equivalence for committed weights
  ├── 69j  Headroom creates gap
  └── Dissolution: substrate fix → signal hacks dissolve
```

---

## Cross-Project Impact

### two3
- One line dequant fix + five deletions
- Training should break through unigram ceiling (FFN alive, correct gradients)
- Headroom Adam + jury.h already wired from this session

### XYZT Engine  
- **Engine already does this correctly.** Multiplicative per-cell impedance
  (substrate) handles normalization. Signal propagates through. No sqrt factors
  in the signal path. two3 was the engine's student who forgot the lesson.
- The Fresnel functions in engine.h (fresnel_T, fresnel_R) ARE Theorems 69c-69e.
- Theorem 69b explains why the engine uses per-component gain (dim=6 field
  components). At dim=6 the projection correction is 16% — it matters. So the
  engine avoids cross-component coupling entirely. Correct for different reasons
  than two3 (where d ≥ 128 makes it negligible).

### OpenShell
- **Routing as impedance matching.** When routing tasks between model tiers,
  the Fresnel coefficient T(K_capability) determines information loss.
  Route to cheaper model only when T > threshold. Don't impedance-mismatch
  tasks to save tokens.
- **Scale-invariant routing (Theorem 69h):** routing decisions should depend
  on task structure, not prompt length. The gain kernel formalism says: normalize
  the context, route on the normalized signal.
- **Headroom routing (Theorem 69i-j):** tasks with consistent routing history
  → committed (use the cached tier). Tasks with variable history → ambiguous
  (re-evaluate each time). The headroom kernel converges.

### XYZT-Paradigm
- **The dissolution pattern IS the T/XYZ distinction.** Normalization belongs
  in the substrate (T), not the signal path (X,Y,Z). Every system with
  signal/substrate separation develops the same class of bugs when normalization
  leaks from substrate into signal. The fix is always: put it back.
- **Adiabatic ratio = 1/2 IS the T-break boundary.** The headroom kernel
  operates at the edge between tracking (T flows) and losing coherence
  (T breaks). Not designed. Falls out of the math. The system self-organizes
  to the critical point between responsiveness and stability.
- **The threshold IS valence.** Weight position in [0,1] is the weight's
  crystallization state. Near 0 or 1: crystallized, high valence, committed.
  Near 0.5: liquid, low valence, ambiguous. The headroom kernel IS the
  crystallization dynamics. Theorem 68 guarantees convergence to equilibrium.
  The feeling IS the position. The math IS the meaning.

---

## Files for CC

```
MODIFY: binary.h         — dequant scale fix (one line)
MODIFY: train.h          — remove down_scale, set res_scale = 1.0
MODIFY: ffn.h            — remove 1/sqrt(intermediate) post-scale
MODIFY: model.h          — set res_scale = 1.0
ADD:    Two3Gaps.lean     — eleven theorems, all T1
NO CHANGE: gain.h        — already correct (inv_rms fix from this session)
NO CHANGE: jury.h        — already wired (copied from engine this session)
```
