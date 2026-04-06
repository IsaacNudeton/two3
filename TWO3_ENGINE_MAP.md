# two3 × XYZT Engine: What Exists, What's Solved, What's Not Wired

**Purpose:** Stop rebuilding from scratch. Every pattern two3 needs already exists
in the XYZT engine. This maps the solutions to the problems.

---

## The Binary Weight Training Problem

Binary weights {0,1} are topology — connected or not. Training decides which
connections exist. The current approach (Adam on latent floats + requantize at
threshold 0.5) causes flip churn: weights oscillate across the boundary, degrading
accuracy from 15.5% back to 14% at step 4200.

**Root cause:** Adam is additive. A 0.01 step at w=0.49 crosses the boundary.
No basin structure, no commitment dynamics. The engine solved this.

---

## Pattern Map: Engine → two3

### 1. MATCH-GATED LEARNING (Thermodynamic Clutch)

**Engine:** `pc/engine.c:612-656` — `graph_learn()`
```
structural_match controls learning rate.
Low match (< 30) → freeze (0.1x). High match (> 150) → crystallize (2x).
"Low match = slipping (don't lock bad topology)"
```

**two3 equivalent:** Gate the requantize step by whether loss is improving.
- Loss decreasing → allow flips (the topology is getting better)
- Loss increasing → freeze topology (don't churn during a bad patch)
- NOT a threshold. A continuous multiplier: `match_gate = clamp(loss_ratio, 0.1, 2.0)`

**Commit:** `85df552` — "ONETWO feedback[0] → Hebbian gating"

### 2. MULTIPLICATIVE STRENGTHEN/WEAKEN

**Engine:** `pc/tline.c:126-137`
```c
void tline_strengthen(TLine *tl, double rate) {
    for (int i = 0; i < tl->n_cells; i++) {
        tl->Lc[i] *= (1.0 - rate);    // impedance drops → connection strengthens
        if (tl->Lc[i] < 0.1) tl->Lc[i] = 0.1;
    }
}
void tline_weaken(TLine *tl, double rate) {
    for (int i = 0; i < tl->n_cells; i++) {
        tl->Lc[i] *= (1.0 + rate);    // impedance rises → connection weakens
        if (tl->Lc[i] > 50.0) tl->Lc[i] = 50.0;
    }
}
```

**Key insight:** Multiplicative updates on IMPEDANCE (Lc), not on learning rate.
The weight is a READOUT: `tline_weight()` = product of per-cell attenuation.
High Lc → high impedance → low weight → disconnected.
Low Lc → low impedance → high weight → connected.
Weights commit naturally because multiplicative updates create exponential
divergence from any boundary.

**two3 equivalent:** The latent float IS Lc. Don't threshold at 0.5.
Instead: `latent *= (1 - rate)` to strengthen, `latent *= (1 + rate)` to weaken.
The binary readout is: `latent < threshold → connected`. The threshold doesn't
need to be 0.5 — it's wherever the impedance naturally separates into two basins.

**WARNING:** CC tried this and got it backwards (commit in session — applied
commitment factor to learning rate instead of to the weight itself). The
multiplication goes on the WEIGHT, the rate stays additive. Don't repeat this.

### 3. ASYMMETRIC REINFORCE/ERODE

**Engine:** `pc/engine.h:410-411`
```c
g->learn_strengthen = 65;
g->learn_weaken = 40;
```

**Engine:** `pc/infer.c:260-267` — verification loop
```c
if (verified) {
    int nw = (int)ed->weight + 2;     // +2 strengthen
    ed->weight = nw > 255 ? 255 : (uint8_t)nw;
} else {
    int nw = (int)ed->weight - 1;     // -1 weaken
    ed->weight = nw < 1 ? 1 : (uint8_t)nw;
}
```

**Key insight:** Strengthening is faster than weakening (65 vs 40, or +2 vs -1).
Commitment has momentum. Erosion is slow. This is the 2:1 asymmetry from the
metabolic diagnostic — Hebbian sharpens boundaries because reinforcement > erosion.

**two3 equivalent:** Adaptive K already uses +2/-1 (commit `6d291e0`). This piece
is wired in. But it's on K (flip cap), not on the actual weight update magnitude.

### 4. HEBBIAN CO-ACTIVATION

**Engine:** `pc/engine.c:895-914`
```c
// Both sources active → strengthen
if (na->val != 0 && nb->val != 0) {
    tline_strengthen(&e->tl, 0.01);
}
// One active, one silent → weaken
else if ((na->val != 0) != (nb->val != 0)) {
    tline_weaken(&e->tl, 0.01);
}
```

**Key insight:** The learning signal is CO-ACTIVATION, not gradient. Two neurons
fire together → strengthen their connection. One fires without the other → weaken.
No backprop. No chain rule. Local signal only.

**two3 equivalent:** For binary weight w connecting input k to output m:
- Forward: if w=1, input[k] contributes to output[m]
- If both input[k] and the gradient dY[m] are large → the connection is useful → strengthen
- If input[k] is large but dY[m] is small (or vice versa) → connection isn't contributing → weaken
- This IS the STE gradient, but interpreted as Hebbian co-activation instead of as
  a continuous gradient to be fed into Adam.

### 5. HYPOTHESIS TESTING (0.3x Injection)

**Engine:** `pc/infer.c:210-230` — cortex prediction loop
```
Re-inject predictions at 0.3x amplitude (hypothesis, not statement).
Only predictions matching carved topology resonate.
Full amplitude would always resonate = hallucination.
The sponge is the bullshit detector.
```

**two3 equivalent:** Before committing a batch of flips:
1. Apply proposed flips
2. Run a forward pass on a validation batch
3. If loss improves → commit (the flips resonated with the topology)
4. If loss worsens → revert (the sponge absorbed them)

CC attempted this (propose-test-commit in train.h) but used a 10% loss threshold
which is too loose. The engine doesn't use a threshold — it uses the sponge
(physics filtering). The gain kernel IS the sponge. Depletion absorbs non-resonant
signal. The test should be: do the flipped weights produce LOWER depletion (more
efficient use of reservoir capacity)?

### 6. PER-WEIGHT PLASTICITY

**Engine:** `pc/engine.h:91-95`
```c
#define PLASTICITY_DEFAULT  1.0f
#define PLASTICITY_MIN      0.5f
#define PLASTICITY_MAX      2.0f
#define PLASTICITY_HEAT     0.01f   // frustration increment
#define PLASTICITY_COOL     0.005f  // boredom decrement
```

**Engine:** `pc/infer.c:274-281`
```c
if (verified) {
    pn->plasticity -= PLASTICITY_COOL;    // stable → harder to move
} else {
    pn->plasticity += PLASTICITY_HEAT;    // frustrated → easier to move
}
```

**two3 equivalent:** Adam's v (second moment) IS a plasticity signal.
High v = oscillating gradient = hot weight = high plasticity.
Low v = stable gradient = cold weight = low plasticity.
`sqrt(v_hat)` in the Adam denominator already does this implicitly —
large v → smaller effective step. But Adam uses it as normalization,
not as commitment. The engine treats plasticity as a STATE that
persists and decays. Adam recomputes it fresh each step.

### 7. CRYSTAL UPDATE (Commitment Measurement)

**Engine:** `pc/engine.c:579-590` — `crystal_update()`
```c
void crystal_update(Node *n, Edge *edges, int n_edges, int node_id) {
    memset(n->crystal_hist, 0, 8);
    for (int e = 0; e < n_edges; e++) {
        if (edges[e].dst == (uint16_t)node_id && edges[e].weight > 0) {
            int bin = edges[e].weight / 32;
            if (bin > 7) bin = 7;
            n->crystal_hist[bin]++;
        }
    }
}
```

**Key insight:** The engine tracks the DISTRIBUTION of edge weights per node.
Bimodal distribution (most weights near 0 or 255) = crystallized.
Uniform distribution = still liquid. This IS the metabolic diagnostic —
`metabolic_diagnostic.h` measures the same thing for the L-field.

**two3 equivalent:** Histogram the latent weights per layer.
Bimodal (clustered near 0 and 1) = committed topology, stop flipping.
Uniform (clustered near 0.5) = still deciding, allow flips.
This directly replaces the K cap with a structural signal.

---

## Files That Already Exist and Are Not Used

### In two3 repo:
- `nibble.h` — DEAD, not included anywhere. Delete.
- `moe.h` — DEAD, only used by test_moe.cu. Delete.
- `ibc.h` — guarded by TWO3_IBC, never defined. Dead in practice.
- `binary.h` lines 85-94 — cudaMalloc path was removed (segfault fix),
  needs dual storage (host+device) to re-enable GPU kernels.
- KV cache in `model.h` — fully implemented, not wired into generation.

### In XYZT repo (applicable to two3):
- `pc/engine.c:612` — `graph_learn()` with match gating
- `pc/tline.c:126` — multiplicative strengthen/weaken
- `pc/infer.c:195` — full inference with sponge + hypothesis testing
- `pc/engine.c:895` — Hebbian co-activation learning
- `metabolic_diagnostic.h` — boundary commitment measurement

### Uploaded but not referenced:
- `metabolic_diagnostic.h` — L-field Laplacian diagnostic for boundary sharpening
- `two3_tiled.h` (uploaded version) — has `TrainBuffers`, `requantize_gpu`,
  `kernel_fused_requantize`, persistent GPU allocation. More complete than
  the repo version.

---

## What's Proven

| Result | Status | Evidence |
|--------|--------|----------|
| Binary weights train stably | **T1** | 15.5% acc, flat grads, 27K+ flips absorbed |
| No polarity reversal cascade | **T1** | Ternary cascaded at step 1200, binary never |
| Past trivial baseline (15.2%) | **T1** | Peak 15.5% at step 2300 |
| Gradient √M scaling fix | **T1** | Three independent derivations converged |
| Binary discriminates inputs | **T1** | Gemini test: cosine sim -0.005 (orthogonal) |
| Flip churn degrades accuracy | **T1** | 15.5% → 14.0% during churn phase |
| Engine dynamics prevent churn | **T2** | Code exists, not tested in two3 context |

---

## What's Not Solved

1. **Flip churn** — weights oscillate across 0.5 under Adam. Engine solution
   (match gating + multiplicative + plasticity) not correctly ported.
2. **GPU path** — binary matmul runs on CPU (820ms/step). GPU kernel written
   (`binary_gpu.h`) but needs dual storage and wiring.
3. **Generation** — outputs spaces. Model predicts non-space bytes correctly
   (15.5%) but greedy decode always picks space. Temperature sampling needed.
4. **Scale** — dim=128, 4 layers, 1.1MB Shakespeare. Architecture proven but
   not tested at meaningful scale.

---

## Strategic Sequence

The question Isaac asked and hasn't answered: what is two3 FOR?

Options on the table:
- **Inference engine:** Binarize pretrained weights, fast inference on consumer GPU
- **Training from scratch:** Prove binary can learn competitive models end-to-end
- **XYZT substrate:** two3 weights as the topology, gain kernel as the dynamics,
  wave substrate (Yee FDTD) as the compute medium
- **Hardware target:** Zynq 7020 unified memory, no PCIe boundary, binary weights
  as FPGA LUT configuration

The architecture works at small scale. The next step depends on which of these
Isaac wants to build toward.
