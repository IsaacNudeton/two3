/-
  LRFiring.lean — Firing condition for crystallization under Adam + CFL clamp.

  The C code (train.h adam_update / adam_update_headroom) hard-clamps the
  Adam update to ±0.1, then headroom multiplies by at most headroom_peak.
  This bounds per-step displacement to CFL_CLAMP * HEADROOM_PEAK = 0.15.

  For crystallization to fire, per-step displacement must be < bin_width = 1/3
  (otherwise the weight crosses bins every step, variance = 1,
  contractionFactor = 1, no contraction).

  0.15 < 1/3.  This is the firing condition.  It holds from the CFL clamp
  alone — the LR schedule is irrelevant to firing.

  The LR schedule's role is persistence (keeping displacement low for M
  consecutive steps so crystallization locks in).  That theorem is open;
  see proofs/LR_Schedule.lean.todo.

  References:
    train.h:349-350  — CFL clamp: |update| ≤ 0.1
    train.h:404-405  — headroom: update *= h, h ≤ headroom_peak
    train.h:382      — headroom_peak = 1.5 (passed from train_driver.cu)
    Crystallization.lean — contraction requires variance < 1
-/

import Mathlib.Data.Real.Basic
import Mathlib.Tactic

namespace LRFiring

/-! ## The CFL update clamp

  Models train.h lines 349-350 (adam_update) and 392-393 (adam_update_headroom):
    if (update >  0.1f) update =  0.1f;
    if (update < -0.1f) update = -0.1f;
-/

noncomputable def cflClamp (cfl_bound : ℝ) (u : ℝ) : ℝ :=
  if u > cfl_bound then cfl_bound
  else if u < -cfl_bound then -cfl_bound
  else u

lemma cflClamp_abs_le (cfl_bound : ℝ) (hcfl : 0 ≤ cfl_bound) (u : ℝ) :
    |cflClamp cfl_bound u| ≤ cfl_bound := by
  unfold cflClamp
  split_ifs with h1 h2
  · -- u > cfl_bound → result is cfl_bound
    rw [abs_of_nonneg hcfl]
  · -- u < -cfl_bound → result is -cfl_bound
    rw [abs_neg, abs_of_nonneg hcfl]
  · -- -cfl_bound ≤ u ≤ cfl_bound
    rw [abs_le]
    constructor
    · linarith
    · linarith

/-! ## Shipped constants -/

noncomputable def CFL_BOUND : ℝ := 1 / 10
noncomputable def HEADROOM_PEAK : ℝ := 3 / 2
noncomputable def BIN_WIDTH : ℝ := 1 / 3

lemma cfl_bound_nonneg : (0 : ℝ) ≤ CFL_BOUND := by
  unfold CFL_BOUND; norm_num

lemma headroom_peak_nonneg : (0 : ℝ) ≤ HEADROOM_PEAK := by
  unfold HEADROOM_PEAK; norm_num

/-! ## Firing condition -/

theorem displacement_bounded (u h : ℝ) (hh : |h| ≤ HEADROOM_PEAK) :
    |cflClamp CFL_BOUND u * h| ≤ CFL_BOUND * HEADROOM_PEAK := by
  rw [abs_mul]
  apply mul_le_mul
  · exact cflClamp_abs_le CFL_BOUND cfl_bound_nonneg u
  · exact hh
  · exact abs_nonneg h
  · exact cfl_bound_nonneg

/-- **Firing condition**: CFL_BOUND * HEADROOM_PEAK = 3/20 < 1/3 = BIN_WIDTH.
    T1: zero sorry, proven from shipped constants. -/
theorem firing_condition :
    CFL_BOUND * HEADROOM_PEAK < BIN_WIDTH := by
  unfold CFL_BOUND HEADROOM_PEAK BIN_WIDTH
  field_simp
  norm_num

/-- Combined: any Adam update's displacement is strictly below bin width. -/
theorem displacement_below_bin_width (u h : ℝ) (hh : |h| ≤ HEADROOM_PEAK) :
    |cflClamp CFL_BOUND u * h| < BIN_WIDTH := by
  calc |cflClamp CFL_BOUND u * h|
      ≤ CFL_BOUND * HEADROOM_PEAK := displacement_bounded u h hh
    _ < BIN_WIDTH := firing_condition

end LRFiring
