/-
  CondHoeffding.lean — Conditional Hoeffding's lemma.

  Proves: bounded + conditional mean zero → conditionally sub-Gaussian.
  This is the missing constructor for `HasCondSubgaussianMGF` that Mathlib
  does not yet provide.

  The proof strategy is disintegration via `condExpKernel`:
  1. Rewrite the global a.e. hypotheses as fiberwise a.e. statements
  2. Apply `hasSubgaussianMGF_of_mem_Icc_of_integral_eq_zero` in each fiber
  3. Assemble into `Kernel.HasSubgaussianMGF`

  Key identity: `condExpKernel μ m ∘ₘ μ.trim hm = μ`

  Depends on: Mathlib.Probability.Moments.SubGaussian,
              Mathlib.Probability.Kernel.Condexp

  Author: Isaac + Claude Code, April 2026
-/

import Mathlib.Probability.Moments.SubGaussian
import Mathlib.Probability.Kernel.Condexp

open MeasureTheory ProbabilityTheory Real
open scoped ENNReal NNReal Topology

namespace ProbabilityTheory

variable {Ω : Type*} {m mΩ : MeasurableSpace Ω} [StandardBorelSpace Ω]
  {μ : Measure Ω} [IsFiniteMeasure μ] {X : Ω → ℝ}

/-! ## Fiberwise lemmas

  Lift global a.e. hypotheses to fiberwise a.e. statements over `condExpKernel`.
-/

/-- If `X ω ∈ Icc a b` a.e. `μ`, then for `μ.trim hm`-a.e. `ω'`,
    `X ω ∈ Icc a b` a.e. `condExpKernel μ m ω'`. -/
lemma ae_ae_condExpKernel_of_ae (hm : m ≤ mΩ) {p : Ω → Prop}
    (hp : ∀ᵐ ω ∂μ, p ω) :
    ∀ᵐ ω' ∂(μ.trim hm), ∀ᵐ ω ∂(condExpKernel μ m ω'), p ω := by
  apply Measure.ae_ae_of_ae_comp
  rwa [condExpKernel_comp_trim hm]

/-- If `X` is `μ`-integrable, then for `μ.trim hm`-a.e. `ω'`,
    `X` is integrable with respect to `condExpKernel μ m ω'`. -/
lemma ae_integrable_condExpKernel (hm : m ≤ mΩ)
    (h_int : Integrable X μ) :
    ∀ᵐ ω' ∂(μ.trim hm), Integrable X (condExpKernel μ m ω') := by
  apply Measure.ae_integrable_of_integrable_comp
  rwa [condExpKernel_comp_trim hm]

/-- If `μ[X | m] =ᵐ[μ] 0`, then for `μ.trim hm`-a.e. `ω'`,
    `∫ y, X y ∂(condExpKernel μ m ω') = 0`.

    This is the critical Step 2 bridge. The path:
    1. `ae_eq_trim_of_stronglyMeasurable` lifts `μ[X|m] =ᵐ[μ] 0`
       to `μ[X|m] =ᵐ[μ.trim hm] 0` (both sides are m-strongly measurable)
    2. `condExp_ae_eq_trim_integral_condExpKernel` gives
       `μ[X|m] =ᵐ[μ.trim hm] fun ω ↦ ∫ y, X y ∂(condExpKernel μ m ω)`
    3. Transitivity. -/
lemma ae_integral_condExpKernel_eq_zero (hm : m ≤ mΩ)
    (h_int : Integrable X μ)
    (h_condExp : μ[X | m] =ᵐ[μ] 0) :
    ∀ᵐ ω' ∂(μ.trim hm), ∫ y, X y ∂(condExpKernel μ m ω') = 0 := by
  -- Step 1: lift condExp =ᵐ[μ] 0 to condExp =ᵐ[μ.trim hm] 0
  have h_trim : μ[X | m] =ᵐ[μ.trim hm] (0 : Ω → ℝ) :=
    StronglyMeasurable.ae_eq_trim_of_stronglyMeasurable hm
      stronglyMeasurable_condExp stronglyMeasurable_const h_condExp
  -- Step 2: condExp equals integral over condExpKernel, a.e. μ.trim hm
  have h_kernel := condExp_ae_eq_trim_integral_condExpKernel hm h_int
  -- Step 3: transitivity
  filter_upwards [h_kernel.symm.trans h_trim] with ω' hω'
  exact hω'

/-! ## Main theorem: conditional Hoeffding's lemma -/

section CondHoeffding

/-- **Conditional Hoeffding's lemma.**

    If `X` is bounded in `[a, b]` a.e. `μ` and has conditional expectation
    zero given `m` (i.e., `μ[X | m] = 0` a.e.), then `X` is conditionally
    sub-Gaussian with parameter `((b - a) / 2)²`.

    This is the conditional analog of `hasSubgaussianMGF_of_mem_Icc_of_integral_eq_zero`.
    Combined with Mathlib's `measure_sum_ge_le_of_hasCondSubgaussianMGF`
    (Azuma-Hoeffding), this handles dependent sequences where the unconditional
    Hoeffding bound does not apply.

    T1: zero sorry. -/
theorem hasCondSubgaussianMGF_of_mem_Icc_of_condExp_eq_zero
    {a b : ℝ} (hm : m ≤ mΩ)
    (hX_meas : AEMeasurable X μ)
    (hX_bound : ∀ᵐ ω ∂μ, X ω ∈ Set.Icc a b)
    (hX_condExp : μ[X | m] =ᵐ[μ] 0) :
    HasCondSubgaussianMGF m hm X ((‖b - a‖₊ / 2) ^ 2) μ := by
  -- Unfold to Kernel.HasSubgaussianMGF
  show Kernel.HasSubgaussianMGF X _ (condExpKernel μ m) (μ.trim hm)
  -- Integrability of X (bounded → integrable under finite measure)
  have hX_int : Integrable X μ := Integrable.of_mem_Icc a b hX_meas hX_bound
  -- Field 1: integrability of exp(t * X) w.r.t. condExpKernel ∘ₘ μ.trim
  have h_exp_int : ∀ t, Integrable (fun ω ↦ exp (t * X ω)) (condExpKernel μ m ∘ₘ μ.trim hm) := by
    intro t
    rw [condExpKernel_comp_trim hm]
    exact integrable_exp_mul_of_mem_Icc hX_meas hX_bound
  -- Fiberwise hypotheses
  have h_fiber_bound := ae_ae_condExpKernel_of_ae hm hX_bound
  have h_fiber_int_zero := ae_integral_condExpKernel_eq_zero hm hX_int hX_condExp
  have h_fiber_integrable := ae_integrable_condExpKernel hm hX_int
  -- Field 2: fiberwise MGF bound
  -- For a.e. ω', the fiber measure is a probability measure (IsMarkovKernel),
  -- X is bounded a.e. in the fiber, and has integral zero in the fiber.
  -- Apply hasSubgaussianMGF_of_mem_Icc_of_integral_eq_zero fiberwise.
  constructor
  · exact h_exp_int
  · filter_upwards [h_fiber_bound, h_fiber_int_zero, h_fiber_integrable] with
        ω' hω'_bound hω'_zero hω'_int
    -- In each fiber, condExpKernel μ m ω' is a probability measure
    have : IsProbabilityMeasure (condExpKernel μ m ω') :=
      IsMarkovKernel.isProbabilityMeasure ω'
    -- AEMeasurable in the fiber follows from integrability
    have hω'_meas : AEMeasurable X (condExpKernel μ m ω') := hω'_int.1.aemeasurable
    -- Apply unconditional Hoeffding in the fiber
    exact (hasSubgaussianMGF_of_mem_Icc_of_integral_eq_zero hω'_meas hω'_bound hω'_zero).mgf_le

/-- Corollary: bounded + conditional mean zero → conditionally sub-Gaussian.
    Version that takes `μ[X | m] =ᵐ[μ] 0` directly, with `[0,1]` bounds.
    Convenient for binary/ternary transition indicators. -/
theorem hasCondSubgaussianMGF_of_mem_Icc01_of_condExp_eq_zero
    (hm : m ≤ mΩ)
    (hX_meas : AEMeasurable X μ)
    (hX_bound : ∀ᵐ ω ∂μ, X ω ∈ Set.Icc (0 : ℝ) 1)
    (hX_condExp : μ[X | m] =ᵐ[μ] 0) :
    HasCondSubgaussianMGF m hm X ((‖(1 : ℝ) - 0‖₊ / 2) ^ 2) μ :=
  hasCondSubgaussianMGF_of_mem_Icc_of_condExp_eq_zero hm hX_meas hX_bound hX_condExp

end CondHoeffding

end ProbabilityTheory
