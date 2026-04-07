/*
 * jury.h — Jury stability check for 2x2 discrete maps
 *
 * Given a discrete map with Jacobian trace τ and determinant δ,
 * checks all three Jury conditions for eigenvalues inside unit circle.
 *
 * Works for any system of the form:
 *   x' = f(x, y)
 *   y' = g(x, y)
 *
 * Compute Jacobian at the fixed point, pass trace and det.
 * Returns: stability verdict + safety margins.
 *
 * Derived from GainKernel.lean (Lean4 verified, T1).
 * Isaac & Claude — March 2026
 */

#ifndef JURY_H
#define JURY_H

#include <math.h>

typedef struct {
    int stable;        /* 1 = all conditions met */
    float j1;          /* 1 - τ + δ  (must be > 0) */
    float j2;          /* 1 + τ + δ  (must be > 0, binding constraint) */
    float j3;          /* 1 - δ      (must be > 0) */
    float margin;      /* min(j1, j2, j3) — distance from instability */
    float spectral_r;  /* estimated spectral radius */
} JuryResult;

/* Check Jury conditions given trace and determinant of Jacobian */
static JuryResult jury_check(float trace, float det) {
    JuryResult r;
    r.j1 = 1.0f - trace + det;
    r.j2 = 1.0f + trace + det;
    r.j3 = 1.0f - det;
    r.stable = (r.j1 > 0) && (r.j2 > 0) && (r.j3 > 0);
    r.margin = r.j1;
    if (r.j2 < r.margin) r.margin = r.j2;
    if (r.j3 < r.margin) r.margin = r.j3;

    /* Spectral radius from eigenvalues: λ = (τ ± √(τ²-4δ))/2 */
    float disc = trace * trace - 4.0f * det;
    if (disc >= 0) {
        float sq = sqrtf(disc);
        float l1 = fabsf((trace + sq) * 0.5f);
        float l2 = fabsf((trace - sq) * 0.5f);
        r.spectral_r = l1 > l2 ? l1 : l2;
    } else {
        /* Complex eigenvalues: |λ| = √δ */
        r.spectral_r = sqrtf(fabsf(det));
    }
    return r;
}

/* Metabolic gain kernel: check stability for given parameters.
 * α = gain coupling, β = grid loss, γ = recharge rate, C = capacity.
 * From GainKernel.lean Theorems 1-3. */
static JuryResult jury_gain_kernel(float alpha, float beta,
                                   float gamma, float C) {
    float theta = alpha * C / beta;  /* over-threshold ratio */
    float trace = 2.0f - gamma * theta;
    float det = 1.0f - gamma * theta * (1.0f - beta) - beta * gamma;
    return jury_check(trace, det);
}

/* Derive maximum safe gain coupling for given parameters.
 * Returns the upper bound on α from the metabolic CFL condition.
 * Use α < 0.5 * max for a 2x safety margin. */
static float jury_max_gain(float beta, float gamma, float C) {
    if (C <= 0 || gamma <= 0 || beta <= 0 || beta >= 1.0f) return 0;
    return (4.0f - gamma * beta) * beta / (gamma * C * (2.0f - beta));
}

/* Generic: compute Jacobian numerically for any 2x2 map.
 * f(x,y) → (x', y').  Evaluates at (x0, y0) with step h.
 * Then runs jury_check on the result. */
typedef void (*Map2x2)(float x, float y, float *xn, float *yn, void *ctx);

static JuryResult jury_check_numeric(Map2x2 f, float x0, float y0,
                                     float h, void *ctx) {
    float fx, fy, gx, gy;
    float xn, yn, xp, yp;

    /* ∂f/∂x, ∂g/∂x */
    f(x0 + h, y0, &xp, &yp, ctx);
    f(x0 - h, y0, &xn, &yn, ctx);
    float dfdx = (xp - xn) / (2.0f * h);
    float dgdx = (yp - yn) / (2.0f * h);

    /* ∂f/∂y, ∂g/∂y */
    f(x0, y0 + h, &xp, &yp, ctx);
    f(x0, y0 - h, &xn, &yn, ctx);
    float dfdy = (xp - xn) / (2.0f * h);
    float dgdy = (yp - yn) / (2.0f * h);

    float trace = dfdx + dgdy;
    float det = dfdx * dgdy - dfdy * dgdx;
    return jury_check(trace, det);
}

#endif /* JURY_H */
