# REPORT_LEDUC_V3 — Overnight follow-up experiments T1–T5

*Run started 2026-09-23 21:17 UTC on the V1 pipeline (same 300 held-out opponents, same exact ε-safe LP,
same OpenSpiel audit, same train bank / bank posterior / tabular EM).  Nothing below is tuned on the
evaluation opponents; the 150 validation opponents (carved from the training population in V1) are used for
every selection.*

## 0. Pre-registration (written before any task was run; not edited afterwards)

Common protocol: primary ε = 0.10; N ∈ {5, 10, 20, 50, 100, 200, 500}; ≥ 3 seeds per headline arm; paired
bootstrap 95 % CIs over the 300 held-out opponents (8 streams averaged within opponent); seed spread reported
separately; every deployed strategy audited with OpenSpiel's C++ best response (requirement
Expl(x) ≤ ε + 1e−7, violation count must be 0); per-family and pooled reporting.  Compute budget: 4 CPU
cores, ≈ 10 h; each task has a time box and is cut at the box with partial results recorded.

### T1 — confound fixes (box 2.5 h)
* Arms: DEC-133k (decision head hidden 100, 133 393 params) vs RECON-131k (V1 model, 131 331 params);
  DEC-889k (V1 model) vs RECON-889k (hidden 834, ≈ 889k); RECON-JAC-889k (cross-entropy weighted per opponent
  infoset by the mean Frobenius norm of ∂g/∂q(I,·) at the true q over training opponents, normalized to
  mean 1); RECON-REACH-889k (weights = chance × learner-blueprint reach into the infoset, normalized).
  All new runs: 6 000-step cap, validation every 250 steps, early stopping with patience 6 validations
  (1 500 steps) on the arm's own validation loss, step-3000 checkpoint saved for T2.
* Prediction P1.1: the decision advantage (V1: −0.010 to −0.035 chips at ε=0.10) shrinks but survives
  parameter matching in both directions: DEC-133k < RECON-131k and DEC-889k < RECON-889k in regret at every
  N ≥ 20 with paired CIs excluding 0.
* Prediction P1.2: Jacobian-weighted reconstruction closes part of the gap (regret between RECON-889k and
  DEC-889k) but does not reach the decision route.
* Prediction P1.3: RECON-889k's validation cross-entropy is still (slowly) improving at 6 000 steps; the
  decision arms plateau.
* Falsification: if DEC-133k ≥ RECON-131k or DEC-889k ≥ RECON-889k (CIs overlapping 0 or reversed) at
  N ≥ 20, the V1 headline is a capacity confound and this becomes the headline of the run.

### T3 — ε-rank oracle curve (box 1 h, oracle only)
* Prediction P3.1: the rank k needed to retain 90 % of the oracle-safe gain (k_90) increases monotonically
  with ε; at ε ≤ 0.02, k_90 ≤ 10; at ε = 0.40, k_90 ≥ 30.
* Prediction P3.2: the affine hull of the Nash set has dimension ≥ 1 (V1 found non-unique equilibria) and
  ≤ 50; the within-Nash component of g retains > 50 % of the gain at ε ≤ 0.02 and the orthogonal component's
  retained value grows roughly linearly with ε (slope fit R² > 0.9).
* Prediction P3.3: a Hoffman-type bound max_{S_ε} xᵀg − max_N xᵀg ≤ κ ε ‖g‖ holds for all tested (g, ε)
  with a single fitted κ; dist(x, N) vs Expl(x) is bounded by a line through the origin.
* Falsification: k_90 flat or decreasing in ε; orthogonal-component value not increasing in ε; bound
  violations.

### T2 — covariance gap and censoring (box 2 h)
* Unit test gate: with a factorized Dirichlet posterior and full revelation, A E[y] = A y_{E[q]} to 1e−10.
* Prediction P2.1: gap_g(H) and gap_reg(H) from the bank posterior correlate positively (Spearman ρ > 0.2,
  pooled) with the per-opponent decision advantage (RECON-889k regret − DEC-889k regret from T1); the
  largest gaps occur in STRUCTURED_CORRELATED.
* Prediction P2.2: with the opponent's card revealed after every hand, the decision advantage on
  UNSTRUCTURED_DIRICHLET collapses to |Δ| < 0.005 chips (CI includes 0) at every N; on
  STRUCTURED_CORRELATED it remains > 0.
* Design: censored vs uncensored compared at an equal 3 000-step budget (matched 131k/133k heads, 3 seeds;
  the uncensored arm uses T1's step-3000 checkpoints).
* Falsification: no correlation (ρ ≤ 0), or an advantage on the Dirichlet family that persists under full
  revelation.

### T4 — SPO+ loss (box 2.5 h)
* Gate: numerical checks that ℓ(ĝ, g) ≥ 0, ℓ(g, g) = 0, convexity along random segments, gradient
  2x*(2ĝ−g) − 2x*(g) matches finite differences.
* Arms (DEC-133k architecture, ε_train = 0.10): MSE only (= T1 DEC-133k), SPO+ only, MSE + λ·SPO+ with λ
  chosen on validation from {0.3, 1.0} (seed 0), 3 seeds for the chosen λ.  LP cost decides the batch
  treatment (exact SPO+ on a subset of each batch; reported).
* Prediction P4.1: SPO+ (alone or mixed) lowers safe regret at N ≥ 20 relative to MSE only (paired CI
  excluding 0) while g-NMSE is equal or worse.
* Falsification: no regret improvement, or improvement only with improved g-NMSE.

### T5 — empirical-Bayes hybrid (box 2 h)
* Arms: (a) BLEND λ_N ĝ_net + (1−λ_N) ĝ_EM with λ_N per N from validation; (b) COUNT FEATURES (per public
  state opponent action counts, per rank-infoset counts on showdown hands, showdown/fold counts) as extra
  encoder input, DEC-133k retrained, 3 seeds; (c) LEARNED PRIOR: tabular EM with prior mean = the
  reconstruction model's q̂ and concentration κ_N from validation (a simplification of "network outputs
  Dirichlet concentrations"; declared).
* Prediction P5.1: BLEND sits on or below the envelope of {bank posterior, decision net, tabular EM} at
  every N (within CI); (b) improves the decision net at N ≥ 100; (c) is at least as good as tabular EM at
  every N and better at N ≤ 50.
* Falsification: no hybrid reaches the envelope at N = 500 (i.e. none matches tabular EM there).

(Sections 1–7 below are filled in as each task completes.)
