# REPORT_LEDUC_GEOMETRY — Does the safe-set geometry explain safe regret? (go/no-go diagnostic)

*Started 2026-09-24 ~12:55 UTC.  No training: reuses the V3 pipeline, the exact ε-safe LP, the saved
test-split predictions (`outputs/eval/test/ghat_*.npy`, raw chip units) and the saved deployed-strategy
values (`solve_*.npz`).  ε = 0.10 only.*

## 0. Pre-registration (written before any metric was computed; not edited afterwards)

**Theory being tested.**  For a prediction error δ = g − ĝ and x̂ = argmax_{S_ε} ĝᵀx, the deployed safe regret
R = gᵀx*(g) − gᵀx̂ ≤ δᵀ(x*(g) − x̂) ≤ W(δ) = max_{x,x'∈S_ε} δᵀ(x − x'), and for any ellipsoid
{c + (sM)^{1/2}v : ‖v‖ ≤ 1} that contains S_ε, W(δ) ≤ 2·sqrt(s·δᵀMδ).  So errors should cost in proportion to
the width of S_ε in their direction, not their Euclidean size.

**Data points.**  One point = (method, N, opponent), first stream only.  Methods (seed 0 / single run):
DEC-133k, DEC-889k (V1 decision), RECON-131k (V1 recon), RECON-889k, RECON-JAC-889k, MSE+0.3·SPO+ (seed 0),
tabular EM (uniform prior), train-bank posterior, learned-prior EM (single-seed prior, `HYB_PRIOR_EM_S0`).
N ∈ {5, 20, 100, 500}; all 300 test opponents → 9 × 4 × 300 = 10 800 points.  R is taken from the saved
deployed values (V_oracle − gᵀx̂_deployed); δ is computed in raw chips from the saved ĝ exactly as passed to
the LP.

**Metrics.**  ‖δ‖₂; ‖δ‖_M = sqrt(δᵀMδ) for
* M_support: covariance of the S_ε support points argmax_{S_ε} uᵀx for 1 000 directions u (250 random
  Gaussian and the top-250 PCA directions of training-opponent g, each with both signs);
* M_resp: covariance of the oracle responses x*(g_i) of the 1 200 training opponents;
* controls, rescaled to M_support's trace: M_rand (M_support's eigenvalues, random orthonormal
  eigenvectors) and M_gvar (covariance of training g);
* exact W(δ) (two LPs per point: max and min of δᵀx over S_ε).
For the bound check only, s = smallest inflation of M_support such that 200 fresh support points (100 new
Gaussian directions and 100 random PCA-weighted combinations of training-g principal directions) lie in
the ellipsoid centred at their mean (pseudo-inverse; residual outside range(M_support) reported).
None of this uses the test split except the scored points themselves.

**Primary statistic.**  Spearman ρ(R, metric) pooled over the 10 800 points.  Because both R and every
error metric fall with N, pooled ρ is inflated by the N-trend for every metric alike; the mean within-cell ρ
(per method × N, 36 cells) is reported alongside as a secondary, descriptive statistic.

**Predictions.**
* P1: pooled ρ(R, ‖δ‖_{M_support}) − ρ(R, ‖δ‖₂) ≥ 0.15.
* P2: at N = 100 and at N = 500, mean ‖δ‖_{M_support} ranks DEC-133k worse than RECON-889k (the same order as
  their mean R on these points), while mean ‖δ‖₂ ranks them the other way.
* P3: pooled ρ(R, W) ≥ pooled ρ(R, ‖δ‖_{M_support}).
* P4: neither control (M_rand, M_gvar) exceeds pooled ρ(R, ‖δ‖₂).

**Decision rule.**
* NO-GO (headline): the best S_ε metric (max over ‖δ‖_{M_support}, ‖δ‖_{M_resp}, W) beats ‖δ‖₂ by < 0.05 in
  pooled ρ **and** P2 fails.
* GO (for training with an S_ε-shaped loss): P1 holds, or the best S_ε metric beats ‖δ‖₂ by ≥ 0.05 and P2
  holds.
* Anything else: WEAK / INCONCLUSIVE, stated as such.

**Hard checks.**  R ≤ W(δ) is a theorem: any violation above 1e−6 chips is a bug and stops the analysis until
fixed.  R ≤ 2·sqrt(s·δᵀM_support δ) is not guaranteed (s is fitted on 200 points); its violation rate is
reported as a measure of how well the ellipsoid encloses S_ε.  Every strategy produced by any LP in this
diagnostic is audited with OpenSpiel's best response (Expl ≤ ε + 1e−7).
