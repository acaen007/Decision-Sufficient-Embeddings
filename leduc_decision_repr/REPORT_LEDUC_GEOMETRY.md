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

## 1. Verdict: NO-GO

**By the pre-registered rule this is NO-GO for training with an S_ε-shaped loss.**  The best safe-set metric,
the exact width W(δ), beats Euclidean error by only 0.041 in pooled Spearman (0.730 vs 0.689).  That is short
of the 0.15 predicted (P1) and of the 0.05 GO floor.  The quadratic forms that a loss could actually use
gain less: M_resp +0.034, M_support +0.016.  The metric also does not explain the V3 paradox (P2 fails):
M_support scores DEC-133k and RECON-889k identically (0.0513 vs 0.0514 at N = 500).  On these single-stream,
seed-0 points the regret gap between them is itself not significant.  The direction of the effect is
right: W beats ‖δ‖₂ in 28 of 36 method × N cells, both controls lose to ‖δ‖₂ (P4 holds), and weighting errors by
g-variance is much worse than Euclidean (0.469).  So the geometry is real but explains little of the
point-to-point variation in regret.  The one large exception is tabular EM, whose regret its Euclidean
error barely predicts (ρ = 0.27) but W predicts moderately (0.57).  The covariance ellipsoid is a poor
stand-in for S_ε.  It needs an inflation of 2.4 million to contain fresh support points, so the ellipsoid
bound is about 1 500× loose.

| pre-registered item | result | outcome |
|---|---|---|
| P1: ρ(M_support) − ρ(‖δ‖₂) ≥ 0.15 | 0.705 − 0.689 = +0.016 | falsified |
| P2: M_support orders DEC-133k worse than RECON-889k at N = 100, 500, ‖δ‖₂ the other way | M_support ties them (Δ ≈ 0.0000); ‖δ‖₂ reversed as predicted | fails |
| P3: ρ(W) ≥ ρ(M_support) | 0.730 ≥ 0.705 | holds |
| P4: controls do not beat ‖δ‖₂ | M_rand 0.630, M_gvar 0.469 < 0.689 | holds |
| NO-GO rule: best S_ε metric beats ‖δ‖₂ by < 0.05 and P2 fails | +0.041 (W); P2 fails | **NO-GO** |

## 2. Step 1 — the metrics

| matrix | rank (tol 1e−10) | condition no. (non-zero) | participation ratio | eigenvalues for 90 % of trace | trace share in the 192-dim span of training g |
|---|---|---|---|---|---|
| M_support (1 000 support points of S_ε) | 607 | 7.0e8 | 12.2 | 74 | 31 % |
| M_resp (oracle responses of 1 200 training opponents) | 404 | 1.2e9 | 5.7 | 22 | 49 % |
| M_gvar (training-g covariance, trace-matched) | 192 | 2.5e6 | 6.2 | 29 | 100 % |
| M_rand (M_support spectrum, random eigenvectors) | 607 | 7.0e8 | 12.2 | 74 | — |

* The realization-plan affine hull has dimension 624, and the 1 000 support points span 607 of it.  All
  1 200 support LPs (build and fresh) solved and passed the OpenSpiel audit, with max Expl − ε = 3.7e−11.
* **Inflation factor s = 2.40e6**, centred on the fresh points' mean (2.42e6 centred on the build points'
  mean).  The fresh points leave 5.7 % of their norm outside range(M_support).  The median fresh point has
  Mahalanobis q = 1 644, against 609 for the build points themselves.  The covariance of vertex samples puts
  almost no mass on many thin directions that other vertices still reach.
* The oracle responses x*(g) are not unique: a fresh solve and the cached response for the same opponent
  agree in value to 1e−15 but differ by up to 0.57 in realization plan.  So M_resp depends on which optimal
  vertex HiGHS returns.

## 3. Step 2–3 — does geometry predict regret?

10 800 points: 9 methods × N ∈ {5, 20, 100, 500} × 300 test opponents, first stream, ε = 0.10.
δ = g − ĝ is in raw chips.  W comes from 21 600 exact LPs, all solved and audited (max Expl − ε = 1.9e−10,
0 violations).

**Spearman ρ(R, metric).**

| metric | pooled | within-N ranks, pooled | mean within cell (36) | cells beating ‖δ‖₂ | N = 5 | N = 20 | N = 100 | N = 500 |
|---|---|---|---|---|---|---|---|---|
| ‖δ‖₂ | 0.689 | 0.665 | 0.639 | — | 0.668 | 0.657 | 0.654 | 0.680 |
| ‖δ‖_{M_support} | 0.705 | 0.685 | 0.665 | 18 / 36 | 0.725 | 0.684 | 0.654 | 0.677 |
| ‖δ‖_{M_resp} | 0.723 | 0.703 | 0.685 | 22 / 36 | 0.744 | 0.689 | 0.673 | 0.704 |
| W(δ) | **0.730** | **0.710** | **0.694** | **28 / 36** | **0.752** | **0.711** | **0.681** | **0.698** |
| ‖δ‖_{M_rand} (control) | 0.630 | 0.607 | 0.583 | 1 / 36 | 0.583 | 0.593 | 0.610 | 0.640 |
| ‖δ‖_{M_gvar} (control) | 0.469 | 0.433 | 0.415 | 0 / 36 | 0.380 | 0.405 | 0.428 | 0.519 |

**Per method** (pooled over N):

| method | ‖δ‖₂ | M_support | M_resp | W | M_gvar |
|---|---|---|---|---|---|
| DEC-133k | 0.730 | 0.720 | 0.741 | 0.742 | 0.492 |
| DEC-889k | 0.718 | 0.711 | 0.737 | 0.738 | 0.476 |
| RECON-131k | 0.749 | 0.737 | 0.750 | 0.764 | 0.537 |
| RECON-889k | 0.739 | 0.732 | 0.745 | 0.759 | 0.506 |
| RECON-JAC-889k | 0.699 | 0.712 | 0.725 | 0.736 | 0.482 |
| MSE+0.3·SPO+ | 0.739 | 0.710 | 0.736 | 0.733 | 0.517 |
| tabular EM | **0.271** | 0.527 | 0.532 | **0.566** | 0.072 |
| bank posterior | 0.750 | 0.744 | 0.774 | 0.777 | 0.541 |
| learned-prior EM | 0.684 | 0.674 | 0.697 | 0.695 | 0.512 |

W fails to beat ‖δ‖₂ in only eight cells.  Six of them are the decision-trained heads (DEC-133k, DEC-889k,
MSE+0.3·SPO+) at N = 100 and N = 500; the other two are RECON-131k and learned-prior EM at N = 100.

**Method-level ranking** (Kendall τ between mean metric and mean regret over the 9 methods):

| N | ‖δ‖₂ | M_support | M_resp | W | M_rand | M_gvar |
|---|---|---|---|---|---|---|
| 5 | 0.44 | 0.67 | 0.72 | 0.78 | 0.67 | 0.17 |
| 20 | 0.44 | 0.72 | 0.61 | 0.72 | 0.44 | 0.44 |
| 100 | 0.50 | 0.56 | 0.56 | 0.56 | 0.56 | 0.44 |
| 500 | 0.44 | 0.61 | 0.61 | 0.50 | 0.39 | 0.39 |

At the method level the S_ε metrics rank methods better than ‖δ‖₂ at every N.  With nine methods,
though, one swapped pair moves τ by 0.056, and M_rand does as well as M_support at N = 5 and 100.  The
largest misranking is tabular EM, and the geometry does not fix it: at N = 500 it has the second-lowest regret
(0.075) but the largest ‖δ‖₂ (0.695), the second-largest W (0.574) and the third-largest ‖δ‖_{M_support}.  Its
within-method correlation improves a lot under W (0.27 → 0.57), but its level does not: its errors are wide
in S_ε terms and still cost little.

**Paradox pairs** (mean over the 300 points; paired bootstrap 95 % CI of a − b).

| pair | N | R | ‖δ‖₂ | ‖δ‖_{M_support} | W |
|---|---|---|---|---|---|
| DEC-133k − RECON-889k | 20 | +0.0044 [−0.0037, 0.0124] | −0.011 [−0.029, 0.006] | −0.0006 [−0.0023, 0.0010] | +0.001 [−0.013, 0.015] |
| | 100 | +0.0054 [−0.0027, 0.0127] | −0.014 [−0.029, 0.002] | −0.0000 [−0.0017, 0.0016] | +0.007 [−0.008, 0.020] |
| | 500 | +0.0049 [−0.0027, 0.0122] | −0.014 [−0.029, −0.002] | −0.0001 [−0.0016, 0.0015] | +0.008 [−0.004, 0.022] |
| DEC-133k − RECON-131k | 20 | +0.0042 [−0.0036, 0.0119] | −0.022 [−0.040, −0.006] | −0.0010 [−0.0024, 0.0005] | −0.000 [−0.013, 0.013] |
| | 100 | +0.0035 [−0.0045, 0.0107] | −0.035 [−0.053, −0.019] | −0.0011 [−0.0024, 0.0004] | −0.004 [−0.016, 0.009] |
| | 500 | +0.0025 [−0.0046, 0.0090] | −0.037 [−0.053, −0.022] | −0.0009 [−0.0023, 0.0004] | −0.004 [−0.015, 0.007] |

* DEC-133k vs RECON-889k: ‖δ‖₂ gets the order wrong at N ≥ 20, significantly so at N = 500.  M_support
  and M_resp call it a tie.  **W gets the order right at N = 20, 100 and 500**, though not significantly.
  P2 as pre-registered (M_support) fails.
* DEC-133k vs RECON-131k: ‖δ‖₂ strongly favours DEC-133k (−0.035 at N = 100, CI excludes 0), yet regret is
  equal or slightly worse.  All S_ε metrics shrink that gap to ≈ 0 and none reverse it.  The geometry
  removes the spurious Euclidean advantage but does not recover the regret order.
* For the two pairs where the regret difference is large (DEC-889k vs RECON-889k, RECON-JAC-889k vs
  DEC-889k at N ≥ 100), every metric, ‖δ‖₂ included, orders them correctly.

**Bounds.**
* R ≤ W(δ) held at every point: max(R − W) = −0.021, 0 violations.  The theorem check passes.
* The inflated ellipsoid bound 2·sqrt(s·δᵀM_support δ) held everywhere (0 % violations) but is useless as a
  quantity: median R / bound = 6.6e−4.  Without inflation, 2·‖δ‖_{M_support} is exceeded by R at 51 % of
  points.
* **Tightness:** median R / W = 0.19 (p10 0.06, p90 0.37, max 0.84), falling slightly with N (0.20 → 0.17).
  Median W / (2‖δ‖₂) = 0.44.  Deployed regret is typically a fifth of the worst case the error direction
  allows; the gap between R ≤ δᵀ(x* − x̂) and W is where most of the unexplained variance lives.

## 4. Stretch items

**Realizability (DEC-889k).**  Each ĝ was projected onto {A y : F y = f, y ≥ 0} (Euclidean QP, Clarabel, all
solved), and the safe LP was re-solved on the projection.  First stream, 300 test opponents:

| N | regret, original ĝ | regret, projected ĝ | change [95 % CI] | improved / worse | ‖projection − ĝ‖₂ | ‖δ‖₂ before → after |
|---|---|---|---|---|---|---|
| 20 | 0.1362 | 0.1346 | −0.0016 [−0.0028, −0.0006] | 58 % / 42 % | 0.027 | 0.716 → 0.715 |
| 500 | 0.1018 | 0.1008 | −0.0009 [−0.0029, +0.0009] | 49 % / 51 % | 0.036 | 0.540 → 0.537 |

The decision head's ĝ is already almost realizable: the projection moves it 0.03 chips, against an error
of 0.54–0.72.  Enforcing realizability buys at most 0.002 chips.  This does not support the V3 §1 conjecture
that unrealizable errors are what make the direct-g route lose.  All 600 re-solved deployed strategies pass
the audit (max Expl − ε = 6.8e−13).

**Small ε.**  Mean oracle safe gain over the mean-g response, 300 test opponents (ε ≤ 0.01 computed here;
ε > 0.01 from T3, same opponents and baseline):

| ε | 1e−4 | 1e−3 | 3e−3 | 0.01 | 0.02 | 0.05 | 0.10 | 0.20 | 0.40 |
|---|---|---|---|---|---|---|---|---|---|
| gain (chips) | 0.0094 | 0.0227 | 0.0364 | 0.0638 | 0.0968 | 0.1488 | 0.1946 | 0.2869 | 0.4230 |
| local log-log slope to the next ε | 0.38 | 0.43 | 0.47 | 0.60 | 0.47 | 0.39 | 0.56 | 0.56 | — |

The curve does not turn linear down to ε = 1e−4.  If anything it flattens, because the gain has a
non-zero intercept: at ε = 0, T3 found 0.0051 from choosing within the 43-dimensional Nash set.  After
subtracting that intercept the excess gain still scales as ε^0.5–0.6 between 1e−4 and 1e−2.  For a single
opponent the safe value is a concave piecewise-linear function of ε (a parametric right-hand side), so
it must be linear below that opponent's first breakpoint.  The population mean stays sub-linear down to
1e−4, so those breakpoints are spread over several decades below 0.01.  The mean initial slope is at least
43 chips per unit ε.  All 2 400 strategies audited, max Expl − ε = 1.3e−10.

## 5. What surprised me

* **Weighting errors by g-variance is actively harmful.**  ρ = 0.47 against 0.69 for plain Euclidean error,
  losing in all 36 cells.  This is the metric a variance-oriented (PCA, reconstruction-MSE) view implicitly
  optimizes.  It agrees with T3: the decision value lives in low-variance directions of g.
* **For tabular EM, Euclidean error is almost uninformative** (within-cell ρ 0.03–0.28).  The S_ε metrics
  raise it to 0.30–0.58 (W: 0.40–0.58).  EM's errors sit where S_ε is wide, and they still cost little
  on average.  It is the method whose regret level the geometry fails to explain.
* **For the decision-trained heads at N ≥ 100, W does not beat Euclidean error.**  These are six of the
  eight cells where W loses.  Whatever these heads learned, their residual errors are not shaped the way
  the width argument would predict.
* **The bound chain is loose at every link.**  R is on median 19 % of W; the uninflated covariance
  ellipsoid is not a bound at all (51 % violations); the inflated one is about 1 500× loose.  S_ε is far
  from ellipsoidal: 607 of 624 affine dimensions are reached, with a condition number of 7e8.
* **Oracle responses are faces, not points.**  Equal-value optimal vertices can lie 0.57 apart, so any
  "target x*" loss would need a tie-breaking rule.

## 6. Bookkeeping

* Scripts: `leduc_decision_repr/geometry_diag.py` (`build`, `score`, `analyze`, `stretch`).  Outputs:
  `outputs/geometry/{build,score,analysis,stretch}.json` and `points.npz` (the 10 800 scored points).
  `M.npz` (37 MB) is regenerated deterministically by `build` (seed 0) and is not committed.
* Compute: build 1.3 min; scoring 22 min (21 600 LPs and audits on 4 cores); stretch 2 min.  About 40 min
  of wall-clock including analysis, well inside the 2 h budget.  No training.
* Audits: every LP solution in this diagnostic was checked with OpenSpiel's best response, 25 800 in total
  (1 200 support, 21 600 width, 600 projection re-solves, 2 400 small-ε).  Max Expl − ε = 1.9e−10,
  0 violations, 0 LP failures.
* Deviation: the stretch stage's realizability audit stored Expl instead of Expl − ε while running and
  printed a spurious "300 violations".  The saved fields were corrected post hoc (true max excess 6.8e−13;
  the correction is noted in `stretch.json`), and the code is fixed.
* Scope limits: seed 0 and one stream per opponent.  The DEC-133k vs RECON-889k regret gap that motivated
  P2 is significant with 3 seeds and 8 streams (V3 §1), but not on these 300 single-stream points.  So
  P2's failure is partly a power issue; no S_ε metric reverses the Euclidean order significantly either.
