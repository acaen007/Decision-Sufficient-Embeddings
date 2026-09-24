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

## 1. T1 — confound fixes: parameter matching, convergence, reach-weighted reconstruction

*Status: DEC-133k (3 seeds) and RECON-889k (3 seeds) evaluated at 04:09 UTC; RECON-JAC-889k seed 0 is trained
and queued for evaluation; RECON-REACH-889k is queued for training.  Those two arms are appended to this
section when they land; the verdicts on P1.1 and P1.3 do not depend on them.*

**Headline.**  The pre-registered falsification clause fires on one of the two matched pairs.  At ≈ 131k head
parameters the decision route has **no advantage** over reconstruction: DEC-133k − RECON-131k is −0.006
chips at N = 5 (CI excludes 0) and +0.000 … +0.003 at N ≥ 20 with every CI covering 0.  At ≈ 889k head
parameters the advantage **survives, weakened**: DEC-889k − RECON-889k is −0.003 … −0.009 chips at ε = 0.10
(CI excludes 0 at 6 of 7 N; at N = 50 it is [−0.008, +0.001]), −0.008 … −0.012 at ε = 0.20 (every CI excludes
0) and −0.002 … −0.005 at ε = 0.05 (CIs exclude 0 only at N ≤ 10).  That is one fifth to one third of the
−0.010 … −0.035 reported in V1 against the smaller reconstruction head.  So the V1 headline was largely a
capacity confound.  What remains is an *interaction*: both routes gain from head capacity, but the decision
route gains about twice as much (DEC-889k over DEC-133k: 0.004 … 0.015, growing with N; RECON-889k over
RECON-131k: 0.001 … 0.007), while the reconstruction cross-entropy floor is the same for both head sizes
(0.634–0.635 vs 0.635–0.637).

**Exact parameter counts** (encoder shared by every arm: 618 752).

| arm | head | head params | total | seeds | source |
|---|---|---|---|---|---|
| DEC-133k | MLP hidden 100 → standardized g | 133 393 | 752 145 | 3 | V3 |
| RECON-131k | infoset-conditioned softmax, hidden 256 | 131 331 | 750 083 | 3 | V1 |
| DEC-889k | MLP hidden 512 | 889 413 | 1 508 165 | 3 | V1 |
| RECON-889k | hidden 834 | 889 089 | 1 507 841 | 3 | V3 |
| RECON-JAC-889k | hidden 834, Jacobian-norm infoset weights | 889 089 | 1 507 841 | 1 (evaluation queued) | V3 |
| RECON-REACH-889k | hidden 834, chance × reach weights | 889 089 | 1 507 841 | queued | V3 |

**Training to convergence.**  Every V3 run hit the 6 000-step cap; none triggered the patience rule (6
validations = 1 500 steps).  Relative improvement of the validation loss over the last 1 500 steps: DEC-133k
2.4 / 1.9 / 1.7 % (seeds 0/1/2; best checkpoint at step 6000, 5750, 6000); RECON-889k 0.35 / 0.19 / 0.45 %
(best at 6000, 5750, 6000; best validation cross-entropy 0.6352, 0.6352, 0.6344).  The V1 runs (same 6 000
steps, no early stopping): DEC-889k 0.7 / 1.3 / 1.0 %, RECON-131k 0.3 / 0.1 / 0.3 % (0.6367, 0.6367, 0.6350).
So: **the reconstruction arms have plateaued** (their validation cross-entropy sits at 0.634–0.637
irrespective of head size, i.e. at what looks like the posterior-entropy floor of the tokenized histories),
whereas **the decision arms are still improving** slowly at the cap, the smaller one more than the larger
one.  If anything, longer training would widen the DEC-889k gap and might create a DEC-133k one; it cannot
rescue the reconstruction route.  Validation curves: figure V3-T1(c).

**Safe regret at ε = 0.10 (mean over 300 test opponents; 8 streams; seeds averaged).**

| method | N=5 | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|
| DEC-133k (3 seeds) | 0.1710 | 0.1552 | 0.1428 | 0.1316 | 0.1260 | 0.1228 | 0.1203 |
| RECON-131k (V1, 3 seeds) | 0.1771 | 0.1574 | 0.1424 | 0.1284 | 0.1241 | 0.1203 | 0.1180 |
| DEC-889k (V1, 3 seeds) | 0.1669 | 0.1488 | 0.1346 | 0.1201 | 0.1118 | 0.1079 | 0.1051 |
| RECON-889k (3 seeds) | 0.1763 | 0.1552 | 0.1389 | 0.1231 | 0.1177 | 0.1134 | 0.1116 |
| train-bank posterior | 0.1583 | 0.1436 | 0.1361 | 0.1282 | 0.1269 | 0.1233 | 0.1217 |
| tabular EM (uniform prior) | 0.1960 | 0.1800 | 0.1654 | 0.1348 | 0.1138 | 0.0935 | 0.0748 |

Seed spread (max − min over seeds of the per-N mean): DEC-133k 0.001–0.004, DEC-889k 0.001–0.003, RECON-131k
0.001–0.003, RECON-889k 0.002–0.006 (per seed at N = 500: 0.1151 / 0.1092 / 0.1104) — the RECON-889k spread
is of the same size as its difference to DEC-889k.  AUC over log N at ε = 0.10: DEC-133k 0.1366, RECON-131k
0.1358, DEC-889k 0.1256, RECON-889k 0.1310.

**Paired differences (a − b; negative = a better; 95 % paired bootstrap over opponents).**

| pair | ε | N=5 | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|---|
| DEC-133k − RECON-131k | 0.05 | −0.003 [−0.007, −0.000] | −0.001 [−0.004, 0.002] | +0.001 [−0.002, 0.004] | +0.002 [−0.001, 0.005] | +0.002 [−0.002, 0.005] | +0.003 [−0.002, 0.006] | +0.002 [−0.002, 0.006] |
| | 0.10 | −0.006 [−0.011, −0.002] | −0.002 [−0.006, 0.002] | +0.000 [−0.004, 0.005] | +0.003 [−0.002, 0.008] | +0.002 [−0.004, 0.007] | +0.003 [−0.003, 0.008] | +0.002 [−0.004, 0.009] |
| | 0.20 | −0.008 [−0.014, −0.002] | −0.006 [−0.011, −0.001] | −0.003 [−0.008, 0.003] | −0.001 [−0.009, 0.006] | −0.003 [−0.011, 0.005] | −0.001 [−0.011, 0.007] | −0.003 [−0.013, 0.006] |
| DEC-889k − RECON-889k | 0.05 | −0.005 [−0.008, −0.002] | −0.004 [−0.007, −0.001] | −0.002 [−0.005, 0.000] | −0.002 [−0.006, 0.001] | −0.003 [−0.007, 0.001] | −0.002 [−0.006, 0.002] | −0.003 [−0.007, 0.001] |
| | 0.10 | −0.009 [−0.014, −0.004] | −0.006 [−0.010, −0.003] | −0.004 [−0.008, −0.001] | −0.003 [−0.008, 0.001] | −0.006 [−0.011, −0.001] | −0.006 [−0.011, −0.000] | −0.007 [−0.013, −0.001] |
| | 0.20 | −0.012 [−0.019, −0.006] | −0.010 [−0.015, −0.006] | −0.008 [−0.013, −0.003] | −0.008 [−0.014, −0.002] | −0.011 [−0.019, −0.004] | −0.011 [−0.019, −0.003] | −0.012 [−0.020, −0.003] |
| DEC-133k − DEC-889k | 0.10 | +0.004 [0.002, 0.006] | +0.006 [0.004, 0.009] | +0.008 [0.006, 0.011] | +0.012 [0.008, 0.015] | +0.014 [0.010, 0.019] | +0.015 [0.011, 0.020] | +0.015 [0.011, 0.020] |
| RECON-889k − RECON-131k | 0.10 | −0.001 [−0.003, 0.001] | −0.002 [−0.004, −0.001] | −0.004 [−0.006, −0.002] | −0.005 [−0.008, −0.003] | −0.006 [−0.009, −0.004] | −0.007 [−0.010, −0.004] | −0.007 [−0.010, −0.004] |

**Per family (ε = 0.10; diff [95 % CI]).**

| pair | family | N=5 | 20 | 100 | 500 |
|---|---|---|---|---|---|
| DEC-133k − RECON-131k | NASH_LOGIT_PERTURB | −0.009 [−0.014, −0.005] | +0.006 [0.002, 0.011] | +0.014 [0.006, 0.022] | +0.019 [0.008, 0.029] |
| | NASH_RANDOM_MIX | +0.008 [0.001, 0.015] | +0.009 [0.004, 0.015] | +0.006 [0.001, 0.011] | +0.004 [−0.002, 0.011] |
| | STRUCTURED_CORRELATED | −0.017 [−0.030, −0.003] | −0.011 [−0.020, −0.002] | +0.002 [−0.008, 0.013] | +0.003 [−0.007, 0.013] |
| | UNSTRUCTURED_DIRICHLET | −0.006 [−0.015, 0.003] | −0.003 [−0.016, 0.008] | −0.014 [−0.032, 0.001] | −0.017 [−0.037, 0.000] |
| DEC-889k − RECON-889k | NASH_LOGIT_PERTURB | −0.013 [−0.020, −0.007] | −0.005 [−0.010, −0.001] | −0.001 [−0.009, 0.005] | +0.002 [−0.008, 0.011] |
| | NASH_RANDOM_MIX | +0.006 [−0.001, 0.014] | +0.006 [0.002, 0.011] | +0.003 [−0.001, 0.007] | +0.003 [−0.004, 0.008] |
| | STRUCTURED_CORRELATED | −0.019 [−0.033, −0.004] | −0.011 [−0.018, −0.003] | −0.007 [−0.018, 0.002] | −0.008 [−0.018, 0.002] |
| | UNSTRUCTURED_DIRICHLET | −0.012 [−0.022, −0.001] | −0.008 [−0.016, −0.000] | −0.018 [−0.033, −0.005] | −0.022 [−0.040, −0.004] |

The pooled null result at 131k is a cancellation: the decision head is *worse* on the two near-Nash families
at N ≥ 20 (up to +0.019 on NASH_LOGIT_PERTURB at N = 500) and *better* on STRUCTURED_CORRELATED at small N and
on UNSTRUCTURED_DIRICHLET at large N.  At 889k the near-Nash families are a tie or a small reversal at
N ≥ 100 and the whole surviving advantage comes from the two families that are far from equilibrium, growing
with N on the Dirichlet family (−0.022 at N = 500).  This is the same family pattern as V1 and V2.

**g accuracy is not what separates the routes.**  Test g-NMSE at N = 5 … 500: DEC-133k 0.79 → 0.42, RECON-131k
0.82 → 0.50, DEC-889k 0.79 → 0.38, RECON-889k 0.81 → 0.46.  DEC-133k predicts g *more* accurately than
RECON-889k at every N (0.42 vs 0.46 at N = 500) yet deploys *worse* (0.120 vs 0.112), and it matches
RECON-131k in regret while beating it by 0.08 in NMSE.  A reconstruction ĝ = A y_{q̂} is always the value
vector of some legal opponent policy; the decision head's ĝ is not, and the ε-safe LP appears to penalize
unrealizable errors more than realizable ones of larger norm.  Combined with T3 (value concentrated in
low-variance directions of g) this says the decision route's edge, where it exists, comes from what the extra
capacity is spent on, not from a smaller g error.

**Verdict against the pre-registration.**
* P1.1 (advantage shrinks but survives matching in both directions, CIs excluding 0 at every N ≥ 20):
  **falsified for the 131k pair** (DEC-133k ≥ RECON-131k at every N ≥ 20, CIs cover 0; the reversal is
  significant on the near-Nash families) and **only partly confirmed for the 889k pair** (DEC-889k <
  RECON-889k at every N and ε; at ε = 0.10 the CI excludes 0 at 6 of 7 N but not at N = 50; at ε = 0.05 it
  excludes 0 only at N ≤ 10; at ε = 0.20 at every N).  The falsification clause names this the headline of
  the run.
* P1.2 (Jacobian-weighted reconstruction closes part of the gap): pending (RECON-JAC-889k seed 0 trained;
  validation g-NMSE at N = 500 0.40 vs 0.53 for unweighted RECON-889k at the same step — the weighting does
  make ĝ more accurate; whether it deploys better is the open question).
* P1.3 (RECON-889k still improving at 6 000 steps, decision arms plateau): **falsified in direction** — the
  reconstruction arms plateaued (0.2–0.45 % over the last 1 500 steps) and the decision arms were still
  improving (1.7–2.4 %).

**Deliverable figure.**  `figures/figV3_T1_matched.{png,pdf}`: (a) regret vs N for the matched arms with the
bank posterior and tabular EM; (b) paired differences with CIs; (c) validation curves of the V3 runs.
Safety: every deployed strategy of every arm audited; max Expl − ε = 6.7e−10 over the 403 200 LPs of the
six V3 T1 runs (table in §6).  Time: 1.6–3.0 h per run, 4 in parallel; the T1 box (2.5 h) was exceeded
because the RECON-889k seeds 1–2 and the weighted arms had to queue behind round A (declared in §8).

## 3. T3 — ε-rank oracle curve, Nash-hull component split, κ (oracle only; 33 min of the 1 h box)

**What was run.** PCA basis of g fitted on the 1 200 training opponents' true g (the centred training g
matrix has rank 192 of 1 093 sequence dimensions).  For each of the 300 test opponents, each
ε ∈ {0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40} and each rank k ∈ {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96,
128, 192, 256}: solve the exact ε-safe LP for the rank-k truncation ĝ_k = ḡ + P_k(g − ḡ), evaluate it against
the true g, and report the fraction of the oracle-safe gain retained, gain = gᵀx*(g) − gᵀx*(ḡ) (relative to
the response to the population mean ḡ; opponents with gain < 0.02 chips at that ε are excluded from
fractions, pooled fraction = ratio of means).  Nash-set hull: 400 random directions d, ε = 0 LP each, SVD of
the solution points.  Component split: g decomposed into the part whose deviation from ḡ lies in the hull's
direction span (in x-space) and the orthogonal rest; each part solved and scored on the true g.  κ: (i)
dist(x, N) via an exact projection QP (ε = 0 constraints, quadratic objective) against Expl(x), on 60 test
opponents × 6 ε safe responses plus 60 random behavioural strategies (420 points); (ii) the value bound
(V_ε(g) − V_0(g)) / (ε ‖g‖₂) on 100 test opponents × 6 ε.
Every one of the 40 767 solved strategies was audited with the OpenSpiel best response: max violation
1.1e−9 ≤ 1e−7, 0 LP failures.  Figure: `figures/figV3_T3_eps_rank.{png,pdf}` (data in
`figures/figV3_T3_eps_rank_data.json`; raw arrays in `outputs/t3_eps_rank/`).

**Result 1 — rank needed (pooled, 300 test opponents).**

| ε | valid opp. | oracle gain (chips) | k_50 | k_80 | k_90 | variance captured at k_90 |
|---|---|---|---|---|---|---|
| 0 | 26 | 0.0051 | 32 | 48 | 128 | 99.9 % |
| 0.01 | 218 | 0.064 | 24 | 96 | 128 | 99.9 % |
| 0.02 | 266 | 0.097 | 16 | 96 | 128 | 99.9 % |
| 0.05 | 295 | 0.149 | 16 | 96 | 128 | 99.9 % |
| 0.10 | 294 | 0.195 | 16 | 64 | 128 | 99.9 % |
| 0.20 | 300 | 0.287 | 12 | 64 | 96 | 99.3 % |
| 0.40 | 300 | 0.423 | 12 | 64 | 96 | 99.3 % |

Retained fraction at ε = 0.10 by rank: k=1 0.06, 2 0.09, 4 0.26, 8 0.41, 16 0.58, 32 0.65, 64 0.81, 96 0.89,
128 0.97, 192 1.00.  The 29-component subspace that holds 90 % of the *variance* of g (V1 §11) retains only
≈ 63 % of the *decision value* at ε = 0.10 (≈ 60 % at ε = 0.02); 90 % of the value needs 96–128 components,
i.e. directions that together carry < 1 % of the variance.  Per family (k_90, ε > 0): NASH_LOGIT_PERTURB
128 → 96 → 64 as ε grows from 0.10 to 0.40; NASH_RANDOM_MIX 128 at every ε; STRUCTURED_CORRELATED 96 at
every ε; UNSTRUCTURED_DIRICHLET 128 at every ε.  The per-opponent mean fraction (rather than pooled) is
lower still at small ε (k=32: 0.46 at ε = 0.01, 0.55 at ε = 0.10).

**Result 2 — Nash-set hull and component split.**  The affine hull of the ε = 0 solutions has dimension
43 (identical at tolerances 1e−6 and 1e−4 of the top singular value; all hull points have exploitability
≤ 1e−11).  Responding to the within-hull component of g alone retains 100 % of the gain at ε = 0 (by
construction) but ≈ 0 or slightly negative value at every ε > 0 (fraction −0.06, −0.04, −0.11, −0.20, −0.18,
−0.05 at ε = 0.01 … 0.40; mean value −0.002 … −0.021 chips).  Responding to the orthogonal component retains
0.84, 0.82, 0.70, 0.72, 0.81, 0.89 of the gain.  The orthogonal component's value is not linear in ε: a line
through the origin fits with slope 1.08 and R² = 0.77, whereas a power law fits with exponent 0.52 (log-log
R² = 0.996).  The full oracle gain itself follows gain ≈ 0.65 · ε^0.50 (log-log R² = 0.997) over
ε ∈ [0.01, 0.40]: a square-root law, so each doubling of the safety budget buys ≈ 41 % more exploitation
value.

**Result 3 — κ.**  dist(x, N)/Expl(x) ranges from 2.1 to 359 (median 49) over the 420 points; the max
ratio is attained by an ε = 0.01-safe response that sits 3.6 (L2, realization plan) from the Nash set.  The
value constant (V_ε − V_0)/(ε‖g‖) has max 41, 26, 15, 9.6, 5.7, 3.5 (medians 9.5, 7.5, 5.1, 3.7, 2.7, 1.7)
at ε = 0.01, 0.02, 0.05, 0.10, 0.20, 0.40.  A single κ = 41 (or κ_dist = 359) makes both bounds hold on
every tested point — necessarily, since it is the maximum ratio — but the constant is loose by an order of
magnitude across the ε range because of the √ε scaling.

**Verdict against the pre-registration.**
* P3.1 (k_90 increases with ε; ≤ 10 at ε ≤ 0.02; ≥ 30 at ε = 0.40): **falsified in direction.**  k_90 is
  flat at 128 for ε ≤ 0.10 and *decreases* to 96 for ε ≥ 0.20; k_50 decreases from 24–32 to 12.  The
  ε-rank signature exists, but it runs the other way: the safe response at small ε depends on *more*
  directions of g, not fewer, and those directions are the low-variance ones.
* P3.2: hull dimension 43 is in the predicted range (confirmed).  Within-hull component retaining > 50 % at
  ε ≤ 0.02: **falsified** (≈ −5 %).  Orthogonal value growing linearly (R² > 0.9): **falsified** (R² 0.77);
  the growth is ∝ ε^0.52.
* P3.3: the bounds hold with fitted constants, but the constants are not "single": the dist ratio spreads
  over two orders of magnitude and the value ratio shrinks 12× from ε = 0.01 to 0.40.  Recorded as
  "holds but uninformative"; the falsification criterion (bound violations) is not met.

**What this means for the representation question.**  V1 reported that g is more compressible than q
(29 vs 83 PCA components for 90 % variance) and took that as support for a low-dimensional decision-relevant
code.  T3 shows the variance ranking is the wrong ranking: the value of the ε-safe response is carried
mostly by low-variance directions of g, and the Nash-hull directions (where the ε = 0 value lives) are
essentially worthless once ε > 0.  A decision-trained head therefore has to allocate capacity to directions a
variance-weighted reconstruction objective treats as noise, which is consistent with the V1/V2 finding that
reconstruction-trained latents lose out at ε > 0 even when their q̂ is more accurate.  The √ε gain law also
sets the scale for every later comparison: at ε = 0.10 the whole oracle gain is 0.195 chips/hand, so the
0.010–0.035 differences between methods reported in V1 are 5–18 % of what is attainable.

**Deviation.**  The stored PCA spectrum in `t3_results.json` is truncated to 64 entries (storage only; all
computations used the full 192-dimensional basis).  Panel (d) of the figure was switched to log-log axes
after the first render because the fitted κ_dist line hid the points; nothing else was changed.
