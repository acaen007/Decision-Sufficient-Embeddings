# REPORT_LEDUC_V3 — Overnight follow-up experiments T1–T5

*Run started 2026-09-23 21:17 UTC on the V1 pipeline (same 300 held-out opponents, same exact ε-safe LP,
same OpenSpiel audit, same train bank / bank posterior / tabular EM).  Nothing below is tuned on the
evaluation opponents; the 150 validation opponents (carved from the training population in V1) are used for
every selection.*

## Summary (one paragraph; written 07:30 UTC at the end of the 10 h box — tail arms still running, see §7)

The V1 decision-over-reconstruction advantage was mostly a capacity confound and, where it survives, it is a
statement about *what the loss weights*, not about the decision objective: at matched ≈ 131k-parameter heads
the two routes are indistinguishable (DEC-133k − RECON-131k = +0.000 … +0.003 chips at N ≥ 20, CIs cover 0);
at matched ≈ 889k heads a weakened advantage remains (−0.003 … −0.009 at ε = 0.10, one fifth to one third of
V1's); and a reconstruction head whose per-infoset cross-entropy is weighted by the Jacobian norm of g
overtakes the decision head at N ≥ 20 at every ε (one seed, two more running) while also fitting g best.
The strongest new method of the run is not a representation at all but an empirical-Bayes hybrid: tabular
EM with the reconstruction network's q̂ as a κ = 3 Dirichlet prior sits *below* the envelope of {bank
posterior, decision net, tabular EM} at every N ≥ 20 (by 0.014–0.031 chips, all CIs exclude 0; 0.0525 at
N = 500 vs 0.075 for EM and 0.105 for DEC-889k), with controls showing the gain is the Bayesian update, not
ensembling or head size.  The oracle analysis inverted the pre-registered ε-rank signature — the rank of g
needed to retain 90 % of the safe gain is 128 for ε ≤ 0.10 and *falls* to 96 at ε ≥ 0.20, so the value lives
in the low-variance directions of g — and found a clean √ε law for the safe gain (0.65·ε^0.50, R² 0.997)
with a 43-dimensional Nash hull whose directions are worthless once ε > 0.  SPO+ mixed at the
validation-selected λ = 0.3 lowers regret by 0.004–0.009 at N ≥ 20 with a slightly *worse* g-NMSE (the
pre-registered signature; one seed), while SPO+ alone collapses.  Hiding the opponent's card is the source
of the whole (small) decision advantage at 133k/3 000 steps (censored − revealed advantage = +0.003 … +0.006,
CIs exclude 0), but the bank-posterior covariance gap does not predict which opponents benefit (ρ ≤ 0.15).
Every one of the 1.65 million deployed strategies was audited with the OpenSpiel best response: max
Expl − ε = 1.5e−9, zero violations, zero LP failures.  The run overran its 10 h budget: all pre-registered arms
were started, but seeds 1–2 of four tail arms were still training or queued at the box (§7).

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

*Status: DEC-133k (3 seeds), RECON-889k (3 seeds) and RECON-JAC-889k (seed 0; seeds 1–2 promoted in the
queue after its result) evaluated by 05:16 UTC; RECON-REACH-889k is queued.  Seeds 1–2 of the Jacobian arm
and the reach-weighted arm are appended when they land.*

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
(0.634–0.635 vs 0.635–0.637).  **The weighted-reconstruction arm then removes the remaining advantage**: with
the per-infoset cross-entropy weighted by the Jacobian norm of g with respect to that infoset's policy,
RECON-JAC-889k (seed 0) is *better* than DEC-889k at every N ≥ 20 (−0.004 … −0.013, CIs exclude 0) and
worse only at N = 5 (+0.008); it beats unweighted RECON-889k by 0.003 … 0.020 (CIs exclude 0 at N ≥ 10)
and has the lowest test g-NMSE of any arm (0.36 at N = 500).  The same holds at ε = 0.05 (−0.004 … −0.010
vs DEC-889k at N ≥ 20) and ε = 0.20 (−0.005 … −0.016), with the N = 5 loss (+0.005 / +0.010) in both.  One
seed so far; seeds 1–2 are queued.

**Exact parameter counts** (encoder shared by every arm: 618 752).

| arm | head | head params | total | seeds | source |
|---|---|---|---|---|---|
| DEC-133k | MLP hidden 100 → standardized g | 133 393 | 752 145 | 3 | V3 |
| RECON-131k | infoset-conditioned softmax, hidden 256 | 131 331 | 750 083 | 3 | V1 |
| DEC-889k | MLP hidden 512 | 889 413 | 1 508 165 | 3 | V1 |
| RECON-889k | hidden 834 | 889 089 | 1 507 841 | 3 | V3 |
| RECON-JAC-889k | hidden 834, Jacobian-norm infoset weights | 889 089 | 1 507 841 | 1 (2 more queued) | V3 |
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
| RECON-JAC-889k (seed 0) | 0.1749 | 0.1518 | 0.1304 | 0.1108 | 0.1022 | 0.0970 | 0.0920 |
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
| RECON-JAC-889k − RECON-889k | 0.10 | −0.001 [−0.004, 0.001] | −0.003 [−0.006, −0.001] | −0.009 [−0.012, −0.006] | −0.012 [−0.016, −0.008] | −0.016 [−0.020, −0.011] | −0.016 [−0.021, −0.012] | −0.020 [−0.025, −0.014] |
| RECON-JAC-889k − DEC-889k | 0.10 | +0.008 [0.003, 0.013] | +0.003 [−0.001, 0.007] | −0.004 [−0.008, −0.001] | −0.009 [−0.013, −0.006] | −0.010 [−0.013, −0.006] | −0.011 [−0.015, −0.007] | −0.013 [−0.018, −0.009] |

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
| RECON-JAC-889k − DEC-889k | NASH_LOGIT_PERTURB | +0.016 [0.009, 0.025] | +0.001 [−0.005, 0.006] | −0.014 [−0.021, −0.008] | −0.022 [−0.031, −0.014] |
| | NASH_RANDOM_MIX | −0.007 [−0.013, −0.000] | −0.007 [−0.012, −0.002] | −0.005 [−0.009, −0.002] | −0.009 [−0.014, −0.004] |
| | STRUCTURED_CORRELATED | +0.020 [0.007, 0.033] | −0.009 [−0.016, −0.003] | −0.021 [−0.030, −0.013] | −0.021 [−0.031, −0.012] |
| | UNSTRUCTURED_DIRICHLET | +0.003 [−0.010, 0.013] | −0.002 [−0.010, 0.006] | +0.002 [−0.005, 0.009] | −0.000 [−0.009, 0.009] |

The pooled null result at 131k is a cancellation: the decision head is *worse* on the two near-Nash families
at N ≥ 20 (up to +0.019 on NASH_LOGIT_PERTURB at N = 500) and *better* on STRUCTURED_CORRELATED at small N and
on UNSTRUCTURED_DIRICHLET at large N.  At 889k the near-Nash families are a tie or a small reversal at
N ≥ 100 and the whole surviving advantage comes from the two families that are far from equilibrium, growing
with N on the Dirichlet family (−0.022 at N = 500).  This is the same family pattern as V1 and V2.

**g accuracy is not what separates the routes.**  Test g-NMSE at N = 5 … 500: DEC-133k 0.79 → 0.42, RECON-131k
0.82 → 0.50, DEC-889k 0.79 → 0.38, RECON-889k 0.81 → 0.46, RECON-JAC-889k 0.79 → 0.36.  DEC-133k predicts g *more* accurately than
RECON-889k at every N (0.42 vs 0.46 at N = 500) yet deploys *worse* (0.120 vs 0.112), and it matches
RECON-131k in regret while beating it by 0.08 in NMSE.  A reconstruction ĝ = A y_{q̂} is always the value
vector of some legal opponent policy; the decision head's ĝ is not, and the ε-safe LP appears to penalize
unrealizable errors more than realizable ones of larger norm.  Combined with T3 (value concentrated in
low-variance directions of g) this says the decision route's edge, where it exists, comes from what the extra
capacity is spent on, not from a smaller g error.  The Jacobian-weighted arm makes the same point from the
other side: telling the reconstruction loss *which infosets matter for g* (weights range 0.03–1.13, median
0.16, so most infosets are down-weighted ~6×) is enough to make a realizable-ĝ model beat the direct
decision head.  Decision relevance, not the decision objective, is what the V1 advantage was measuring.

**Verdict against the pre-registration.**
* P1.1 (advantage shrinks but survives matching in both directions, CIs excluding 0 at every N ≥ 20):
  **falsified for the 131k pair** (DEC-133k ≥ RECON-131k at every N ≥ 20, CIs cover 0; the reversal is
  significant on the near-Nash families) and **only partly confirmed for the 889k pair** (DEC-889k <
  RECON-889k at every N and ε; at ε = 0.10 the CI excludes 0 at 6 of 7 N but not at N = 50; at ε = 0.05 it
  excludes 0 only at N ≤ 10; at ε = 0.20 at every N).  The falsification clause names this the headline of
  the run.
* P1.2 (Jacobian-weighted reconstruction closes part of the gap but does not reach the decision route):
  **falsified in the favourable direction** (1 seed): RECON-JAC-889k closes the whole gap and overtakes
  DEC-889k at N ≥ 20 (−0.004 … −0.013, CIs exclude 0), losing only at N = 5.  Per family it beats
  DEC-889k on the three structured families at N ≥ 100 and ties on the Dirichlet family; the N = 5 loss is
  on NASH_LOGIT_PERTURB and STRUCTURED_CORRELATED.  Seeds 1–2 were promoted in the queue to confirm this.
* P1.3 (RECON-889k still improving at 6 000 steps, decision arms plateau): **falsified in direction** — the
  reconstruction arms plateaued (0.2–0.45 % over the last 1 500 steps) and the decision arms were still
  improving (1.7–2.4 %).

**Deliverable figure.**  `figures/figV3_T1_matched.{png,pdf}`: (a) regret vs N for the matched arms with the
bank posterior and tabular EM; (b) paired differences with CIs; (c) validation curves of the V3 runs.
Safety: every deployed strategy of every arm audited; max Expl − ε = 6.7e−10 over the 420 000 LPs of the
seven V3 T1 runs (table in §6).  Time: 1.6–3.0 h per run, 4 in parallel; the T1 box (2.5 h) was exceeded
because the RECON-889k seeds 1–2 and the weighted arms had to queue behind round A (declared in §8).

## 2. T2 — covariance gap and censoring

**Unit-test gate.**  `tests/test_t2_factorization.py` passes: for a factorized Dirichlet posterior over the
opponent's rank-level policy, the exact sequence-form expectation A E[y] equals A y_{E[q]} to < 1e−12 (and a
2 000-sample Monte-Carlo estimate agrees to 5e−3).  So a covariance gap between E[g] and g(E[q]) can only
arise from posterior dependence between infosets — which the hidden-card marginalization creates.

**Gap magnitudes (train-bank posterior on the 2 400 held-out traces; ε = 0.10; 15 min).**

| N | 5 | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|
| posterior entropy over the 1 200-policy bank (nats) | 5.59 | 4.67 | 3.64 | 2.38 | 1.66 | 1.10 | 0.58 |
| gap_g = ‖A(E[y] − y_{E[q]})‖₂ (chips) | 0.113 | 0.087 | 0.065 | 0.040 | 0.028 | 0.018 | 0.009 |
| gap_reg = E[g]ᵀ(x*(E[g]) − x*(g(E[q]))) (chips) | 0.0075 | 0.0041 | 0.0024 | 0.0011 | 0.0007 | 0.0004 | 0.0002 |
| decision advantage, RECON-889k − DEC-889k (3 seeds each, §1) | 0.0094 | 0.0064 | 0.0044 | 0.0030 | 0.0059 | 0.0055 | 0.0065 |

gap_reg is the value a Bayesian who deploys the posterior-mean *policy* forgoes relative to one who deploys
the posterior-mean *value vector*.  It is 80 % of the observed decision advantage at N = 5, 64 % at N = 10,
55 % at N = 20 — and 3–12 % at N ≥ 100, where the advantage is as large as at N = 20.  The mechanism can
therefore account for the small-N part of the advantage in size but not for its persistence.

**Correlation with the per-opponent advantage (Spearman ρ, 300 opponents).**

| N | 5 | 10 | 20 | 50 | 100 | 200 | 500 | all N pooled |
|---|---|---|---|---|---|---|---|---|
| ρ(gap_g, advantage), pooled | 0.07 | 0.15 | 0.13 | 0.05 | 0.04 | 0.14 | 0.08 | 0.125 |
| ρ(gap_reg, advantage), pooled | −0.04 | 0.00 | −0.01 | 0.03 | 0.00 | 0.12 | 0.04 | 0.093 |
| ρ(gap_g, adv.) NASH_LOGIT_PERTURB | 0.29 | 0.43 | 0.40 | 0.16 | 0.12 | −0.04 | −0.01 | |
| NASH_RANDOM_MIX | −0.04 | −0.11 | −0.13 | −0.06 | −0.07 | 0.18 | 0.23 | |
| STRUCTURED_CORRELATED | 0.00 | 0.20 | 0.20 | 0.05 | −0.07 | 0.07 | −0.11 | |
| UNSTRUCTURED_DIRICHLET | −0.08 | −0.01 | −0.14 | −0.18 | −0.11 | 0.10 | 0.11 | |

Mean gap_g by family at N = 5 / 20 / 100: NASH_LOGIT 0.097 / 0.037 / 0.014, NASH_RANDOM 0.102 / 0.052 / 0.022,
STRUCTURED 0.143 / 0.089 / 0.031, DIRICHLET 0.108 / 0.082 / 0.043 — the largest gaps are on
STRUCTURED_CORRELATED for N ≤ 20 and on the Dirichlet family after that.  Figure V3-T2(a,b).

**Censoring toggle (opponent's card revealed after every hand; new tokenizer with 1 116 observation types,
new datasets, bank posterior and EM unchanged).**  Six revealed runs (DEC-133k × 3, RECON-131k × 3, 3 000
steps, 1.2–1.5 h each) vs the budget-matched censored pair (DEC-133k step-3000 checkpoints from T1 and
three fresh RECON-131k runs stopped at 3 000 steps, 0.9–1.3 h each).  All 3 000-step runs had their best
checkpoint at or within 250 steps of the cap.  Regret at ε = 0.10, seeds averaged:

| condition | arm | N=5 | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|---|
| censored (standard) | DEC-133k @3000 | 0.1736 | 0.1586 | 0.1469 | 0.1354 | 0.1313 | 0.1293 | 0.1274 |
| | RECON-131k @3000 | 0.1768 | 0.1607 | 0.1496 | 0.1399 | 0.1370 | 0.1347 | 0.1323 |
| | advantage (rec − dec) | +0.003 [−0.000, 0.007] | +0.002 [−0.001, 0.006] | +0.003 [−0.002, 0.007] | +0.005 [−0.001, 0.011] | +0.006 [−0.000, 0.012] | +0.005 [−0.001, 0.013] | +0.005 [−0.001, 0.012] |
| revealed | DEC-133k @3000 | 0.1709 | 0.1570 | 0.1448 | 0.1347 | 0.1314 | 0.1301 | 0.1292 |
| | RECON-131k @3000 | 0.1707 | 0.1552 | 0.1425 | 0.1340 | 0.1317 | 0.1302 | 0.1283 |
| | advantage (rec − dec) | −0.000 [−0.004, 0.003] | −0.002 [−0.005, 0.002] | −0.002 [−0.006, 0.002] | −0.001 [−0.006, 0.005] | +0.000 [−0.006, 0.007] | +0.000 [−0.006, 0.006] | −0.001 [−0.007, 0.006] |
| | censored − revealed advantage (paired) | +0.003 [0.001, 0.006] | +0.004 [0.001, 0.007] | +0.005 [0.002, 0.008] | +0.005 [0.002, 0.009] | +0.006 [0.002, 0.009] | +0.005 [0.002, 0.009] | +0.006 [0.002, 0.010] |

Per family (advantage = recon − decision, positive = decision better; N = 5 / 50 / 500):

| family | censored | revealed |
|---|---|---|
| NASH_LOGIT_PERTURB | +0.007 [0.004, 0.010] / −0.004 [−0.009, 0.002] / −0.005 [−0.015, 0.006] | +0.002 [−0.003, 0.006] / −0.011 [−0.019, −0.002] / −0.016 [−0.025, −0.006] |
| NASH_RANDOM_MIX | −0.009 [−0.013, −0.004] / −0.005 [−0.010, −0.001] / −0.005 [−0.010, 0.001] | −0.007 [−0.011, −0.002] / −0.004 [−0.009, −0.000] / −0.004 [−0.008, 0.001] |
| STRUCTURED_CORRELATED | +0.017 [0.007, 0.026] / +0.022 [0.005, 0.041] / +0.022 [0.004, 0.043] | +0.005 [−0.005, 0.016] / +0.010 [−0.006, 0.029] / +0.010 [−0.007, 0.030] |
| UNSTRUCTURED_DIRICHLET | −0.003 [−0.009, 0.004] / +0.005 [−0.005, 0.019] / +0.008 [−0.004, 0.023] | −0.001 [−0.007, 0.005] / +0.002 [−0.007, 0.013] / +0.007 [−0.003, 0.017] |

Revealing the card also improves both arms in absolute terms only marginally at this budget (≈ 0.002–0.006
at N ≤ 20, nothing at N ≥ 100): with the card visible the count statistics become sufficient and the amortized
nets gain little.  Every revealed-set strategy audited: 6 × 16 800 LPs, max Expl − ε = 1.6e−9.

**Verdict against the pre-registration.**
* P2.1 (ρ > 0.2 pooled; largest gaps on STRUCTURED_CORRELATED): **falsified on the correlation** — pooled ρ
  is 0.04–0.15 per N (0.125 over all N) for gap_g and ≈ 0 (0.093 pooled) for gap_reg; the only family where
  the gap tracks the advantage is NASH_LOGIT_PERTURB at N ≤ 20 (ρ 0.29–0.43).  The "largest gaps on
  STRUCTURED_CORRELATED" part holds for N ≤ 20.
* P2.2 (under full revelation the Dirichlet-family advantage collapses to |Δ| < 0.005 with CIs covering 0 at
  every N; the STRUCTURED_CORRELATED advantage stays > 0): **directionally confirmed, underpowered.**  The
  revealed Dirichlet advantage is −0.001 … +0.007 with every CI covering 0 (|Δ| < 0.005 at N ≤ 50, 0.005–0.007
  at N ≥ 100), but its censored counterpart at this 3 000-step budget is itself only −0.003 … +0.008 with
  CIs covering 0, so the family-level collapse cannot be resolved.  The STRUCTURED_CORRELATED advantage stays
  positive in mean under revelation (+0.005 … +0.012) but loses significance and halves.  What *is*
  resolved is the pooled effect: the censored-minus-revealed difference is +0.003 … +0.006 at every N with
  every CI excluding 0, i.e. **hiding the card is the source of the whole (small) decision advantage at this
  scale**, and with the card revealed the two routes are indistinguishable.
* Falsification criterion (ρ ≤ 0, or a Dirichlet advantage that persists under revelation): not met on the
  second clause (nothing persists); the first is met marginally for gap_reg at N ≤ 20 (ρ ≈ 0).

**Reading.**  The hidden information is what the decision route exploits — but not through the specific
bank-posterior covariance gap that was hypothesized: that gap is the right order of magnitude only at
N ≤ 20 and does not predict which opponents benefit.  With 3 000-step 133k models the effect is ≈ 0.005
chips (2–3 % of the oracle gain), matching the near-null 131k result of §1; §1 and §4 locate the
larger effects in head capacity (DEC-889k) and in decision-relevance weighting (RECON-JAC, SPO+), not in
the posterior-mean-vs-mean-value distinction.

Figure: `figures/figV3_T2_covariance.{png,pdf}` — (a) per-opponent gap_g vs advantage at N = 10, (b) ρ vs N
pooled and per family, (c) censored vs revealed advantage per family at N = 50.  Time: the T2 box (2 h)
covers the gap computation (15 min) and the analysis; the twelve 3 000-step training runs (0.9–1.5 h each,
4-way parallel) pushed the task to ≈ 5 h of wall-clock inside the shared queue (recorded in §8).

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

## 4. T4 — SPO+ loss on the matched DEC-133k head

*Status (06:45 UTC): all three seed-0 arms evaluated; seeds 1–2 of the validation-selected mixed arm
(λ = 0.3) are training and are appended here when evaluated.*

**Gate (numerical checks, `outputs/weights_v3/t4_spo_checks.json`).**  With ℓ(ĝ, g) = max_{x ∈ S_ε}
(2ĝ − g)ᵀx − 2ĝᵀx*(g) + gᵀx*(g) at ε = 0.10: ℓ(g, g) = −1.5e−16; ℓ ≥ 0 on every random (ĝ, g) pair tested;
midpoint convexity held on every random segment; the subgradient 2x*(2ĝ − g) − 2x*(g) matched central
finite differences with median relative error 1.2e−10 (six sampled coordinates agree to 10 digits); and
ℓ(ĝ, g) ≥ gᵀx*(g) − gᵀx*(ĝ) (SPO+ upper-bounds the safe regret) held on every pair.  One exact safe LP costs
41.3 ms (HiGHS, warm start).

**Batch treatment.**  x*(g) was cached once for the 1 200 training opponents at ε_train = 0.10.  Per
training step the MSE term uses the whole batch of 32 histories; the exact SPO+ term (one LP for
x*(2ĝ − g) per sample) is applied to a random subset of 8 of the 32, i.e. ≈ 0.33 s of LP time per step.
Wall-clock: 2.4 s/step for the SPO+ arms vs 1.1 s/step for MSE-only under the same 4-way contention (4.1 h
vs 1.8 h for 6 000 steps).  The same subset rule is applied inside the validation loss of these arms, which
makes that loss a noisy estimate; early stopping was nevertheless left on the pre-registered rule.

**Arms (DEC-133k architecture, 133 393-parameter head; seed 0).**

| arm | loss | best step | stopped | best val loss (own objective) | train time |
|---|---|---|---|---|---|
| MSE only (= T1 DEC-133k) | standardized-g MSE | 6000 / 5750 / 6000 | cap (3 seeds) | 0.592–0.597 | 1.7–1.8 h |
| SPO+ only | SPO+ | 500 | early, step 2000 | 0.777 | 1.9 h |
| MSE + 1.0·SPO+ | MSE + SPO+ | 2000 | early, step 3500 | 1.52 | 2.5 h |
| MSE + 0.3·SPO+ | MSE + 0.3·SPO+ | 6000 | cap | 0.860 | 4.1 h |

SPO+-only diverged in the sense that matters: its validation SPO+ loss rose from step 500 on and its g-NMSE
never left ≈ 1.05 (worse than predicting the population mean g), so the patience rule stopped it at 2 000
steps.  MSE + 1.0·SPO+ also stopped early (best at 2 000) with g-NMSE plateauing at 0.56.  Only the λ = 0.3
mix trained to the cap with its best checkpoint at the end.

**λ selection (validation only; 150 validation opponents, 1 stream, exact-LP regret at ε = 0.10 averaged over
N ∈ {20, 100, 500}).**  SPO+-only 0.198, MSE + 1.0·SPO+ 0.143, MSE only 0.138, MSE + 0.3·SPO+ 0.130 →
λ = 0.3 (`outputs/t4_val_select.json`; marker `outputs/T4_LAMBDA` steered seeds 1–2 to λ = 0.3).

**Test results at ε = 0.10 (300 held-out opponents; MSE-only averaged over 3 seeds, others seed 0).**

| arm | N=5 | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|
| MSE only — regret | 0.1710 | 0.1552 | 0.1428 | 0.1316 | 0.1260 | 0.1228 | 0.1203 |
| MSE only — g-NMSE | 0.787 | 0.673 | 0.573 | 0.488 | 0.453 | 0.432 | 0.418 |
| MSE + 0.3·SPO+ — regret | 0.1723 | 0.1537 | 0.1391 | 0.1232 | 0.1180 | 0.1150 | 0.1113 |
| MSE + 0.3·SPO+ — g-NMSE | 0.786 | 0.667 | 0.571 | 0.490 | 0.455 | 0.435 | 0.422 |
| MSE + 1.0·SPO+ — regret | 0.1844 | 0.1678 | 0.1530 | 0.1387 | 0.1344 | 0.1317 | 0.1288 |
| MSE + 1.0·SPO+ — g-NMSE | 0.840 | 0.746 | 0.669 | 0.605 | 0.582 | 0.573 | 0.565 |
| SPO+ only — regret | 0.2211 | 0.2079 | 0.2001 | 0.1955 | 0.1960 | 0.1948 | 0.1910 |
| SPO+ only — g-NMSE | 1.048 | 1.047 | 1.048 | 1.047 | 1.048 | 1.048 | 1.047 |

Paired differences vs MSE only (negative = SPO+ arm better; 95 % paired bootstrap):

| arm | N=5 | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|
| MSE + 0.3·SPO+ | +0.001 [−0.002, 0.005] | −0.002 [−0.004, 0.001] | −0.004 [−0.006, −0.001] | −0.008 [−0.011, −0.005] | −0.008 [−0.011, −0.005] | −0.008 [−0.011, −0.004] | −0.009 [−0.013, −0.005] |
| MSE + 1.0·SPO+ | +0.013 [0.008, 0.019] | +0.013 [0.007, 0.018] | +0.010 [0.005, 0.015] | +0.007 [0.002, 0.013] | +0.008 [0.003, 0.015] | +0.009 [0.003, 0.016] | +0.009 [0.002, 0.015] |
| SPO+ only | +0.050 [0.038, 0.062] | +0.053 [0.041, 0.065] | +0.057 [0.045, 0.070] | +0.064 [0.051, 0.078] | +0.070 [0.056, 0.085] | +0.072 [0.058, 0.087] | +0.071 [0.057, 0.085] |

Per family, MSE + 0.3·SPO+ vs MSE only at N = 5 / 50 / 500: NASH_LOGIT_PERTURB −0.005 / −0.008 / −0.009 (CIs
exclude 0 at N = 50), NASH_RANDOM_MIX +0.002 / −0.006 / −0.006 (exclude 0 at N ≥ 20), STRUCTURED_CORRELATED
+0.001 / −0.013 / −0.016 (exclude 0 at N ≥ 20), UNSTRUCTURED_DIRICHLET +0.008 / −0.005 / −0.005 (worse at
N ≤ 10, CIs exclude 0; better at N ≥ 50, CIs just include 0).

**Verdict against the pre-registration.**
* P4.1 (SPO+, alone or mixed, lowers safe regret at N ≥ 20 relative to MSE only with the CI excluding 0,
  while g-NMSE is equal or worse): **confirmed for the mixed arm at the validation-selected λ = 0.3**, one
  seed: regret is lower by 0.004 (N = 20) to 0.009 (N = 500) chips with every CI at N ≥ 20 excluding 0, and
  its g-NMSE is *worse* by 0.002–0.004 at N ≥ 50 and equal at N ≤ 20.  The two other SPO+ arms are worse
  than MSE only in regret *and* in g-NMSE, so they neither confirm nor falsify the mechanism; they show that
  the exact SPO+ gradient on 8 samples per step is too noisy to carry the training on its own.
* Falsification criterion (no regret improvement, or improvement only with improved g-NMSE): **not met**.

**Point of the task.**  At the same architecture, the same data and a *slightly worse* g fit, adding a
decision-aware term lowers deployed safe regret by ≈ 0.008 chips at N ≥ 50 — 4 % of the oracle-safe gain at
ε = 0.10 and about the size of the whole DEC-889k vs RECON-889k gap in §1.  It is the cleanest evidence in
V1–V3 that regret and g accuracy are separable objectives; it is also small.  Seeds 1–2 will tell whether
0.008 is stable (the DEC-133k seed spread is 0.001–0.004).

Figure: `figures/figV3_T4_spo.{png,pdf}` — (a) regret vs N, (b) g-NMSE vs N for the four arms.  Safety:
3 × 16 800 LPs, max Expl − ε = 2.1e−10, 0 failures.  Time: the T4 box (2.5 h) was exceeded by the λ = 0.3 run
alone (4.1 h under contention); recorded in §8.

## 5. T5 — empirical-Bayes hybrids

*Status (08:20 UTC): (a) BLEND and (c) LEARNED PRIOR evaluated on the test split with their controls;
(b) COUNT FEATURES seed 0 evaluated (seeds 1–2 queued; appended when they land).*

**Arms as run.**
* (a) **BLEND**: ĝ = λ_N ĝ_net + (1 − λ_N) ĝ_EM, with ĝ_net the mean over the three DEC-133k seeds (a
  single-seed variant and a no-EM ensemble control are reported alongside) and ĝ_EM
  the tabular-EM (uniform prior, α = 1) estimate mapped through g(·).  λ_N chosen per N on the 150 validation
  opponents (1 stream) from {0, 0.1, …, 1}: 0.7, 0.7, 0.8, 0.7, 0.6, 0.4, 0.5 at N = 5 … 500 — i.e. the net
  is weighted more than the counts up to N = 100 and roughly equally after.
* (b) **COUNT FEATURES**: the DEC-133k architecture with a 1 158-dimensional count vector (per-public-state
  opponent action counts over all hands, per rank-infoset counts over showdown hands, showdown/fold
  counts; each as frequency ⊕ log1p count / log1p N) added to the history-level CLS input through a linear
  layer; same loss, schedule and early-stopping rule as DEC-133k; 6 000 steps, best checkpoint at 6 000,
  validation loss 0.572 vs 0.592–0.597 for DEC-133k; seed 0 (1.7 h).
* (c) **LEARNED PRIOR** (declared simplification of "network outputs Dirichlet concentrations"): tabular EM
  in which the Dirichlet prior at every rank-level opponent infoset has mean q̂(H) = the reconstruction
  network's output for that history (mean over the three V1 RECON-131k seeds) and concentration κ_N chosen
  on validation from {1, 3, 10, 30, 100}: κ = 3 at every N (validation regret 0.161 → 0.060 from N = 5 to
  500; κ = 1 and κ = 10 are 0.003–0.012 worse, κ ≥ 30 much worse).  Controls: the same with a single seed's
  q̂ (κ = 3 again), the three-seed q̂ ensemble mapped straight to g without EM, and the same recipe with the
  three RECON-889k seeds from T1 (κ = 3 again).
* Nothing in (a) or (c) touches the test split before the single final solve; every deployed strategy comes
  from the exact ε-safe LP and is audited (max Expl − ε over the eight hybrid solves and the count-feature arm = 7.6e−10, 0 LP failures).

**Safe regret at ε = 0.10 (mean over 300 test opponents).**

| method | N=5 | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|
| train-bank posterior | 0.1583 | 0.1436 | 0.1361 | 0.1282 | 0.1269 | 0.1233 | 0.1217 |
| DEC-133k (3 seeds, seed-averaged) | 0.1710 | 0.1552 | 0.1428 | 0.1316 | 0.1260 | 0.1228 | 0.1203 |
| tabular EM (uniform prior) | 0.1960 | 0.1800 | 0.1654 | 0.1348 | 0.1138 | 0.0935 | 0.0748 |
| lower envelope of the three | 0.1583 | 0.1436 | 0.1361 | 0.1282 | 0.1138 | 0.0935 | 0.0748 |
| (b) DEC-133k + COUNT FEATURES (seed 0) | 0.1692 | 0.1500 | 0.1362 | 0.1241 | 0.1193 | 0.1162 | 0.1127 |
| (a) BLEND λ_N, 3-seed ĝ_net | 0.1652 | 0.1469 | 0.1294 | 0.1100 | 0.0978 | 0.0849 | 0.0696 |
| (a) single-seed ĝ_net (λ_N = 0.6, 0.9, 0.6, 0.7, 0.6, 0.4, 0.5) | 0.1693 | 0.1504 | 0.1334 | 0.1118 | 0.0984 | 0.0852 | 0.0702 |
| control: 3-seed DEC-133k ensemble, no EM (λ = 1) | 0.1664 | 0.1508 | 0.1380 | 0.1270 | 0.1211 | 0.1180 | 0.1154 |
| (c) LEARNED-PRIOR EM, 3-seed q̂ | 0.1675 | 0.1444 | 0.1225 | 0.0970 | 0.0831 | 0.0691 | 0.0525 |
| (c) single-seed q̂ | 0.1706 | 0.1458 | 0.1242 | 0.0983 | 0.0838 | 0.0695 | 0.0528 |
| (c) RECON-889k q̂ | 0.1690 | 0.1439 | 0.1218 | 0.0963 | 0.0823 | 0.0682 | 0.0522 |
| control: 3-seed RECON ensemble, no EM | 0.1749 | 0.1557 | 0.1399 | 0.1258 | 0.1218 | 0.1184 | 0.1166 |

**Excess over the per-N best of {bank, DEC-133k, EM} (paired; negative = below the envelope).**

| method | N=5 | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|
| BLEND | +0.007 [0.004, 0.010] | +0.003 [−0.001, 0.008] | −0.007 [−0.012, −0.002] | −0.018 [−0.024, −0.012] | −0.016 [−0.021, −0.011] | −0.009 [−0.012, −0.006] | −0.005 [−0.009, −0.002] |
| LEARNED-PRIOR EM | +0.009 [0.004, 0.014] | +0.001 [−0.003, 0.005] | −0.014 [−0.019, −0.009] | −0.031 [−0.039, −0.025] | −0.031 [−0.036, −0.026] | −0.024 [−0.029, −0.020] | −0.022 [−0.026, −0.018] |
| RECON ensemble, no EM | +0.017 [0.011, 0.022] | +0.012 [0.008, 0.017] | +0.004 [−0.002, 0.009] | −0.002 [−0.008, 0.004] | +0.008 [−0.003, 0.020] | +0.025 [0.014, 0.037] | +0.042 [0.030, 0.056] |
| COUNT FEATURES (seed 0) | +0.011 [0.008, 0.014] | +0.006 [0.003, 0.010] | +0.000 [−0.005, 0.005] | −0.004 [−0.010, 0.002] | +0.006 [−0.004, 0.015] | +0.023 [0.013, 0.033] | +0.038 [0.027, 0.049] |

(The per-N best is the bank posterior for N ≤ 50 and tabular EM for N ≥ 100.)

**Paired comparisons of the learned-prior EM (3-seed q̂; diff [95 % CI]).**

| vs | N=5 | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|
| tabular EM | −0.029 [−0.037, −0.020] | −0.036 [−0.044, −0.027] | −0.043 [−0.051, −0.035] | −0.038 [−0.044, −0.032] | −0.031 [−0.036, −0.026] | −0.024 [−0.029, −0.020] | −0.022 [−0.026, −0.018] |
| bank posterior | +0.009 [0.004, 0.014] | +0.001 [−0.004, 0.005] | −0.014 [−0.019, −0.009] | −0.031 [−0.038, −0.025] | −0.044 [−0.052, −0.036] | −0.054 [−0.063, −0.046] | −0.069 [−0.079, −0.060] |
| DEC-133k | −0.004 [−0.008, 0.001] | −0.011 [−0.015, −0.007] | −0.020 [−0.026, −0.016] | −0.035 [−0.042, −0.028] | −0.043 [−0.051, −0.035] | −0.054 [−0.064, −0.045] | −0.068 [−0.079, −0.058] |
| single-seed q̂ control | −0.003 [−0.005, −0.001] | −0.001 [−0.003, 0.000] | −0.002 [−0.003, −0.001] | −0.001 [−0.002, −0.000] | −0.001 [−0.001, −0.000] | −0.000 [−0.001, 0.000] | −0.000 [−0.001, 0.000] |
| RECON ensemble, no EM | −0.007 [−0.010, −0.005] | −0.011 [−0.015, −0.008] | −0.018 [−0.023, −0.013] | −0.029 [−0.036, −0.022] | −0.039 [−0.048, −0.030] | −0.049 [−0.059, −0.040] | −0.064 [−0.075, −0.053] |
| RECON-889k q̂ variant | −0.002 [−0.003, −0.000] | +0.001 [−0.001, 0.002] | +0.001 [−0.001, 0.002] | +0.001 [−0.000, 0.002] | +0.001 [0.000, 0.002] | +0.001 [0.000, 0.002] | +0.000 [−0.000, 0.001] |

Per family, learned-prior EM vs tabular EM at N = 5 / 50 / 500: NASH_LOGIT_PERTURB −0.024 / −0.047 / −0.043,
NASH_RANDOM_MIX −0.008 / −0.010 / −0.013, STRUCTURED_CORRELATED −0.095 / −0.091 / −0.030 (all CIs exclude
0), UNSTRUCTURED_DIRICHLET +0.013 [0.005, 0.024] / −0.004 [−0.012, 0.003] / −0.002 [−0.007, 0.002].  Versus
the bank posterior the hybrid is worse only at N = 5 on the three structured families (+0.010 … +0.020) and
better everywhere from N = 20 on, by up to −0.145 on the Dirichlet family at N = 500.  BLEND vs DEC-133k:
−0.006 … −0.051, all CIs exclude 0; BLEND vs tabular EM: −0.031 … −0.005, all CIs exclude 0, but on the
Dirichlet family BLEND is slightly *worse* than EM (+0.002 … +0.013).

**Count features.**  Giving the decision network the sufficient statistics of the count models improves it
at every N ≥ 10: COUNT − DEC-133k = −0.002 [−0.004, 0.001], −0.005 [−0.007, −0.003], −0.007 [−0.009, −0.004],
−0.008 [−0.010, −0.005], −0.007 [−0.010, −0.004], −0.007 [−0.010, −0.004], −0.008 [−0.011, −0.004] at
N = 5 … 500 (one seed).  It reaches the envelope at N = 20–50 (excess +0.000 / −0.004, CIs cover 0) but stays
an amortized model with the amortized floor: 0.113 at N = 500 against 0.075 for tabular EM and 0.053 for
the learned-prior EM.  The network uses the counts to sharpen its fit (g-NMSE and validation loss both
improve), not to become a count-based estimator.

**What the controls say.**  For BLEND: averaging the three decision nets without any EM improves DEC-133k
by a constant 0.005 at every N (CIs [−0.006, −0.004]); the single-seed blend is within 0.004 of the
three-seed blend (significant only at N ≤ 50); and the blend's gain over the ensemble alone grows from
0.001 at N = 5 to 0.046 at N = 500 — the large-N gain is the count-based component, not the ensemble.
For the learned prior: the three-seed q̂ prior is better than a single seed's by ≤ 0.003 chips
(significant only at N ≤ 50), so ensembling is not the source of the gain.  The three-seed reconstruction
ensemble *without* the EM step is worse than the single-seed RECON-131k curve of V1 by nothing (0.1749 vs
0.1771 at N = 5, 0.1166 vs 0.1180 at N = 500) and worse than the hybrid by 0.007 … 0.064: the gain is the
Bayesian update of the network's policy estimate with the observed counts, not the network's prediction
itself.  Swapping the RECON-131k prior for the RECON-889k prior changes nothing (|Δ| ≤ 0.002), consistent
with §1 (both heads sit at the same cross-entropy floor).

**Verdict against the pre-registration.**
* P5.1(a) — BLEND on or below the envelope at every N (within CI): **confirmed for N ≥ 10** (within the CI at
  N = 10, strictly below at N ≥ 20), **falsified at N = 5** (+0.007 [0.004, 0.010] above the bank posterior).
* P5.1(b) — count features improve the decision net at N ≥ 100: **confirmed** (one seed; improvement at every
  N ≥ 10 with CIs excluding 0, 0.005–0.008 chips), though the improved net is still far above the envelope at
  N ≥ 200.
* P5.1(c) — learned-prior EM at least as good as tabular EM at every N and better at N ≤ 50: **confirmed and
  exceeded**: it beats tabular EM at every N (by 0.022 … 0.043 chips, all CIs exclude 0), and beats every
  other method in this program at every N ≥ 20.  The one exception is the Dirichlet family at N = 5, where
  the learned prior is worse than the uniform prior by 0.013 (that family has no learnable structure and its
  q̂ is a poor prior at 5 hands).
* Falsification criterion (no hybrid reaches the envelope at N = 500): **not met** — both hybrids are below
  tabular EM at N = 500 (BLEND −0.005, learned prior −0.022).

**Signature.**  The learned-prior EM sits *below* the envelope of {bank posterior, decision net, tabular EM}
at every N ≥ 20, by 0.014 (N = 20) to 0.031 (N = 50–100) chips, with every CI excluding 0; it ties the bank
posterior at N = 10 and loses to it only at N = 5.  At N = 500 its regret (0.0525) is 27 % of the oracle-safe
gain at ε = 0.10 (0.195), versus 38 % for tabular EM, 62 % for DEC-133k and 54 % for DEC-889k.  It is the first
method in V1–V3 whose regret keeps falling at the rate of the count-based estimator while starting near the
amortized estimators' level at N ≤ 10.  Of the three hybrids, only the two that route the counts through the
exact likelihood (BLEND's EM component, the learned-prior EM) escape the amortized floor; feeding the same
counts into the network as features does not.

**Reading.**  The amortized networks' large-N floor (§1, V1) and tabular EM's small-N weakness are
complementary, and the cheapest way to combine them is not to blend ĝ's but to let the network supply the
*prior over opponent policies* that the exact likelihood then updates.  Note what this uses: the
*reconstruction* network's q̂ — the route that loses to the decision route when its output is deployed
directly — and not the decision network, whose ĝ has no policy-space interpretation and cannot serve as an
EM prior.  For the program's question this cuts both ways: decision training gives the better amortized
predictor, but the reconstruction representation is the one that composes with exact inference.

Figure: `figures/figV3_T5_hybrids.{png,pdf}` — (a) regret vs N for the hybrids, their controls and the three
envelope methods, (b) paired excess over the per-N best baseline.  Compute: 11 min per hybrid fit
(validation grid + test predictions, single core) and 7–11 min per test solve (16 800 LPs, 2 workers); the
T5 box was not exceeded by (a)/(c); (b) is charged to the queue overrun in §8.

## 6. Safety audit across tasks

Every deployed strategy in V1–V3 (every method × history × N × ε that was solved) was checked with
OpenSpiel's C++ tabular best response; requirement Expl(x) ≤ ε + 1e−7.  Full per-method table:
`outputs/v3_audit_table.md` (`v3_audit_table.py`).  Condensed (seeds pooled; ε = 0.10 unless stated):

| task / arm | strategies | max Expl − ε | violations | LP failures |
|---|---|---|---|---|
| V1 DEC-889k, RECON-131k (4 ε each, 3 seeds) | 403 200 | 5.5e−10 | 0 | 0 |
| V2 SAFE_REGRET (4 ε, 3 seeds) | 201 600 | 9.4e−11 | 0 | 0 |
| classical: bank posterior, EM uniform, EM Nash (4 ε) | 201 600 | 7.3e−10 | 0 | 0 |
| T1 DEC-133k (3 ε, 3 seeds) + step-3000 ckpts | 235 200 | 3.4e−10 | 0 | 0 |
| T1 RECON-889k (3 ε, 3 seeds) + step-3000 ckpts | 201 600 | 6.6e−10 | 0 | 0 |
| T1 RECON-JAC-889k seed 0 (3 ε) | 50 400 | 5.6e−10 | 0 | 0 |
| T2 revealed DEC-133k / RECON-131k (3 seeds each) | 100 800 | 1.5e−9 | 0 | 0 |
| T2 uncensored RECON-131k @3000 (3 seeds) | 50 400 | 8.9e−10 | 0 | 0 |
| T2 gap computation (subsample of the 2 × 16 800 LPs) | 84 | 1.3e−13 | 0 | 0 |
| T3 oracle: rank-k, hull, components, κ (7 ε) | 40 767 | 1.1e−9 | 0 | 0 |
| T4 SPO+ only / MSE+1.0 / MSE+0.3 (seed 0) | 50 400 | 2.1e−10 | 0 | 0 |
| T5 hybrids and controls (8 methods) + post hoc JAC-prior pilot | 151 200 | 7.6e−10 | 0 | 0 |
| **total (as of 07:22 UTC)** | **1 653 651** | **1.5e−9** | **0** | **0** |

The table is regenerated at the end of the run; arms that finish after the box are appended to
`v3_audit_table.md` and never change the zero counts unless a violation occurs (none has, in three runs of
the pipeline).

## 7. Tasks not completed or cut short

Time box: 10 h total from 21:17 UTC; the box closed at 07:17 UTC.  Status at the box:

| task | done inside the box | still running / queued at the box (kept running; sections are updated in place) |
|---|---|---|
| T1 | DEC-133k × 3, RECON-889k × 3 evaluated at ε ∈ {0.05, 0.10, 0.20}; RECON-JAC-889k seed 0 at 3 ε; training curves; §1 | RECON-JAC-889k seeds 1–2 (training, step ≈ 3 500 / 2 000 of 6 000); RECON-REACH-889k seeds 0–2 (queued last, not started) |
| T2 | everything: unit test, gaps, correlations, censoring toggle with 3 seeds per arm; §2 | — |
| T3 | everything (33 min); §3 | — |
| T4 | gate checks, LP timing, batch treatment, all three seed-0 arms evaluated, λ selected on validation; §4 | MSE+0.3·SPO+ seeds 1–2 (seed 1 training, seed 2 queued); SPO+-only seeds 1–2 (queued; low value after the seed-0 collapse) |
| T5 | (a) BLEND with single-seed and no-EM controls; (c) learned-prior EM with single-seed, no-EM and RECON-889k-prior controls; envelope figure; §5 | (b) COUNT FEATURES seed 0 finished and evaluated at 08:20 (after the box); seeds 1–2 queued |

Not run at all: the literal "network outputs Dirichlet concentrations" version of T5(c) (replaced by the
declared κ_N-from-validation simplification); SPO+ with a fixed validation subset (would remove the noise that
early-stopped two arms); RECON-REACH-889k (queued; if it does not finish it is reported as not run).

## 8. Declared deviations from the pre-registration and the protocol

1. **Compute overrun.**  All training shared one 4-slot queue; per-task boxes were exceeded by queueing, not
   by any single arm, except MSE+0.3·SPO+ (4.1 h alone).  No pre-registered arm was dropped; the only arm
   replaced is MSE+1.0·SPO+ seeds 1–2 → MSE+0.3·SPO+ seeds 1–2, per the pre-registered validation rule.
   The queue was re-ordered twice on the basis of results (RECON-JAC seeds 1–2 promoted ahead of the T4/T5
   tail after seed 0 beat DEC-889k; T4 seed-0 arms moved before the T2 runs so λ could be chosen early).
   Re-ordering changes which arms have 3 seeds at the box, not any reported number.
2. **λ selection** used exact-LP validation regret at N ∈ {20, 100, 500} (1 stream, 150 opponents) rather than
   all seven N, to keep it at ≈ 900 LPs; the choice (0.3 over 1.0) is not close (0.130 vs 0.143).
3. **SPO+ arms' validation loss** is computed with the same 8-of-batch exact-SPO+ subset as training and is
   therefore noisy; the pre-registered early-stopping rule was left in place and stopped SPO+-only at 2 000
   and MSE+1.0·SPO+ at 3 500 steps.  This is recorded, not corrected.
4. **Evaluation scope.**  Step-3000 checkpoints (T2 only) were solved at ε = 0.10 only; non-headline V3 arms
   (SPO+, censoring, hybrids) at ε = 0.10 only; RECON-JAC/REACH were added to the {0.05, 0.10, 0.20} set after
   seed 0's result (seed 0 extended by a separate solve; an incremental-solve option was added to
   `evaluate.py` for this, which leaves existing ε untouched).
5. **T5 hybrids** use the mean over the three available network seeds as ĝ_net / q̂ (an ensemble); this was
   not pre-registered, so single-seed and no-EM controls were added post hoc and are reported alongside
   (§5).  The RECON-889k-prior variant and the 3-seed no-EM ensembles are likewise post hoc controls.
6. **T5(c) simplification** (κ_N from validation instead of network-predicted concentrations) as declared
   in §0.
7. **T3 storage/figure**: the PCA spectrum saved in `t3_results.json` is truncated to 64 entries (storage
   only); panel (d) of the T3 figure was re-drawn on log-log axes after the first render.
8. **Bookkeeping incidents with no effect on results**: the scheduler was restarted three times (to run
   evaluations in a separate process, to add a restart-safe marker, and to re-order the queue); a few
   evaluations were run manually in parallel with the scheduler's; the ensemble-only hybrid control briefly
   overwrote the 3-seed prior-EM selection file, which was restored from git (its contents are also stored
   inside `hyb_meta.json`).  Every evaluation writes its own solve file and audit line; none was lost.
9. **Post hoc pilot** (§9): a learned-prior EM using the RECON-JAC-889k seed-0 q̂ was run after the box as a
   first data point for the recommended next experiment; it is labelled as such and not used in any verdict
   (it came out negative for the simplest form of the recommendation, and §9 says so).

## 9. Recommended next experiment

Combine the two things that worked, and test them against the one thing that did not.  Concretely:
**decision-relevance-weighted reconstruction as the prior of an exact-likelihood update.**  Train RECON-JAC-889k
(3 seeds, done or in progress), use its q̂(H) as the Dirichlet prior mean of the tabular EM (κ_N from
validation; then the amortized version where the network also outputs κ(H) per infoset, i.e. the T5(c) arm
as originally worded), and compare, on the same 300 opponents, against (i) the RECON-131k-prior EM of §5,
(ii) DEC-889k, (iii) the bank posterior at N ≤ 10.  Pre-registrable predictions: the JAC-prior EM is below
the RECON-131k-prior EM at every N ≥ 20 (the Jacobian weighting improved deployed regret by 0.003–0.020
when used directly; whether that survives the EM update is the question), and an amortized κ(H) closes the
N = 5 gap to the bank posterior (+0.009).  Two smaller items belong in the same run: a fixed-subset (or full)
SPO+ validation estimator so that the SPO+ arms are early-stopped on signal rather than noise, and SPO+ with
λ = 0.3 on top of the Jacobian-weighted head, since §1 and §4 both point at *what the loss weights* as the
lever.  The cheapest first data point was run after the box (post hoc, one seed, not used in any verdict):
**learned-prior EM with the RECON-JAC-889k seed-0 q̂**, κ_N from validation (10, 10, 3, 10, 3, 3, 3), test regret
at ε = 0.10: 0.1731, 0.1481, 0.1247, 0.1008, 0.0832, 0.0690, 0.0522 — i.e. **not better** than the
RECON-131k-prior EM (+0.006 [0.003, 0.008], +0.004 [0.001, 0.006], +0.002 [0.000, 0.004], +0.004 [0.001, 0.007]
at N = 5 … 50 against the 3-seed prior; within ±0.003 of the single-seed control; identical at N ≥ 100), even
though the same network deployed directly beats every other amortized model (§1).  Audit: 16 800 LPs,
max Expl − ε 3.2e−11.  So the prediction above is already falsified in its simplest form: the exact
likelihood update saturates what the prior can contribute at N ≥ 20 (131k, 889k and Jacobian-weighted q̂ all
give the same curve), and the Jacobian weighting, which helps *deployment*, slightly hurts the *prior* at
small N.  The next experiment should therefore split into the two places where headroom demonstrably
remains: (i) the N ≤ 10 gap of the hybrid to the bank posterior (+0.009 at N = 5), via an amortized κ(H) and a
per-family / per-infoset concentration, which is the T5(c) arm as originally worded; and (ii) the
deployed-network route, where §1 and §4 both locate the lever in what the loss weights — RECON-JAC with
3 seeds (running), then SPO+ (λ = 0.3, fixed validation subset) on top of the Jacobian-weighted head.
