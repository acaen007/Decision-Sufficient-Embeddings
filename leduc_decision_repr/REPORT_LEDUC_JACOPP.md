# Opponent-specific decision-relevance weighting for reconstruction (Leduc)

## Headline

**Gate: run all arms.**  Per-opponent weights have plenty to reallocate: the mean top-20 overlap between
opponents is 0.42 (the threshold is ≥ 0.9) and the median cosine to the population mean is 0.942 (the
threshold is ≥ 0.95).

**P1 fails, and the pre-registered falsification criterion is met.**  Opponent-specific Jacobian weighting
(JAC-opp) beats V3's global weighting (RECON-JAC-global) at every N in every seed.  Here is the
seed-averaged difference at ε = 0.10:

| N | 5 | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|
| JAC-opp − global (chips) | −0.0105 | −0.0050 | −0.0017 | −0.0008 | −0.0010 | −0.0015 | −0.0030 |
| 95% CI | [−0.0153, −0.0056] | [−0.0083, −0.0015] | [−0.0043, +0.0010] | [−0.0032, +0.0015] | [−0.0031, +0.0011] | [−0.0042, +0.0011] | [−0.0057, −0.0003] |

* **The gain is U-shaped in N**: large at N ≤ 10, small and not significant at N = 20–200, and growing
  again to −0.0030 at N = 500.
* **P1 needed ≤ −0.003 with a CI below 0 at every N ≥ 50.**  Averaged over N ≥ 50 the difference is
  −0.0016 [−0.0038, +0.0007].

**OPP-REACH decomposition (P4): reach alone does not explain the gain at N ≥ 50.**
* **Worse than baseline.**  Weighting by the opponent's own reach² makes the model *worse* than the
  global baseline at every N ≥ 50: +0.0019 to +0.0026, with CIs above 0.
* **Negative share.**  The pre-registered fraction of JAC-opp's gain captured by reach is therefore
  negative: −0.88 [−5.3, −0.13] at N = 500, the only N ≥ 50 where the denominator's CI excludes 0.
* **Consequences add substantially.**  JAC-opp (reach × consequence) beats OPP-REACH at every N, by
  0.002–0.006 chips with every CI above 0.  Without the payoff-consequence factor, per-opponent weighting
  hurts at N ≥ 50.
* **Small N is the exception.**  At N = 5, reach alone captures 80% [50, 98]% of the gain (descriptive, not
  pre-registered).

**The small-N gain has a simple explanation, confirmed post hoc.**
* **What reach-dependent weights do.**  At small N the network cannot identify the opponent, so it outputs
  close to the population minimizer of its loss.  Reach-dependent weights make that minimizer a
  reach-weighted average of behaviour, which (by Kuhn's theorem) approximates the behavioural form of the
  population mixture.  That mixture's g is the mean g.
* **Deploying the N → 0 limits.**  Deployed at ε = 0.10, the plain average used by the global arms scores
  test regret 0.219.  The JAC-opp limit scores 0.199 and the exact mixture 0.195.  This ordering matches
  the trained arms at N = 5.

**Other predictions:**
* **P2 fails.**  The g-space term makes JAC-opp slightly worse: +0.0009 to +0.0026, with upper CI bounds
  up to +0.0052.
* **P3.**  Projecting, squaring and normalizing per opponent — the correction of V3's global weights —
  changes nothing measurable at ε = 0.10 (all CIs include 0).
* **EM prior.**  JAC-opp's q̂ as an EM prior gives no gain over the existing learned-prior EM at any N,
  as V3 predicted.  It does remove the small-N penalty that V3's RECON-JAC prior had.

**Safety.**  All 956 415 audited strategies passed (0 violations, 0 LP failures; max Expl − ε = 3.0e−9):
* 756 000 deployed test strategies (15 runs × 50 400);
* 16 800 EM-prior test strategies;
* 183 600 validation strategies;
* 15 post-hoc strategies.

## 0. Pre-registration

Written and committed after Step 0 (code check) and Step 1 (gate statistics, which need no training) and
before any training run of this study.  Not edited afterwards; deviations go in the deviations section.

### Questions
1. Does opponent-specific weighting of the reconstruction cross-entropy beat V3's global weighting?
2. Is any gain explained by the opponent's own reach alone, or by reach × payoff consequence?

### Theory being tested
g = A y_q.  For action a at opponent infoset I, ∂g/∂q(a|I) = r_I(q) · c_{I,a}(q), where r_I is the
opponent's own reach to I and c_{I,a} is the payoff-vector consequence of a.  The local pullback metric of
g-space error onto behaviour has per-infoset block M_I(q) = P J_I(q)ᵀ J_I(q) P, with P = I − 11ᵀ/k on the k
legal actions.  Per-infoset weighting is a block-diagonal approximation of that metric; the exact g-space
loss ‖A y_q̂ − g‖² also carries the cross-infoset terms.

### Unit of analysis
The reconstruction head predicts a rank policy (144 suit-isomorphic opponent infosets × 3 action slots).  Every
weight below is per rank infoset.  J_I for a rank infoset is the derivative with respect to that shared
policy row; it sums the physical members' r·c terms.  All physical members of a class have the same
opponent reach, which Step 1 checked exactly.

### Arms
All arms use the 889k head and the V3 optimizer and schedule, unchanged (AdamW lr 3e−4, wd 0.01, warm-up
200, cosine to 0.1×, clip 1.0, batch 32, N drawn per batch).  Each runs 6 000 steps with no early stopping
(the V3 RECON-JAC runs also ran all 6 000 steps).  Same seed ⇒ same initialization and same batch sequence
in every arm (paired design).  The loss per sample is Σ_I w_I CE_I / Σ_I w_I, as in V3.  Weights are computed
at the true training q and cached (`outputs/jacopp/weights.npz`).

| arm | name | weights |
|---|---|---|
| 0 | RECON-JAC-global | V3 file: mean over training opponents of ‖J_I‖_F (unprojected, unsquared), mean-1 |
| 1 | JAC-global-proj | population average of w̃_I(q), w̃ = ‖J_I P‖_F² normalized to sum 1 per opponent; mean-1 |
| 2 | OPP-REACH | per opponent r_I(q)² normalized to sum 1, × 144 (mean 1 over all (opponent, infoset) pairs) |
| 3 | JAC-opp | per opponent w̃_I(q), × 144 |
| 4 | JAC-opp + g-term | arm 3 loss + λ · ‖A y_q̂ − g‖² (raw chips, summed over the 1093 coordinates, batch mean), through the realization plan; λ ∈ {0.1, 0.3, 1.0} chosen on validation, seed 0 only |
| 5 | block-quadratic (optional) | Σ_I δ_Iᵀ M_I(q) δ_I, δ_I = q̂(I) − q(I) |

### Checkpoint selection (every arm, including the retrained arm 0)
Exact validation safe regret at ε = 0.10 on a fixed set of 450 LPs: 150 validation opponents × N ∈ {20, 100,
500} × stream 0.  It is evaluated every 250 steps, at steps 250 … 6 000, and the lowest wins (ties → earlier
step).  V3 saved only best/last checkpoints, so arm 0 is retrained under this rule.  λ* for arm 4 is the
lowest seed-0 selected validation regret (ties → smaller λ).  "Whichever of arms 1/2 did better" also uses
seed-0 selected validation regret.

### Order
1. Seed 0 of arms 0, 1, 2, 3 and 4 (the latter at all three λ).
2. Seeds 1–2 of arms 3 and 4 (λ*) and of the better of arms 1/2.
3. The rest: seeds 1–2 of arm 0 and of the other of arms 1/2.

Arm 5 runs only if time allows.  The report records which arms reached 3 seeds.

### Predictions
All differences are regret(arm) − regret(reference) at ε = 0.10, averaged over seeds, on 300 test opponents
× 8 streams (streams averaged per opponent).  95% CIs are paired bootstrap over opponents (2 000 resamples).

* **P1:** JAC-opp beats RECON-JAC-global at every N ≥ 50 by ≥ 0.003 chips, with paired CIs excluding 0 and
  the gap growing in N.  Operationally: at each N ∈ {50, 100, 200, 500}, the difference (arm 3 − arm 0) is
  ≤ −0.003 and its CI upper bound is < 0.  "Growing" means the gap at N = 500 exceeds the gap at N = 50, and
  the least-squares slope of the gap on log N over those four N is > 0.
* **P2:** JAC-opp + g-term is at least as good as JAC-opp at N ≥ 20.  Operationally: at each N ∈ {20, 50,
  100, 200, 500}, the upper CI bound of (arm 4 − arm 3) is < +0.003, i.e. non-inferiority at the P1 effect
  size.
* **P3:** Step 0 found that V3 did **not** project (and used the unsquared norm, not normalized per
  opponent).  So the difference (arm 1 − arm 0), with CIs at every N, is reported as the effect of that
  correction, with no directional prediction.
* **P4 (exploratory, no direction):** the fraction of JAC-opp's gain over the global baseline captured by
  OPP-REACH, (arm 2 − arm 0) / (arm 3 − arm 0).  It is reported at each N ≥ 50 and pooled over N ≥ 50, with a
  bootstrap CI from the same resamples.  If the denominator's CI includes 0, the ratio is undefined and the
  numerator and denominator are reported separately.
* **Falsification:** JAC-opp does not beat RECON-JAC-global at N ≥ 50, meaning the CI upper bound of
  (arm 3 − arm 0), averaged over N ∈ {50, 100, 200, 500}, is ≥ 0.  "P1 fails but not falsified" is reported
  as such.

### Gate rule (from the spec; outcome in Step 1)
If median cosine ≥ 0.95 AND mean top-20 overlap ≥ 0.9, per-opponent weighting has little to reallocate: run
only arms 0–1 and report that as the headline.  Otherwise run all arms.

### Evaluation and guardrails
* **Test set:** 300 test opponents, N ∈ {5, 10, 20, 50, 100, 200, 500}, 8 streams, ε ∈ {0.05, 0.10, 0.20}.
* **Reported:** safe regret, g-NMSE and per-family results, seed-averaged and per seed.
* **EM-prior check:** learned-prior EM with the best arm's q̂ as prior (κ from validation) vs the existing
  learned-prior EM.
* **Safety audit:** exact on every deployed strategy (Expl ≤ ε + 1e−7), with violations and LP failures
  reported.
* **No test tuning:** nothing is tuned on test opponents.
* **Instability:** if per-opponent weights make training unstable, it is reported with the fix used.

## 1. Step 0 — how V3 built the RECON-JAC weights

The weights (`outputs/weights_v3/recon_weights_jacobian.npy`) were produced by an inline script that was never
committed as a module.  It was recovered verbatim from the session transcript.  For each of the 1 200 training
opponents it took `torch.autograd.functional.jacobian` of the float64 rank-policy → g map at the true q.  The
Jacobian has shape (1093, 144, 3).  The script added `sqrt((J**2).sum over the 1093 g coordinates and all 3
action slots)` per rank infoset, then divided by 1 200.  `train.py` rescales the vector to mean 1, and the loss
per sample is Σ_I w_I CE_I / Σ_I w_I.

* **Not projected.** J is taken with respect to the raw per-action coordinates q(a|I), not the simplex
  tangent space; there is no P.  Illegal slots have exactly zero columns (checked: max |J| on illegal slots =
  0), so in effect the sum runs over legal actions only.
* **Unsquared.** The per-opponent quantity is ‖J_I‖_F, not ‖J_I‖_F².
* **Averaged without per-opponent normalization.** The weight is the mean of the norms over opponents.  An
  opponent with a larger overall g-sensitivity contributes more.
* g is in raw chips (not standardized); the unit is the rank infoset (144).
* **Exact reproduction.** The analytic Jacobian used in Step 1 (J = A·D, D the sparse leave-one-out product
  matrix of the realization plan) reproduces the V3 file to a maximum relative error of 2.1e−8.

Consequence for P3: arm 1 differs from arm 0 in **three** ways at once — projection, squaring, and
per-opponent normalization before averaging.  The two global vectors have cosine 0.88 (Spearman 0.94).
* **Concentration.** Arm 1's vector is far more concentrated: coefficient of variation 2.12 vs 0.90, max/min
  ratio 1 390 vs 33.
* **Where the mass sits.** Arm 1 moves mass onto the six round-1 first decisions (43% vs 19%) and away from
  round 2 (52% vs 76%).  Squaring amplifies the infosets whose reach is 1 for everybody.

## 2. Step 1 — the gate (no training; `jacopp_gate.py`, 41 s on one core)

Computed for all 1 200 training opponents and all 144 opponent rank infosets:
* w_I(q) = ‖J_I(q) P_I‖_F²;
* u_I(q) = ‖J_I(q)‖_F² (unprojected);
* reach²_I = r_I(q)²;
* the consequence-only term ‖C_I(q) P_I‖_F², where J_I = r_I·C_I.

Each is normalized to sum 1 per opponent (w̃).  Outputs are `outputs/jacopp/gate.json` and
`outputs/jacopp/gate_mass.png`.

**Checks** (5 random training opponents):
* Coordinate central differences vs J: median relative error 5.2e−11, maximum 2.4e−7 (200 coordinates).
* Random tangent (sum-zero) directions vs J_I P: median 1.8e−10, maximum 8.3e−8 (50 directions).
* The factorization J_I = r_I · C_I holds to 2.6e−16.
* Opponent reach is identical across the suit-isomorphic members of every rank class (spread 0).
* No training opponent has zero total weight, and no (opponent, infoset) pair has zero reach.

**Gate: run all arms.**  The gate fails decisively on overlap and narrowly on cosine:

| statistic | value | threshold |
|---|---|---|
| median cosine of w̃ to the population mean | 0.942 | ≥ 0.95 |
| mean top-20 Jaccard overlap between opponents | 0.423 | ≥ 0.9 |

The cosine is inflated by the six round-1 first-decision infosets.  Their reach is 1 for every opponent and
they carry 35–48% of each opponent's mass.  Dropping them lowers the median cosine to 0.81, and to 0.64 and
0.67 for the STRUCTURED and DIRICHLET families.

**a. Cosine to the population average** (median [IQR]):
* all: 0.942 [0.913, 0.955]
* NASH-LOGIT: 0.945; NASH-MIX: 0.963; STRUCTURED: 0.898 (5th percentile 0.81); DIRICHLET: 0.929.

**b. Top-k overlap (mean Jaccard).**  Top-10: 0.55, top-20: 0.42 over all pairs.

| top-20 | NASH-LOGIT | NASH-MIX | STRUCTURED | DIRICHLET |
|---|---|---|---|---|
| NASH-LOGIT | 0.73 | 0.63 | 0.34 | 0.36 |
| NASH-MIX | | 0.58 | 0.35 | 0.37 |
| STRUCTURED | | | 0.34 | 0.33 |
| DIRICHLET | | | | 0.36 |

Only the Nash-perturbation families share their top infosets.  Within STRUCTURED and DIRICHLET, two opponents
typically share about half of their top 20 (Jaccard 0.34 ≈ 10 of 20 in common).

**c. Coefficient of variation across opponents, per infoset:**
* Median 1.18 [0.91, 1.78].
* Mass-weighted mean 0.72.
* Over the 20 heaviest infosets, median 0.62.  The root infosets vary least (0.25–0.35); the round-2
  raise-facing nodes vary most (0.6–1.1).

**d. Effectively zero weights** (< 1e−4 of the opponent's total):
* 25.7% of (opponent, infoset) pairs overall.
* By family: NASH-MIX 13.8%, NASH-LOGIT 25.9%, DIRICHLET 24.4%, STRUCTURED 38.7%.
* Per opponent, median 22.9% (5th–95th percentile 8–54%).
* A typical opponent puts 90% of its mass on 46 of the 144 infosets; the top 10 hold 54% and the top 20
  hold 70%.

**e. Mass by round and depth** (figure `outputs/jacopp/gate_mass.png`; mean share per opponent):

| family | r1 first decision | r1 second decision | r2 depth 1 | r2 depth 2 | r2 depth 3 | round-2 total |
|---|---|---|---|---|---|---|
| NASH-LOGIT | 0.35 | 0.06 | 0.29 | 0.26 | 0.04 | 0.59 |
| NASH-MIX | 0.40 | 0.05 | 0.33 | 0.20 | 0.02 | 0.55 |
| STRUCTURED | 0.48 | 0.06 | 0.36 | 0.10 | 0.01 | 0.47 |
| DIRICHLET | 0.48 | 0.05 | 0.36 | 0.11 | 0.01 | 0.47 |
| V3 global vector | 0.19 | 0.05 | 0.46 | 0.26 | 0.04 | 0.76 |

Yes — tight opponents concentrate mass early, and aggressive ones shift it to deep round-2 nodes:
* Per opponent, the round-2 share falls with round-1 tightness: Spearman −0.67 with the mean fold probability
  facing a raise.
* The deep round-2 share (own depth ≥ 2) rises with aggression: Spearman +0.45 with the mean raise frequency.
* Near-Nash opponents keep the most round-2 weight and put the most weight on the K holdings (47% vs 33% for
  DIRICHLET).

**f. Reach alone explains most of the opponent-specificity:**
* **Within an opponent**, Spearman(w̃_I, reach²_I) has median 0.93 [0.89, 0.95] (NASH-MIX lowest, 0.89).
  Against the consequence factor alone it is 0.58.
* **Across opponents at a fixed infoset**, log reach² explains a median R² of 0.997 of the variation of
  log w̃_I (5th percentile 0.97).
* **The consequence factor is almost the same for every opponent:** its normalized per-opponent vector has
  median cosine 0.99 to its population mean, against 0.86 for reach² alone.

So within this population, "which infosets matter for this opponent" = (global consequence profile) ×
(this opponent's reach²).  Prediction for P4: OPP-REACH differs from JAC-opp mainly in missing the global
consequence profile, which JAC-global already carries.

**g. Projected vs unprojected** (both squared, normalized per opponent):
* Per-opponent Spearman 0.98 (Pearson 0.99).
* P keeps a median 60% of an infoset's squared norm: 53% at 2-action and 67% at 3-action infosets.
* **Largest cuts (to 6–7%):** round-2 fold/call decisions facing a re-raise while holding an unpaired card
  (e.g. Q|J, J|K after …/crr).  Fold and call have nearly parallel payoff consequences there, so the tangent
  direction carries little.
* **Relative gains (×1.6–1.7):** the same nodes when holding a pair (J|J, Q|Q, K|K after …/crr).  Fold and
  call have opposite consequences there, and P keeps 95–97%.

## 3. Training and checkpoint selection (17 runs, all 6 000 steps, validation only)

**Arm 0 reproduces V3 exactly.**  The retrained arm 0, seed 0, is bit-identical to the V3 RECON-JAC-889k
seed-0 run: training loss at steps 1, 50 and 250, and validation loss at step 250 (0.779231).  The seed
fixes initialization and batch order, so arm 0 *is* V3's trajectory, now selected by exact validation
regret.  All arms of one seed see the same batches.

Exact validation regret (ε = 0.10, 450 fixed LPs) of the selected checkpoint (selected step in parentheses):

| arm | seed 0 | seed 1 | seed 2 | seed mean |
|---|---|---|---|---|
| A0 RECON-JAC-global | 0.1127 (4750) | 0.1145 (5500) | 0.1125 (4750) | 0.1132 |
| A1 JAC-global-proj | 0.1122 (4000) | 0.1112 (4000) | 0.1121 (4750) | 0.1118 |
| A2 OPP-REACH | 0.1137 (5750) | 0.1164 (6000) | 0.1159 (3750) | 0.1153 |
| A3 JAC-opp | 0.1119 (6000) | 0.1124 (6000) | 0.1106 (5500) | **0.1116** |
| A4 JAC-opp + 0.3·g-term | 0.1134 (6000) | 0.1147 (5250) | 0.1146 (4750) | 0.1143 |
| A4 at λ = 0.1 / 1.0 (seed 0 only) | 0.1146 / 0.1138 | | | |

Pre-registered decisions (seed-0 validation, `outputs/jacopp/decisions.json`):
* λ* = 0.3 (0.1134 vs 0.1146 at λ = 0.1 and 0.1138 at λ = 1.0).
* Better of arms 1/2: arm 1 (0.1122 vs 0.1137).
* **All five arms reached 3 seeds.**  Arm 5 (block-quadratic, optional) was not run.

Training was stable in every arm, so no fix was needed and the weighting was never changed:
* No loss spikes and no divergence.  Validation curves are in `outputs/jacopp/jacopp_val.png`.
* Per-opponent weights did not let any opponent dominate: every opponent's weights sum to the same total
  (144).
* The 1 000-step block-averaged losses fell steadily in every run.  The maximum loss is the first step's,
  and no NaN occurred.
* 99th-percentile gradient norms were 0.43–0.54 for arms 0–3, so the 1.0 clip was essentially never
  active.  The g-term arms have larger gradients (p99 1.5 at λ = 0.3, 4.4 at λ = 1), so clipping was
  active there, as in any run under the V3 optimizer settings.
* The g-term is on the same scale as the CE.  At λ = 1 it fell from 1.34 to 0.25 over training while the
  CE fell from 0.94 to 0.61.

## 4. Test results (300 opponents × 8 streams × 7 N, seeds averaged)

### 4.1 Safe regret at ε = 0.10 (chips)

| arm | 5 | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|
| A0 RECON-JAC-global | 0.1756 | 0.1513 | 0.1307 | 0.1122 | 0.1030 | 0.0975 | 0.0936 |
| A1 JAC-global-proj | 0.1738 | 0.1507 | 0.1312 | 0.1121 | 0.1026 | 0.0966 | 0.0918 |
| A2 OPP-REACH | 0.1673 | 0.1495 | 0.1317 | 0.1146 | 0.1051 | 0.0995 | 0.0962 |
| **A3 JAC-opp** | **0.1652** | **0.1463** | **0.1291** | **0.1114** | **0.1019** | **0.0960** | **0.0907** |
| A4 JAC-opp + g-term | 0.1667 | 0.1473 | 0.1300 | 0.1127 | 0.1032 | 0.0984 | 0.0933 |
| *context:* V3 RECON-889k (plain CE) | 0.1763 | 0.1552 | 0.1389 | 0.1231 | 0.1177 | 0.1134 | 0.1116 |
| *context:* DEC-889k (decision head) | 0.1669 | 0.1488 | 0.1346 | 0.1201 | 0.1118 | 0.1079 | 0.1051 |
| *context:* V3 RECON-JAC (loss-selected) | 0.1755 | 0.1511 | 0.1309 | 0.1115 | 0.1025 | 0.0971 | 0.0926 |
| *context:* learned-prior EM (HYB_PRIOR_EM) | 0.1675 | 0.1444 | 0.1225 | 0.0970 | 0.0831 | 0.0691 | 0.0525 |

* **JAC-opp is the best neural model at every N.**  Against plain reconstruction it gains 0.009–0.021;
  against DEC-889k 0.002–0.014 (CIs exclude 0 from N = 10 upward).
* **Selection rule.**  Regret-based selection of the V3 trajectory (A0) is no better on test than V3's
  loss-based selection of the same runs (differences −0.0002 to +0.0010).

### 4.2 Paired differences at ε = 0.10 (seed-averaged, 95% CIs; \* = CI below 0, † = CI above 0)

Figure: `outputs/jacopp/jacopp_diffs.png` (all arms − A0, by N, at ε = 0.05 / 0.10 / 0.20, with 95% CIs).

| comparison | 5 | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|
| A3 − A0 (P1) | −0.0105\* | −0.0050\* | −0.0017 | −0.0008 | −0.0010 | −0.0015 | −0.0030\* |
| | [−.0153, −.0056] | [−.0083, −.0015] | [−.0043, +.0010] | [−.0032, +.0015] | [−.0031, +.0011] | [−.0042, +.0011] | [−.0057, −.0003] |
| A1 − A0 (P3) | −0.0019 | −0.0006 | +0.0005 | −0.0001 | −0.0004 | −0.0010 | −0.0018 |
| | [−.0040, +.0001] | [−.0023, +.0011] | [−.0012, +.0022] | [−.0019, +.0019] | [−.0021, +.0014] | [−.0031, +.0012] | [−.0039, +.0002] |
| A2 − A0 | −0.0084\* | −0.0018 | +0.0010 | +0.0024† | +0.0022† | +0.0019† | +0.0026† |
| | [−.0133, −.0031] | [−.0051, +.0020] | [−.0014, +.0036] | [+.0004, +.0045] | [+.0005, +.0039] | [+.0003, +.0035] | [+.0008, +.0044] |
| A4 − A3 (P2) | +0.0015 | +0.0010 | +0.0009 | +0.0013 | +0.0012 | +0.0024† | +0.0026 |
| | [−.0002, +.0030] | [−.0006, +.0027] | [−.0008, +.0026] | [−.0008, +.0033] | [−.0008, +.0032] | [+.0001, +.0044] | [−.0000, +.0052] |
| A1 − A3 | +0.0086† | +0.0044† | +0.0021 | +0.0007 | +0.0007 | +0.0006 | +0.0012 |
| A2 − A3 | +0.0021† | +0.0032† | +0.0026† | +0.0032† | +0.0032† | +0.0035† | +0.0056† |

Per seed, A3 − A0 is negative in all 21 (seed, N) cells:

| seed | 5 | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|
| s0 | −0.0115\* | −0.0058\* | −0.0010 | −0.0000 | −0.0006 | −0.0001 | −0.0020 |
| s1 | −0.0096\* | −0.0054\* | −0.0026 | −0.0006 | −0.0008 | −0.0018 | −0.0045\* |
| s2 | −0.0104\* | −0.0037 | −0.0014 | −0.0017 | −0.0017 | −0.0028 | −0.0023 |

### 4.3 Pre-registered verdicts

* **P1 — fails.**  The gap does grow from N = 50 to 500: −0.0008 → −0.0030, slope +0.0009 per unit of
  log N.  But only N = 500 reaches the −0.003 bar with a CI below 0, and N = 50, 100 and 200 have CIs
  that include 0.
* **Falsification — met.**  The difference averaged over N ≥ 50 is −0.0016 [−0.0038, +0.0007], so the
  upper bound is ≥ 0.  Read literally, "JAC-opp does not beat RECON-JAC-global at N ≥ 50".  It is worth
  stating what the data do show: the direction is consistent (all seeds, all N), the effect at N = 50–200
  is 0.001–0.0015 chips, and it is significant at N = 500.
* **P2 — fails.**  Non-inferiority at the +0.003 margin fails: upper CI bounds are 0.0026, 0.0033,
  0.0032, 0.0044 and 0.0052 at N = 20, 50, 100, 200 and 500.  The exact g-space term does not help on top
  of per-opponent weighting, and trends slightly worse at every N.
* **P3.**  V3 did not project, so this is the effect of the correction (projection + squaring +
  per-opponent normalization): −0.0019 to +0.0005, with no CI excluding 0 at ε = 0.10.  At ε = 0.20 it
  reaches −0.0026 and −0.0024 at N = 5 and 10.  The correction to the *global* vector is immaterial.
  JAC-opp's gain over A1 (the same weights averaged over opponents) is +0.0086 and +0.0044 at N = 5 and
  10 and +0.0006 to +0.0012 (n.s.) at N ≥ 50.  So the benefit of making the weights opponent-specific is
  concentrated at small N.
* **P4 — reach share of JAC-opp's gain, (A2 − A0) / (A3 − A0):**

  | | N = 50 | 100 | 200 | 500 | pooled N ≥ 50 |
  |---|---|---|---|---|---|
  | numerator A2 − A0 | +0.0024 [+.0004, +.0043] | +0.0022 [+.0005, +.0037] | +0.0019 [+.0002, +.0035] | +0.0026 [+.0008, +.0043] | +0.0023 [+.0007, +.0038] |
  | denominator A3 − A0 | −0.0008 [−.0033, +.0015] | −0.0010 [−.0031, +.0011] | −0.0015 [−.0042, +.0011] | −0.0030 [−.0056, −.0004] | −0.0016 [−.0039, +.0006] |
  | ratio | undefined (denominator CI spans 0) | undefined | undefined | **−0.88 [−5.3, −0.13]** | undefined |

  By the pre-registered rule the ratio is reported only at N = 500, where it is negative: reach alone moves
  regret in the *opposite* direction to JAC-opp.  At N = 50–200 the numerator is clearly positive (reach
  alone is worse than the global baseline) while the denominator is not significant.  **Payoff
  consequences add substantially; reach does not dominate.**  Descriptively, at small N reach alone
  captures 80% [50, 98]% of the gain at N = 5 and 35% [−75, 75]% at N = 10.

### 4.4 ε transfer (all arms trained at ε-agnostic reconstruction targets; selected at ε = 0.10)

A3 − A0:

| ε | 5 | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|
| 0.05 | −0.0061\* | −0.0024 | −0.0002 | +0.0009 | +0.0011 | +0.0008 | −0.0001 |
| 0.10 | −0.0105\* | −0.0050\* | −0.0017 | −0.0008 | −0.0010 | −0.0015 | −0.0030\* |
| 0.20 | −0.0124\* | −0.0071\* | −0.0025 | −0.0011 | −0.0018 | −0.0028 | −0.0038\* |

A2 − A0 at ε = 0.05 / 0.20: −0.0051\* / −0.0084\* at N = 5, then +0.0020 to +0.0030 / +0.0011 to +0.0032 at
N ≥ 20 (CIs above 0 at every N ≥ 20 at ε = 0.05 and at N ≥ 100 at ε = 0.20).  A1 − A0 at ε = 0.20: −0.0006 to −0.0026 (significant only at N = 5, 10).  A4 − A0 at ε = 0.20:
−0.0094\*, −0.0063\* at N = 5, 10, then −0.0003 to −0.0017 (n.s.).

The JAC-opp gain grows with ε, as expected: a larger safety budget exploits the model more, so model
accuracy on decision-relevant infosets matters more.  At ε = 0.05 only the N = 5 gain survives.

### 4.5 Per family (ε = 0.10, seeds averaged; \* / † as above)

| family | comparison | 5 | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|---|
| NASH-LOGIT | A3 − A0 | −0.0098\* | −0.0043\* | −0.0034 | +0.0004 | +0.0015 | +0.0023 | +0.0014 |
| NASH-MIX | A3 − A0 | +0.0070† | +0.0042 | +0.0039 | +0.0015 | +0.0010 | +0.0012 | +0.0002 |
| STRUCTURED | A3 − A0 | −0.0257\* | −0.0145\* | −0.0029 | −0.0014 | −0.0033 | −0.0043 | −0.0055 |
| DIRICHLET | A3 − A0 | −0.0135\* | −0.0053 | −0.0042 | −0.0037 | −0.0034 | −0.0054 | −0.0078\* |
| NASH-LOGIT | A2 − A0 | −0.0099\* | −0.0045 | −0.0013 | +0.0026† | +0.0038† | +0.0045† | +0.0058† |
| NASH-MIX | A2 − A0 | +0.0095† | +0.0079† | +0.0061† | +0.0033† | +0.0028† | +0.0022† | +0.0015 |
| STRUCTURED | A2 − A0 | −0.0215\* | −0.0072\* | −0.0007 | +0.0032 | +0.0007 | −0.0004 | −0.0002 |
| DIRICHLET | A2 − A0 | −0.0117 | −0.0033 | −0.0002 | +0.0005 | +0.0014 | +0.0014 | +0.0032 |
| DIRICHLET | A1 − A0 | +0.0013 | −0.0007 | +0.0007 | −0.0052\* | −0.0039 | −0.0068\* | −0.0083\* |
| STRUCTURED | A4 − A3 | +0.0079† | +0.0057† | +0.0014 | +0.0049† | +0.0043† | +0.0047† | +0.0039 |

JAC-opp's gains are on the two families whose per-opponent weights differ most from the population average
in the gate (§2: median cosine 0.898 and 0.929; top-20 overlap 0.34–0.36).
* **STRUCTURED:** −0.026 at N = 5; −0.001 to −0.0055 at N ≥ 50.
* **DIRICHLET:** −0.013 at N = 5; −0.003 to −0.008 at N ≥ 50.

On the near-Nash families it gains little or loses slightly:
* **NASH-MIX,** the family with the most global weights (cosine 0.963): +0.007 at N = 5.
* **NASH-LOGIT:** gains at N ≤ 10, small losses at N ≥ 50 (n.s.).

Reach alone hurts the Nash families at N ≥ 50 in particular.

**Exploratory (post hoc), the same pattern within opponents.**  I computed each *test* opponent's own
w̃ at its true q (for analysis only; never used by any model) and its cosine to the training-population
mean.
* JAC-opp's gain over A0 correlates with how far the opponent's weights are from the mean: Spearman −0.23
  (p = 5e−5) over all N, and −0.14 (p = 0.013) at N ≥ 50.
* Lowest-cosine third of test opponents: −0.0056 overall, −0.0028 at N ≥ 50.
* Highest-cosine third: +0.0008.
* The per-opponent part alone (A3 − A1) shows the same sign (Spearman −0.14).
* File: `outputs/jacopp/posthoc_gain_vs_cosine.json`.

### 4.6 g-NMSE (seeds averaged; difference vs A0 with CI)

| arm | 5 | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|
| A0 | 0.793 | 0.668 | 0.558 | 0.456 | 0.413 | 0.385 | 0.368 |
| A3 − A0 | −0.0027 | −0.0060 | −0.0043 | −0.0030 | −0.0052\* | −0.0061\* | −0.0086\* |
| A2 − A0 | −0.0049 | +0.0020 | +0.0083† | +0.0119† | +0.0123† | +0.0114† | +0.0114† |
| A1 − A0 | +0.0009 | −0.0006 | +0.0005 | +0.0025 | +0.0026 | +0.0036 | +0.0031 |
| A4 − A0 | −0.0017 | −0.0045 | −0.0012 | −0.0007 | −0.0027 | −0.0020 | −0.0046 |

JAC-opp also has the most accurate g at N ≥ 100.  Reach-only weighting is the least accurate (+0.011 at
N ≥ 50): it spends the loss on high-reach infosets regardless of whether their actions change g.
Remarkably, adding the exact g-space loss (A4) makes g *less* accurate than JAC-opp alone.

## 5. Why the small-N gain? — the mixture-consistency effect (post hoc, `outputs/jacopp/posthoc_population_limit.json`)

**The N → 0 limit.**  With little data the encoder cannot identify the opponent, and the reconstruction
head tends to the population minimizer of its loss.  For weighted soft-target cross-entropy, that
minimizer at each infoset is the weighted average of the training opponents' behaviour there:

  q̂_I = Σ_o w_I(o) q_I(o) / Σ_o w_I(o).

* **Global weights (A0, A1)** give the plain average of behaviour.
* **Weights proportional to the opponent's own reach r_I** give exactly the behavioural strategy of the
  population *mixture* (Kuhn's theorem).  Its g = A·E[y] is the mean g, which is the correct
  risk-neutral target, since E[gᵀx] = E[g]ᵀx.

I deployed each arm's N → 0 limit (the exact safe LP at the 1 200 training opponents' q) and scored it on
the test opponents.  All 15 strategies were audited: max Expl − ε = 1.1e−13.

| population limit of | ‖g(q̂) − mean g‖ | test regret ε = 0.05 | ε = 0.10 | ε = 0.20 | trained arm at N = 5, ε = 0.10 |
|---|---|---|---|---|---|
| global weights (A0 = A1: plain average) | 0.159 | 0.1656 | 0.2189 | 0.3406 | 0.1756 / 0.1738 |
| per-opponent reach² (A2) | 0.100 | 0.1496 | 0.1965 | 0.2871 | 0.1673 |
| per-opponent w̃ (A3) | 0.108 | 0.1518 | 0.1991 | 0.2869 | 0.1652 |
| reach r, unnormalized (exact mixture) | 0 | 0.1488 | 0.1946 | 0.2869 | — |

* **Same ordering as the trained arms.**  The plain average is 0.02–0.05 chips worse than
  mixture-consistent averaging at N → 0, and the trained arms at N = 5 are ordered the same way.
* **The gap shrinks as N grows**, as expected: as the posterior concentrates on one opponent the weighting
  stops mattering for aggregation.
* **Two mechanisms, explaining both the U shape and P4:**
  - At **small N**, any reach-dependent weight fixes the aggregation.  That is why OPP-REACH captures 80%
    of the gain at N = 5.
  - At **large N**, what matters is spending accuracy on infosets where actions change payoffs, which
    needs the consequence factor.  OPP-REACH lacks it and is worse than the global Jacobian vector.

## 6. EM-prior check (ε = 0.10)

The best arm by seed-averaged validation regret is A3, JAC-opp (`outputs/jacopp/emprior_choice.json`).  Its
3-seed q̂ is used as the prior of the learned-prior EM, following the V3 recipe (per-history Dirichlet
prior κ·q̂, κ_N from validation).  Selected κ: 3, 10, 10, 3, 3, 3, 3 at N = 5 … 500; the existing
HYB_PRIOR_EM selected 3 at every N.

| N | 5 | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|
| EM + JAC-opp prior | 0.1654 | 0.1422 | 0.1229 | 0.0992 | 0.0833 | 0.0688 | 0.0518 |
| existing learned-prior EM | 0.1675 | 0.1444 | 0.1225 | 0.0970 | 0.0831 | 0.0691 | 0.0525 |
| difference | −0.0020 | −0.0022 | +0.0005 | +0.0023 | +0.0002 | −0.0002 | −0.0007 |
| 95% CI | [−.0062, +.0023] | [−.0060, +.0014] | [−.0024, +.0032] | [−.0003, +.0048] | [−.0017, +.0022] | [−.0020, +.0015] | [−.0026, +.0012] |
| vs EM + V3 RECON-JAC prior | −0.0077\* | −0.0059\* | −0.0018 | −0.0016 | +0.0002 | −0.0001 | −0.0003 |

**V3's prediction holds: no gain beyond N = 20, and in fact no significant gain at any N.**
* The JAC-opp prior removes the small-N penalty of the V3 RECON-JAC prior (−0.008 and −0.006 at N = 5, 10),
  consistent with §5: its prior is mixture-consistent.
* The EM + JAC-opp prior beats neural JAC-opp itself by 0.004–0.039 at N ≥ 10.  At large N, tabular
  evidence dominates any learned prior.

## 7. Audit table

| deployed strategies | n | max Expl − ε | violations (> 1e−7) | LP failures |
|---|---|---|---|---|
| test, A0 seeds 0/1/2 | 3 × 50 400 | 2.0e−10 / 2.9e−10 / 3.5e−10 | 0 | 0 |
| test, A1 seeds 0/1/2 | 3 × 50 400 | 3.3e−10 / 6.5e−10 / 1.4e−10 | 0 | 0 |
| test, A2 seeds 0/1/2 | 3 × 50 400 | 9.6e−10 / 2.2e−10 / 2.1e−10 | 0 | 0 |
| test, A3 seeds 0/1/2 | 3 × 50 400 | 3.0e−9 / 1.0e−9 / 1.1e−10 | 0 | 0 |
| test, A4 seeds 0/1/2 | 3 × 50 400 | 8.6e−10 / 5.4e−10 / 1.0e−10 | 0 | 0 |
| test, EM + JAC-opp prior (ε = 0.10) | 16 800 | 5.9e−11 | 0 | 0 |
| validation, 17 runs × 24 checkpoints × 450 | 183 600 | 4.6e−10 | 0 | 0 |
| post hoc, population limits (§5) | 15 | 1.1e−13 | 0 | 0 |
| **total** | **956 415** | **3.0e−9** | **0** | **0** |

## 8. Wall-clock

| stage | wall-clock | notes |
|---|---|---|
| Step 0 + Step 1 (gate) | ~1 h of analysis; gate computation 41 s on one core | |
| Training, 17 runs | 18:26 → 04:27 UTC | 4 concurrent, 1 thread each |
| Test evaluations, 15 runs | 02:37 → 06:12 UTC | ~21 min each on 2 workers |
| EM-prior check | 06:08 → 06:30 UTC (22 min) | |
| Scheduler total | 12.1 h | |

Per-run cost:
* **A0–A3:** 1.87–1.99 h per 6 000-step run.
* **A4 (g-term):** 2.27–2.38 h.
* **Validation inside those times:** about 60 s per validation (7-N validation loss plus 450 exact LPs
  with audits), × 24.

## 9. Deviations and notes

1. **Arm 5** (block-quadratic, optional) was not run.  All five other arms reached 3 seeds.
2. **Pre-registration timing.**  §0 was written after Steps 0–1, as the spec's ordering implies, and
   before any training.  It was not edited afterwards.
3. **P4 undefined at most N.**  The ratio is defined only at N = 500 under the pre-registered rule.  The
   small-N ratios in the headline and in §4.3 are descriptive and not pre-registered.
4. **Post-hoc analyses.**  §4.5's cosine analysis and §5's population limits were not pre-registered.  §4.5
   computes weights at *test* opponents' true q for analysis only; no model or selection uses them.
5. **Container restart.**  The session container restarted once during training (around 03:40 UTC).  The
   training, evaluation and scheduler processes kept running; only my monitoring tasks were lost and
   re-created.  No run was interrupted or re-run.
6. **Validation-row weights.**  Per-opponent weights were also computed for the 150 validation opponents,
   only to log each arm's own validation loss.  Test rows were never computed for training.  Checkpoint
   selection used exact regret, not that loss.

## 10. Recommendation

**Keep per-opponent weighting, but credit it to what it actually fixes, and drop the rest.**

* **What the gain is.**  JAC-opp is the best neural model in this project.  It beats V3's global Jacobian
  weighting at every N in every seed.  But most of its gain is at N ≤ 10 (−0.005 to −0.010 chips), and
  §5 traces that to *aggregation*: reach-dependent weights make the network's low-information prediction
  the behavioural form of the population mixture, instead of an average of behaviours whose g is wrong.
* **What the data rule out:**
  - P1's claimed large-N benefit is not established.
  - Reach-only weighting is actively harmful at N ≥ 50.
  - The exact g-space term adds nothing.
  - Correcting the global vector (projection, squaring) is immaterial.
* **Next step.**  One cheap, principled arm would separate the two mechanisms: CE weighted by r_I(q)
  (unsquared, not normalized per opponent, which makes the N → 0 limit exactly the mixture) times the
  global consequence profile ‖C_I P‖² averaged over the population.  If it matches JAC-opp at small N and
  A1 at large N, the per-opponent consequence factor is unnecessary and the method reduces to "reach-weighted
  CE with a global decision-relevance profile".
* **Where the remaining gap lies.**  At N ≥ 50 the learned-prior EM is still 0.01–0.04 chips better than any
  neural model.  A learned prior adds nothing there (§6), so further gains at large N will come from how
  evidence is used (the likelihood), not from how the reconstruction loss is weighted.
