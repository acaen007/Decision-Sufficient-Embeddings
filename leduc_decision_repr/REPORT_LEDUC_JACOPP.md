# Opponent-specific decision-relevance weighting for reconstruction (Leduc)

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
* Over the 20 heaviest infosets, median 0.62.  The root infosets vary least (0.25–0.34); the round-2
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
