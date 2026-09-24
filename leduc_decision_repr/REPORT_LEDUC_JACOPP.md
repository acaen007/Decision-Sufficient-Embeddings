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
