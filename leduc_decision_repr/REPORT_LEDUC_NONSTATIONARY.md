# Switching and drifting opponents (Leduc, evaluation only)

## Summary and decision

**Full-history methods fail slowly, exactly as feared, and the neural encoders track change no better.**
* **The drop.**  After a switch between contrasting opponents, every full-history method falls from a
  fraction of 0.80–0.89 to about 0.30:
  - tabular EM, bank posterior, JAC-opp, DEC-889k and the full-history learned-prior EM (PRIOR-EM).
* **The recovery.**  They need 225–280 hands to regain 80% of their own pre-switch level, and the bank
  posterior never does by t = 500.
* **The neural encoders are no exception.**  Neither neural model (trained only on stationary opponents)
  recovers relatively faster than EM.

**Simple forgetting helps a lot.**
* **Recovery.**  A 50-hand window or a 0.98 per-hand discount recovers to 80% in 33–47 hands.  Its
  post-switch fraction over the first 100 hands is +0.15 to +0.21 higher than the full-history version.
* **Cost.**  It pays a modest stationary cost (−0.03 to −0.07 before the switch).
* **Best practical method.**  PRIOR-EM-WIN, the learned-prior EM on the last 50 hands, is the best
  non-oracle method after the switch.  It is statistically tied with JAC-opp fed only the last 50 hands.
* **Drift.**  Under gradual drift the same windowed methods stay flat (≈ 0.82) while full-history methods
  decay to 0.46–0.61 by hand 500.

**Decision (pre-registered rule, contrasting pairs): the gap to the oracle is real — recommend the
training experiment, with one caveat on its baseline.**
* **The numbers.**  The fastest practical methods (JAC-opp-W50, PRIOR-EM-WIN and WIN-EM W = 50) need
  32.6–33.6 hands to regain 80% of their pre-switch fraction.  POST-PRIOR-EM, which is told the switch
  time, needs 10.3.  The ratio is 3.2, above the 1.5 threshold.  Even the hindsight-tuned grid point (WIN-EM
  W = 25, 16.3 hands) misses it (ratio 1.6), and that point costs 0.10 of stationary fraction.
* **Where the gap sits.**  The gap is entirely in the first ~50 hands after the switch, while the window
  still holds pre-switch hands.  By 50 hands after the switch PRIOR-EM-WIN *equals* the oracle (0.81 vs
  0.81), because its window then contains only post-switch hands.
* **What would close it.**  Knowing *when* to reset is what is missing.  So the recommended experiment
  (learned change detection or sequence models trained on switching opponents) should be benchmarked
  against a cheap classical change-point detector.  For example, a likelihood-ratio / CUSUM test on
  hand-type counts that triggers a reset of PRIOR-EM-WIN; this was not part of this study.
* **Random pairs are closer to the line.**  On random (less contrasting) pairs the ratio is 1.46
  (JAC-opp-W50 31.8 vs the oracle's 21.8 hands).

**Pre-registered expectations:** S1–S5 all hold.  Safety holds throughout: 23 480 deployed strategies, max
Expl − ε = 2.8e−10, 0 violations, 0 LP failures.

## 0. Pre-registration

Written and committed before any non-stationary process was generated or any method evaluated on one.  Not
edited afterwards; deviations go in the deviations section.

### Processes (100 opponents × 2 streams × 500 hands; the learner plays its blueprint while data is collected)
* **Pool.** The 300 original test opponents plus the 100 FAR-ARCH archetypes of the generalization study
  (`outputs/gen/families.npz`).  Pairs (A, B), A ≠ B, are drawn with `default_rng([2027, 1])`.
  - "Contrasting" pairs have ‖g_A − g_B‖ above the median pairwise distance over the whole pool.
  - "Random" pairs are uniform over all ordered pairs.
  - The g-distance of every pair is reported.
* **SWITCH (60):** 40 contrasting + 20 random pairs.  q_A plays hands 1–200 and q_B hands 201–500.
* **DRIFT (40):** 27 contrasting + 13 random pairs (the same 2 : 1 mix).  At hand t the opponent plays
  q_t = (1 − λ_t) q_A + λ_t q_B per infoset, with λ_t = (t − 1)/499.
* **Simulation** is the project's table-driven simulator generalized to a per-hand mixture of the two node
  tables (exact, because the mixture is linear in the behavioural probabilities).  Stream seeds follow the
  existing convention (seed 7; split ids 30 = SWITCH, 31 = DRIFT).

### Methods (ε = 0.10; neural seed 0; frozen checkpoints; κ = 3)
| group | method | input at time t |
|---|---|---|
| classical | FIXED-NE | none (x_nash) |
| classical | BANK | hands 1..t |
| classical | TAB-EM (uniform prior, α = 1) | hands 1..t |
| forgetting | WIN-EM W ∈ {25, **50**, 100} | hands t−W+1..t |
| forgetting | DISC-EM γ ∈ {0.95, **0.98**, 0.99} | hands 1..t, hand h weighted γ^(t−h) |
| neural | JAC-opp (`runs_jacopp/a3_s0`) | hands 1..t |
| neural | JAC-opp-W50 | hands t−49..t |
| neural | DEC-889k (`runs/decision_s0`) | hands 1..t |
| hybrid | PRIOR-EM (JAC-opp prior, κ = 3) | prior and counts from hands 1..t |
| hybrid | PRIOR-EM-WIN | prior and counts from hands t−49..t |
| oracle (SWITCH only) | POST-EM, POST-PRIOR-EM | hands after the switch (all hands while t ≤ 200) |

* **Defaults:** W = 50 and γ = 0.98 are the pre-declared defaults.  The whole grid is reported, and the best
  grid point is labelled "hindsight-tuned".
* **Windows** shorter than W use all available hands.

### Evaluation
* **Deployment:** at checkpoint t the method sees hands 1..t, deploys x through the exact ε-safe LP, and is
  scored against the current opponent (SWITCH: q_A for t ≤ 200, q_B for t > 200; DRIFT: q_t).
* **Checkpoints:** SWITCH t ∈ {100, 200, 205, 210, 220, 250, 300, 400, 500}; DRIFT t ∈ {20, 50, 100, 200,
  300, 400, 500}.
* **Metrics:** regret_t = V_ε(g_t) − u(x, q_t), and fraction_t = (u − V_0) / (V_ε − V_0).
  - Streams are averaged per process.
  - Fractions are pooled as a ratio of means over processes, with 95% bootstrap CIs over processes (2 000
    resamples).
  - Points with V_ε − V_0 < 0.01 are excluded from fractions (counts reported).
* **Every deployed strategy is audited** (OpenSpiel exact best response).

### Derived quantities (SWITCH; computed separately on contrasting and random pairs; contrasting is primary)
* **pre** = pooled fraction at t = 200.
* **post-mean** = mean of the pooled fractions at t ∈ {205, 210, 220, 250, 300}, i.e. the first 100 hands
  after the switch.
* **Recovery time R_p** = hands after the switch until the pooled fraction first reaches p × pre (p = 0.5,
  0.8).
  - Linear interpolation is used between post-switch checkpoints; "≤ 5" if it is already reached at t = 205.
  - "Not recovered" if it is never reached by t = 500.
  - CIs come from the same process resamples.
  - FIXED-NE is excluded, since its pre-switch fraction is ≈ 0.

### Expectations (operational tests, on contrasting pairs)
* **S1:** for TAB-EM and full-history PRIOR-EM, the fraction at t = 300 is below 0.8 × their own pre.
* **S2:** each pre-declared forgetting method (WIN-EM W = 50 and DISC-EM γ = 0.98 vs TAB-EM; PRIOR-EM-WIN
  vs PRIOR-EM) satisfies both of the following (paired over processes).  The rest of the grid is reported.
  - It has a higher post-mean than its full-history version, with the CI of the difference above 0.
  - It has a lower pre (a stationary cost), with the CI below 0.
* **S3:** neither full-history neural model (JAC-opp, DEC-889k) has a higher relative recovery
  (post-mean / pre) than full-history TAB-EM or PRIOR-EM with the bootstrap CI of the difference above 0.
* **S4:** PRIOR-EM-WIN has the highest post-mean fraction over all post-switch checkpoints (t = 205–500) among
  the non-oracle methods at their pre-declared settings.  The grid is reported separately as
  hindsight-tuned.
* **S5:** 0 audit violations.
* **Decision rule:** compare R_0.8 of the fastest-recovering non-oracle method at pre-declared settings with
  POST-PRIOR-EM's R_0.8.
  - If the ratio is ≤ 1.5, simple forgetting handles non-stationarity and the case for training sequence
    models on switching opponents is weak.
  - If every practical method lags substantially (ratio > 1.5, or the method does not recover), there is a
    real gap for learned change detection, and that training experiment is recommended.
  - The hindsight-tuned grid point is reported alongside but does not decide.

### Guardrails
* Exact audit of every deployed strategy (Expl ≤ ε + 1e−7), with 0 violations required and LP failures
  reported.
* Nothing is tuned on these processes beyond the pre-declared W and γ grid.
* Opponent counts, exclusions and wall-clock per stage are reported.
* If running long: fewer streams or opponents, never whole methods or processes.

## 1. Processes (`ns_eval.py`)

* **Pool:** the 300 original test opponents plus the 100 FAR-ARCH archetypes.  The median pairwise
  ‖g_A − g_B‖ over the pool is 1.468; that is the threshold for "contrasting".
* **SWITCH: 60 processes** (120 histories).
  - 40 contrasting pairs: ‖g_A − g_B‖ median 1.92, range 1.47–4.41.
  - 20 random pairs: median 1.48, range 0.13–3.73.
* **DRIFT: 40 processes** (80 histories): 27 contrasting and 13 random pairs.
* **Simulator check.**  The mixture simulator reproduces the project simulator exactly (identical terminal
  sequences) at λ ≡ 0 and λ ≡ 1.
* **Exclusions.**  No checkpoint of any process had headroom V_ε − V_0 < 0.01, so no points were excluded.
* **Values per checkpoint.**  Each checkpoint's current-opponent values (V_0, V_ε, g_t) were computed exactly
  (1 640 oracle LPs).

## 2. Figures (`outputs/nonstat/`)

**Encoding (consistent across figures):**
* classical: cool colours;
* forgetting: amber (WIN-EM solid, DISC-EM dash-dot; the grid shown by lightness);
* neural: warm colours;
* hybrid: green/aqua, thick lines;
* oracle: black/gray, dashed.

Bands are 95% bootstrap CIs over processes.

* **Fig 1 — `fig1_switch.png`** (full grid: `fig1_switch_full.png`).  SWITCH fraction (top) and regret
  (bottom) vs hands since the switch, for contrasting (left) and random (right) pairs.
* **Fig 2 — `fig2_drift.png`** (full grid: `fig2_drift_full.png`).  DRIFT fraction and regret vs t.
* **Fig 3 — `fig3_recovery.png`.**  Hands after the switch to regain 50% / 80% of the method's own
  pre-switch fraction, with CIs.  Bars marked "never" do not recover by t = 500.
* **Fig 4 — `fig4_forgetting_tradeoff.png`.**  Stationary performance (fraction at t = 200) vs recovery (mean
  fraction over the first 100 post-switch hands) for the whole W and γ grid, full vs windowed hybrids and
  neural models, and the oracles.
* **Fig 5 — `fig5_safety.png`.**
  - (a) Expl − ε of every deployed strategy (symlog; all at ε to within 3e−10).
  - (b, c) Harm rate (u < u(FIXED-NE)) vs t for SWITCH and DRIFT.

## 3. Recovery table (SWITCH)

**pre** = fraction at t = 200; **post-100** = mean fraction over t ∈ {205, 210, 220, 250, 300};
**post-all** = mean over t ∈ {205, …, 500}.  R₅₀ and R₈₀ are hands after the switch to regain 50% / 80% of
own pre ("5" = reached by the first post-switch checkpoint; ∞ = not by t = 500).

**Contrasting pairs (40, primary):**

| method | pre | post-100 | post-all | R₅₀ [95% CI] | R₈₀ [95% CI] |
|---|---|---|---|---|---|
| BANK | 0.803 | 0.351 | 0.425 | 42 [12, 151] | ∞ [168, ∞] |
| TAB-EM | 0.831 | 0.387 | 0.471 | 44 [11, 108] | 231 [127, ∞] |
| WIN-EM W = 25 (hindsight-tuned) | 0.734 | 0.571 | 0.595 | 5 [5, 8] | 16 [8, 295] |
| WIN-EM W = 50 | 0.763 | 0.540 | 0.582 | 6 [5, 17] | 34 [17, 265] |
| WIN-EM W = 100 | 0.805 | 0.484 | 0.561 | 18 [5, 39] | 69 [42, 95] |
| DISC-EM γ = 0.95 | 0.759 | 0.594 | 0.619 | 5 [5, 7] | 18 [7, 295] |
| DISC-EM γ = 0.98 | 0.804 | 0.544 | 0.603 | 7 [5, 19] | 47 [21, 234] |
| DISC-EM γ = 0.99 | 0.825 | 0.493 | 0.576 | 14 [5, 38] | 86 [45, 196] |
| JAC-opp | 0.850 | 0.409 | 0.482 | 35 [7, 112] | 276 [105, ∞] |
| JAC-opp-W50 | 0.818 | 0.607 | 0.653 | 5 [5, 16] | **33** [15, 47] |
| DEC-889k | 0.824 | 0.407 | 0.475 | 34 [5, 125] | 278 [110, ∞] |
| PRIOR-EM | 0.888 | 0.397 | 0.492 | 41 [18, 93] | 225 [125, ∞] |
| PRIOR-EM-WIN | 0.848 | 0.609 | **0.663** | 7 [5, 16] | **33** [19, 43] |
| *POST-EM (oracle)* | 0.831 | 0.663 | 0.706 | 5 [5, 5] | 26 [5, 98] |
| *POST-PRIOR-EM (oracle)* | 0.888 | 0.760 | 0.792 | 5 [5, 5] | **10** [5, 27] |

**Random pairs (20):**
* **Full-history methods:** R₈₀ is 227–264 hands for TAB-EM, JAC-opp and PRIOR-EM; the bank posterior and
  DEC-889k never recover.
* **Forgetting and windowed methods:** WIN-EM W = 50 needs 43, DISC-EM γ = 0.98 52, JAC-opp-W50 32 and
  PRIOR-EM-WIN 36 hands.
* **Oracles:** POST-EM 64, POST-PRIOR-EM 22.
* **Post-all:** PRIOR-EM-WIN 0.653, JAC-opp-W50 0.657, oracle 0.751.

**Absolute fractions after the switch** (contrasting; PRIOR-EM-WIN vs POST-PRIOR-EM):

| hands since the switch | 5 | 10 | 20 | 50 | 100 | 200 | 300 |
|---|---|---|---|---|---|---|---|
| PRIOR-EM-WIN | 0.38 | 0.47 | 0.58 | 0.81 | 0.80 | 0.79 | 0.80 |
| POST-PRIOR-EM | 0.67 | 0.71 | 0.77 | 0.81 | 0.84 | 0.86 | 0.88 |

The practical method equals the oracle from 50 hands on.  The oracle's advantage is confined to the window
flush, plus a small permanent edge (0.04–0.08) from using all post-switch hands rather than the last 50.

## 4. Pre-registered expectations (contrasting pairs)

* **S1 — holds.**  100 hands after the switch, TAB-EM is at 0.53 against its own pre of 0.83 (ratio 0.63).
  PRIOR-EM is at 0.58 against 0.89 (0.65).
* **S2 — holds for all three pre-declared forgetting methods.**  Post-100 and pre differences are paired
  over processes:

  | forgetting method (vs full-history version) | post-100 diff | pre diff (stationary cost) |
  |---|---|---|
  | WIN-EM W = 50 (vs TAB-EM) | +0.153 [+0.117, +0.191] | −0.069 [−0.109, −0.037] |
  | DISC-EM γ = 0.98 (vs TAB-EM) | +0.157 [+0.126, +0.188] | −0.028 [−0.051, −0.009] |
  | PRIOR-EM-WIN (vs PRIOR-EM) | +0.212 [+0.168, +0.257] | −0.040 [−0.061, −0.023] |

  Across the grid every point recovers faster.  The stationary cost is significant everywhere except
  DISC-EM γ = 0.99: +0.106 [+0.082, +0.128] faster at a cost of −0.006 [−0.019, +0.004], which is nearly
  free.
* **S3 — holds.**  Relative recovery (post-100 / pre) of the full-history neural models minus that of
  full-history EM:

  | comparison | difference [95% CI] |
  |---|---|
  | JAC-opp − TAB-EM | +0.016 [−0.059, +0.089] |
  | JAC-opp − PRIOR-EM | +0.034 [−0.018, +0.092] |
  | DEC-889k − TAB-EM | +0.028 [−0.045, +0.117] |
  | DEC-889k − PRIOR-EM | +0.046 [−0.011, +0.125] |

  None is significant.  The encoders pool the whole history like EM does; windowing, not the architecture,
  is what helps (JAC-opp-W50).
* **S4 — holds on the point estimate.**  PRIOR-EM-WIN has the highest post-all fraction (0.663).  Next are:
  - JAC-opp-W50 0.653 (difference +0.011 [−0.013, +0.039], not significant);
  - DISC-EM γ = 0.98 0.603 and WIN-EM W = 50 0.582;
  - then the full-history methods, 0.43–0.49.

  The best grid point (DISC-EM γ = 0.95, 0.619) is also below PRIOR-EM-WIN.
* **S5 — holds.**  0 violations in 23 480 audited strategies (max Expl − ε = 2.8e−10) and 0 LP failures.
  Every modelling strategy sits at Expl = ε.
* **Decision rule — gap.**  R₈₀ is 32.6 hands for the fastest pre-declared practical method (JAC-opp-W50;
  PRIOR-EM-WIN 32.7, WIN-EM W = 50 33.6), against 10.3 for POST-PRIOR-EM.  The ratio is 3.2 > 1.5, so the
  training experiment is recommended.  The hindsight-tuned WIN-EM W = 25 reaches 16.3 (ratio 1.6), and the
  tabular oracle POST-EM needs 26.4.

## 5. DRIFT (linear interpolation over 500 hands)

Pooled fraction (all 40 processes):

| method | t = 20 | 100 | 200 | 300 | 400 | 500 | mean t = 100–500 |
|---|---|---|---|---|---|---|---|
| TAB-EM | 0.73 | 0.83 | 0.82 | 0.77 | 0.70 | 0.65 | 0.753 |
| BANK | 0.80 | 0.78 | 0.76 | 0.72 | 0.64 | 0.53 | 0.686 |
| WIN-EM W = 50 | 0.73 | 0.79 | 0.75 | 0.72 | 0.74 | 0.72 | 0.743 |
| DISC-EM γ = 0.98 | 0.73 | 0.83 | 0.81 | 0.79 | 0.77 | 0.76 | 0.793 |
| DISC-EM γ = 0.99 | 0.73 | 0.84 | 0.83 | 0.80 | 0.78 | 0.77 | 0.804 |
| JAC-opp | 0.81 | 0.86 | 0.85 | 0.80 | 0.73 | 0.65 | 0.779 |
| JAC-opp-W50 | 0.81 | 0.84 | 0.82 | 0.80 | 0.81 | 0.81 | 0.815 |
| DEC-889k | 0.79 | 0.84 | 0.83 | 0.77 | 0.68 | 0.61 | 0.747 |
| PRIOR-EM | 0.82 | 0.86 | 0.85 | 0.81 | 0.76 | 0.70 | 0.796 |
| PRIOR-EM-WIN | 0.82 | 0.84 | 0.81 | 0.79 | 0.83 | 0.84 | **0.823** |

* **Full-history methods decay as the opponent drifts away from the early hands.**  On contrasting pairs
  they end at 0.46–0.61 at t = 500.
* **Windowed methods hold their level.**  PRIOR-EM-WIN reaches 0.82 at t = 500 on contrasting pairs and 0.86
  on random pairs.
* **Under slow drift, discounting at γ = 0.99 is a good compromise:** close to full-history early, and close
  to windowed late.

## 6. Harm and safety

**Harm rate** (u < u(FIXED-NE), streams averaged, pooled over contrasting and random pairs):
* **Right after a switch (t = 205):**
  - full-history methods 18–28% (BANK 28%, JAC-opp 25%, PRIOR-EM 23%, TAB-EM 20%);
  - PRIOR-EM-WIN 17%, falling to 2% by 50 hands after the switch;
  - POST-PRIOR-EM ≤ 3.3% throughout.
* **The price of windowing before the switch:** WIN-EM and DISC-EM harm 12–18% of processes at t = 100–200,
  against 2% for the full-history hybrid.
* **DRIFT at t = 500:** full-history methods 15–25%, JAC-opp-W50 and PRIOR-EM-WIN 2.5%.

**Safety:**

| audited strategies | max Expl − ε | violations (> 1e−7) | LP failures |
|---|---|---|---|
| 23 480 (SWITCH 16 200 + DRIFT 7 280) | 2.8e−10 | 0 | 0 |

The minimum Expl per method is ε − 1e−15: the exact safe LP always spends the whole budget.

## 7. Wall-clock

| stage | wall-clock |
|---|---|
| Build: pairs, simulation, 1 640 oracle LPs | 2.1 min |
| Predictions: 15 methods × 200 histories × 7–9 checkpoints | 1.8 min |
| Exact LPs + audits (23 480, 4 workers) | 5.7 min |
| Analysis (2 000 bootstrap resamples) + figures | < 1 min |
| Total compute | ≈ 10 min |

## 8. Deviations and notes

1. **Recovery-time resolution.**  Recovery times are resolved on the checkpoint grid (5, 10, 20, 50, 100,
   200, 300 hands after the switch) with linear interpolation.  "5" means the level was reached by the first
   post-switch checkpoint, and this floor limits the resolution of the oracle's R₅₀.  Bootstrap resamples
   that never recover are carried as ∞ in the CIs.
2. **Decision rule scope.**  The rule is evaluated on contrasting pairs, as pre-registered.  On random pairs
   the same ratio is 1.46.
3. **Oracles before the switch.**  For t ≤ 200 the oracles coincide with the full-history methods, as
   pre-declared.
4. **Harm pooling.**  Harm rates pool contrasting and random pairs (60 SWITCH / 40 DRIFT processes).
5. **Figure 5(a)** shows Expl − ε (symlog) rather than raw Expl, which is identically ε for every modelling
   strategy.
6. **Not tested:** a classical change-point detector.  It is recommended as the baseline for the follow-up
   training experiment (see the summary).
7. **Nothing was tuned** on these processes beyond the pre-declared W / γ grid.  No stream, opponent or
   method was cut.
