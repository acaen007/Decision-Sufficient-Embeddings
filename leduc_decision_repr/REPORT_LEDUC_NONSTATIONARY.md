# Switching and drifting opponents (Leduc, evaluation only)

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
