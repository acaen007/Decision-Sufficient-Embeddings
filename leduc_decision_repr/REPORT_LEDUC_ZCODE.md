# Predicting a decision code from hands (Leduc)

## Summary and verdict

**Negative.  Predicting the compact decision code from hands is worse than predicting the full policy, at every
number of hands.  Nothing in this study beats the current best method (learned-prior EM).**

**Headline numbers.**  Safe regret in chips per hand at ε = 0.10, on 300 test opponents × 8 streams; lower is
better:

| N | 5 | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|
| **ZC-DEC** (hands → 8-number regret code → frozen decoder) | 0.178 | 0.160 | 0.143 | 0.126 | 0.117 | 0.112 | 0.107 |
| ZC-DISTILL (regress the code) | 0.190 | 0.163 | 0.141 | 0.118 | 0.106 | 0.099 | 0.093 |
| JAC-opp (hands → full policy) | **0.165** | **0.146** | **0.129** | 0.111 | 0.102 | 0.096 | 0.091 |
| PRIOR-EM (JAC-opp prior), the best method | 0.165 | 0.142 | 0.123 | **0.099** | **0.083** | **0.069** | **0.052** |

* **ZC-DEC trails JAC-opp by 0.013–0.017 chips at every N,** with every CI above 0.  The primary expectation Z1
  (better at N ≤ 20) fails in the opposite direction.
* **The distilled code is also worse at small N (+0.025 at N = 5).**  It only catches up with JAC-opp from N = 200.
* **The smaller code is worse still:** d = 4 trails d = 8 at every N.
* **Swapping the prior makes the hybrid worse.**  Using the code model's policy as the prior for EM is 0.010–0.017
  chips worse than the JAC-opp prior at every N.
* **The ceiling is far off.**  With 500 hands, ZC-DEC reaches 0.779 of the attainable safe value, against 0.889 when
  the true policy is given.

**Why it fails: two post-hoc diagnostics.**
1. **The code cannot express uncertainty.**  After a few hands the right thing to deploy against is the *belief*: an
   average over plausible opponents.  The 8-number decoder was built for individual opponents, and the closest it
   can get to a belief loses value:
   - 0.044 at N = 5, and 0.106 for "no information";
   - ≈ 0 from N = 100.
2. **The code is fragile.**  The code packs decision information densely, so small errors matter.
   - Noise of 0.25 / 0.5 standard deviations in the true code costs 0.04 / 0.13 of value.
   - From 500 hands the model recovers only 84% of the code's variance, which predicts about 0.80, close to what it
     achieves.

**The lesson is the flip side of the compression result.**
* **Compression concentrates what matters.**  Squeezing an opponent into a few decision-relevant numbers means every
  number matters, and there is no room to represent "I'm not sure yet".
* **Behaviour spreads it out.**  A behavioural representation spreads information over many directions, many of
  which don't affect decisions.  Estimation errors partly land where they are harmless, and uncertainty can be
  represented as a mixture of behaviours.
* **So a compact decision code is the right *description* of a known opponent, but the wrong *target for
  inference*.**  In Leduc, behaviour-level beliefs plus the exact hand likelihood (learned-prior EM) remain best.

**Expectations:**
* **Z2, Z4, Z5 hold:**
  - the decision loss beats distillation at N ≤ 20, driven by N = 5;
  - learned-prior EM beats the code at large N;
  - the code model makes a worse EM prior.
* **Z1 and Z3 fail.**
* **Practical verdict: does not help in play.**

**Safety:** 151 200 deployed test strategies, plus 4 500 in the diagnostics, all audited: max Expl − ε = 2.9e−9,
**0 violations**, 0 LP failures.

## 0. Pre-registration

Written and committed before any run below was started.  It is not edited afterwards; deviations go in the
deviations section.

### Question
The decision-compression study showed that, given the opponent's true policy, 8 numbers trained with a regret
objective keep 0.889 of the attainable safe value.  Reconstruction needs 8× more numbers for the same value.

**Does predicting that compact decision code from observed hands, instead of the full policy, give better safe
responses — especially from few hands?**

### Frozen decision code
* **Primary:** the REGRET autoencoder at d = 8, seed 0 (`outputs/dcomp/ae_runs/REGRET_d8_s0.pt`); true-policy test
  ceiling 0.889.
* **Secondary:** the d = 4 autoencoder (`REGRET_d4_s0.pt`); ceiling 0.845.
* Its decoder (code → q̂) is frozen and copied into each new model.

### Arms
**What every new arm shares with the JAC-opp arms:**
* the hierarchical Transformer history encoder (z ∈ R¹²⁸);
* the V1 training streams, with N sampled from {5, …, 500};
* 6 000 steps, batch 32, AdamW 3e−4, 200 warm-up steps;
* checkpoint selection by exact validation regret on the 450 fixed LPs every 250 steps, exactly as for the
  JAC-opp arms.

**The new head:** z (128) → MLP 128→834→834→d → ẑ → frozen decoder → q̂ → ĝ = g(q̂) → exact ε-safe LP.  It has about
810k trainable parameters, against 889k for JAC-opp's reconstruction head.

| arm | loss | seeds |
|---|---|---|
| **ZC-DEC** (d = 8) | bank regret of ĝ against the true opponent's g (β = 100, bank = the ε = 0.10 safe responses of the 1 200 training opponents) + 0.1 ‖ĝ − g‖², i.e. the autoencoder's selected loss | 0, 1, 2 |
| **ZC-DISTILL** (d = 8) | ‖ẑ − z*(q)‖² on the standardized code, where z* is the frozen regret encoder applied to the true q ("predict the regret code") | 0, 1, 2 |
| **ZC-DEC-4** (d = 4) | as ZC-DEC, with the d = 4 code | 0, 1 (secondary; fills the second training wave) |
| **PRIOR-EM (ZC prior)** | the V3 recipe: the seed-ensembled q̂ of the three ZC-DEC runs as a κ-Dirichlet prior for tabular EM, κ per N from {1, 3, 10, 30, 100} on validation | — |

**Baselines (already evaluated on the same test histories with the same LP and audit):**
* JAC-opp (`NEURAL_JO_A3`, seeds 0–2), the best pure neural model;
* PRIOR-EM with the JAC-opp prior (`HYB_PRIOR_EM_JOBEST`), the best practical method;
* tabular EM (uniform prior);
* the train-bank posterior.

### Evaluation
* 300 test opponents × 8 streams, N ∈ {5, 10, 20, 50, 100, 200, 500}.
* ε = 0.10: the new arms are solved at ε = 0.10 only.
* Exact LP and OpenSpiel audit of every deployed strategy.
* **Regret (chips)** = V_ε − u, the mean over histories.  Neural arms use the seed mean.
* **Differences** are paired over the same histories, with bootstrap CIs over opponents (2 000 resamples).
* The fraction of attainable safe gain is reported alongside.

### Expectations
* **Z1 (primary):** ZC-DEC has lower regret than JAC-opp averaged over N ∈ {5, 10, 20}, with the 95% CI of the
  difference below 0.
* **Z2:** ZC-DEC beats ZC-DISTILL averaged over N ∈ {5, 10, 20}, with the CI below 0.
* **Z3:** at N = 500, ZC-DEC's fraction is ≥ 0.85, approaching its true-policy ceiling of 0.889.
* **Z4:** PRIOR-EM (JAC-opp prior) beats ZC-DEC at N ≥ 100 (averaged over N ∈ {100, 200, 500}, CI of ZC-DEC − PRIOR-EM
  above 0).
* **Z5:** PRIOR-EM (ZC prior) is worse than PRIOR-EM (JAC-opp prior) at every N, because the regret decoder's
  policies are poor behavioural priors.
* **Practical verdict.**  "Decision compression helps in play" if the best ZC arm (ZC-DEC, ZC-DEC-4 or
  PRIOR-EM (ZC prior)) beats PRIOR-EM (JAC-opp prior) at some N with the CI below 0.  Otherwise the benefit is
  confined to the neural model (Z1) or to the true-policy setting.
* **Prior guess (not a test):**
  - Z1 is roughly a coin flip.  At small N, JAC-opp profits from mixture consistency: its N → 0 output approximates
    the population mixture.  A population mixture may not be representable by a decoder trained on individual
    opponents.
  - Z4 and Z5 are likely.

### Guardrails
* Nothing is tuned on test.
* Every deployed strategy is audited, and 0 violations are required.
* If running long, drop ZC-DEC-4 seed 1 first, then ZC-DISTILL seed 2, never ZC-DEC seeds.


## 1. Training (`zc_schedule.sh`, `outputs/runs_zcode/`)

Eight runs in two waves of 4, each 6 000 steps (79–80 min), all completed.  Selected validation regret (ε = 0.10,
mean over N = 20, 100, 500; 450 fixed LPs; lower is better):

| arm | seed 0 | seed 1 | seed 2 | JAC-opp (reference) |
|---|---|---|---|---|
| ZC-DEC (d = 8) | 0.128 | 0.128 | 0.130 | 0.112 / 0.112 / 0.111 |
| ZC-DISTILL (d = 8) | 0.119 | 0.116 | 0.117 | |
| ZC-DEC (d = 4) | 0.132 | 0.135 | — | |

* **Validation already ranked every code arm below JAC-opp,** before any test evaluation.
* **Best checkpoints came late,** at steps 4 750–6 000, as for JAC-opp.

## 2. Test results (`fig1_regret.png`, `fig2_differences.png`)

**Fraction of attainable safe gain (ε = 0.10):**

| N | 5 | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|
| ZC-DEC (d = 8) | 0.633 | 0.671 | 0.706 | 0.742 | 0.759 | 0.769 | 0.779 |
| ZC-DISTILL (d = 8) | 0.608 | 0.664 | 0.709 | 0.756 | 0.781 | 0.797 | 0.808 |
| ZC-DEC (d = 4) | 0.624 | 0.664 | 0.694 | 0.723 | 0.741 | 0.749 | 0.758 |
| PRIOR-EM (ZC prior) | 0.636 | 0.683 | 0.712 | 0.773 | 0.807 | 0.837 | 0.874 |
| JAC-opp | 0.660 | 0.699 | 0.734 | 0.771 | 0.790 | 0.802 | 0.813 |
| **PRIOR-EM (JAC-opp prior)** | 0.659 | **0.707** | **0.747** | **0.796** | **0.828** | **0.858** | **0.893** |
| TAB-EM | 0.596 | 0.629 | 0.659 | 0.722 | 0.766 | 0.807 | 0.846 |
| BANK | **0.674** | 0.704 | 0.720 | 0.736 | 0.739 | 0.746 | 0.749 |

**Paired differences in regret** (chips; positive = worse; 95% CIs over opponents; the N = 5, 20, 100, 500
columns are shown):

| comparison | N = 5 | N = 20 | N = 100 | N = 500 |
|---|---|---|---|---|
| ZC-DEC − JAC-opp | +0.013 [0.010, 0.016] | +0.014 [0.010, 0.018] | +0.015 [0.011, 0.019] | +0.017 [0.012, 0.022] |
| ZC-DISTILL − JAC-opp | +0.025 [0.019, 0.032] | +0.012 [0.009, 0.016] | +0.005 [0.001, 0.008] | +0.003 [−0.001, 0.007] |
| ZC-DEC − ZC-DISTILL | −0.012 [−0.017, −0.006] | +0.002 [−0.002, 0.005] | +0.011 [0.007, 0.015] | +0.014 [0.009, 0.020] |
| ZC-DEC (d = 4) − ZC-DEC (d = 8) | +0.004 [0.002, 0.007] | +0.006 [0.003, 0.009] | +0.009 [0.005, 0.013] | +0.010 [0.005, 0.015] |
| ZC-DEC − PRIOR-EM (JAC-opp) | +0.013 [0.009, 0.017] | +0.020 [0.016, 0.024] | +0.034 [0.027, 0.041] | +0.056 [0.046, 0.066] |
| PRIOR-EM (ZC) − PRIOR-EM (JAC-opp) | +0.011 [0.007, 0.015] | +0.017 [0.013, 0.021] | +0.010 [0.006, 0.014] | +0.010 [0.006, 0.013] |
| PRIOR-EM (ZC) − JAC-opp | +0.011 [0.008, 0.015] | +0.011 [0.007, 0.015] | −0.008 [−0.013, −0.004] | −0.029 [−0.037, −0.022] |

* **Seed ranges are tight.**  ZC-DEC at N = 5 spans 0.1776–0.1795 across seeds, and at N = 500 it spans
  0.1043–0.1098.  The ordering is not a seed effect.
* **The decision loss helps only when information is scarcest.**  At N = 5 it handles uncertainty better than
  regressing the code (−0.012).  From N = 50 on, distillation is better.
* **The exact likelihood rescues a lot, but not enough.**  Adding EM with the exact hand likelihood on top of the code
  model's prior beats pure JAC-opp from N = 100.  It stays 0.010–0.017 behind EM with the JAC-opp prior at every N,
  because the regret decoder's policies are poor behavioural priors (their per-infoset policy fraction was 0.21 in
  the compression study).

## 3. Why: post-hoc diagnostics (`zc_diag.py`, `fig3_diagnostics.png`)

### D1 — the code cannot express beliefs
* **Method.**  After N hands, the train-bank posterior mean g is a proper belief-averaged decision vector.  It is
  projected onto the frozen decoder's range: the nearest g any 8-number code can produce.  The LP is deployed on
  each; test opponents, stream 0.

| N | belief (posterior mean g) | nearest decodable g | lost |
|---|---|---|---|
| 0 (population mean) | 0.599 | 0.493 | 0.106 |
| 5 | 0.660 | 0.616 | 0.044 |
| 20 | 0.721 | 0.709 | 0.012 |
| 100 | 0.736 | 0.738 | −0.002 |
| 500 | 0.745 | 0.737 | 0.008 |

* **Reading.**  The decoder was trained on individual opponents.  Averages of opponents, which is what uncertainty
  looks like, are largely outside its range.
  - At N = 5, the lost 0.044 is more than the whole ZC-DEC − JAC-opp gap there (0.027 in fraction).
  - JAC-opp's reconstruction head can output mixtures: its N → 0 limit approximates the population mixture (the
    mixture-consistency result of the JAC-opp study).
* **Check on individual opponents.**  Projected by g-distance, they keep 0.869; the exact code keeps 0.889.

### D2 — the code is fragile
* **Noise on the true code.**  Adding Gaussian noise to the true standardized code (test opponents):

| noise σ | 0 | 0.25 | 0.5 | 1.0 |
|---|---|---|---|---|
| safe value kept | 0.889 | 0.846 | 0.764 | 0.633 |

* **How well hands pin down the code.**  ZC-DISTILL's R² for the true code (3 seeds):

| N | 5 | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|
| code R² | 0.28 | 0.44 | 0.58 | 0.71 | 0.77 | 0.81 | 0.84 |

* **The two line up.**  At N = 500 the unexplained 16% of the variance corresponds to σ ≈ 0.40.  The noise curve
  puts that at about 0.80, and ZC-DISTILL achieves 0.808.  The large-N shortfall against the 0.889 ceiling is almost
  entirely estimation error amplified by a dense code.

## 4. Expectations

| test | result | holds? |
|---|---|---|
| **Z1:** ZC-DEC < JAC-opp over N ∈ {5, 10, 20}, CI < 0 | +0.0135 [0.011, 0.016] | **no** (opposite direction) |
| **Z2:** ZC-DEC < ZC-DISTILL over N ∈ {5, 10, 20}, CI < 0 | −0.0046 [−0.008, −0.001] | **yes** (driven by N = 5) |
| **Z3:** ZC-DEC fraction ≥ 0.85 at N = 500 | 0.779 | **no** |
| **Z4:** PRIOR-EM (JAC-opp) beats ZC-DEC over N ∈ {100, 200, 500}, CI > 0 | +0.044 [0.036, 0.053] | **yes** |
| **Z5:** PRIOR-EM (ZC prior) worse than PRIOR-EM (JAC-opp prior) at every N | +0.010 … +0.017 | **yes** |
| **Practical:** some ZC arm beats PRIOR-EM (JAC-opp) at some N | none | **no** |

The prior guess called Z1 a coin flip, citing mixture consistency as the risk.  That risk is what materialised (D1).

## 5. What this means

1. **Compression and inference pull in opposite directions.**
   - The compression study showed that decisions need far less information than behaviour, *when the opponent is
     known*.
   - This study shows the other side.  The same density that makes the code compact makes it a poor inference
     target: no room for uncertainty, and every error counts.
2. **The earlier conclusion stands, with a sharper reason.**  From hands in Leduc, the best pipeline keeps inference
   in behaviour space (a learned prior plus the exact likelihood) and only turns beliefs into decisions at the end
   (LP on the posterior-mean g).
3. **What could still make decision compression useful for play** (not tested):
   - **Predict a distribution over codes instead of one code,** then deploy on the average of the decoded g's.  That
     average is exact because g is linear, and it addresses D1.  It needs a way to score codes against hands.
     - The regret decoder's policies are poor behavioural models (0.21 policy fraction), so the exact hand likelihood
       cannot be used on them directly.
     - A decoder that is both decision-sufficient and behaviourally faithful would be needed.  The compression study
       showed those goals conflict at small d.
   - **Use decision compression where behaviour cannot be represented at all** (much larger games), or where the
     opponent is known or cheaply identified and only the response must be stored or communicated.
4. **The regret-driven curriculum idea is not affected.**  It targets the code's coverage given the true policy, not
   inference from hands.

## 6. Audits and wall-clock

**Audits.**  Every deployed test strategy is an exact LP solution audited with OpenSpiel's exact best response:
* 8 models × 16 800;
* the PRIOR-EM (ZC) hybrid, 16 800;
* diagnostics, 4 500.

Total 155 700: max Expl − ε = 2.9e−9, **0 violations**, 0 LP failures.

**Wall-clock (4 CPUs):**

| stage | wall-clock |
|---|---|
| Training, 2 waves of 4 runs | 2 h 39 min |
| Test evaluation, 8 models (predict + 16 800 LPs each) | 38 min |
| PRIOR-EM (ZC prior): κ selection on val + test + solve | 14 min |
| Diagnostics | 2.4 min |
| **Total** | **≈ 3.6 h** |

## 7. Deviations and notes

1. **Diagnostics D1 and D2 are post-hoc,** added to explain the negative result.  They do not enter any verdict.
2. **Baselines were reused, not re-run.**  JAC-opp, PRIOR-EM (JAC-opp prior), TAB-EM and BANK come from their
   existing solves on the same test histories, with the same LP and audit.
3. **The new arms were solved at ε = 0.10 only,** as pre-registered.
4. **Head capacity.**  The code head has 810 656 trainable parameters, plus the 179 120-parameter frozen decoder,
   against JAC-opp's 889 089.  Encoders are identical (618 752).
5. **Training was faster than the 2 h estimated** (80 min per run), so the whole plan ran as registered.
