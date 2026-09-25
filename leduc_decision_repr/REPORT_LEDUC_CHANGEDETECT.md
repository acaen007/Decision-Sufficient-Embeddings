# Change detection + reset vs the oracle reset (Leduc, evaluation only)

## Summary and decision

**Pre-registered verdict: INCONCLUSIVE.  The better classical detector lands 0.02 above the no-go line.**
* **The rule.**  The ratio of recovery times, R₈₀(detector) / R₈₀(oracle reset), on the 40 contrasting
  SWITCH pairs:
  - no-go: ≤ 1.5;
  - go: > 2.0;
  - anything in between is inconclusive.
* **CUSUM change-detect + reset (CPD-PRIOR-EM):**
  - R₈₀ 19.4 hands [14.6, 49.2] vs the oracle's 10.3, a ratio of **1.88** [1.22, 3.84]: inconclusive.
  - This triggered the pre-registered BOCPD arm.
* **Bayesian online change-point detection (BOCPD-PRIOR-EM), calibrated hazard 1/100:**
  - R₈₀ 15.6 hands [8.1, 36.4], a ratio of **1.52** [1.07, 2.56]: still inconclusive, by 0.018.
  - The robustness numbers fall on either side of the 1.5 line:
    - the fine checkpoint grid gives 1.48;
    - random pairs give 1.49;
    - a post-hoc check with BOCPD's change-point grid moved off the true switch time gives 1.57.

**What the numbers say beyond the rule: the best classical detector already matches a detector that knows
both opponents exactly.**
* **BOCPD ties the oracle detector.**
  - KNOWN-CUSUM-RESET runs a CUSUM on the exact likelihood ratio of the true q_A and q_B.  It recovers in
    14.9 hands (ratio 1.45).
  - Over the first 100 hands after the switch, BOCPD's fraction differs from it by −0.000 [−0.018, +0.017].
    Over all post-switch hands the difference is −0.002 [−0.015, +0.011].
* **What is left to the oracle is switch-time uncertainty.**
  - Both detectors trail the oracle reset by the same 0.039 of post-100 fraction, about 5 hands of R₈₀.
  - That is the size of the information floor.  The exact per-hand KL divergence between the switched
    opponents is 1.38 nats (median, contrasting).  At one false alarm per ~1 000 hands this gives Lorden's
    detection delay of **5.0 hands**, and the oracle detector's observed median delay is 5 hands.
* **Consequence for a learned switching model.**
  - To close the remaining gap it would have to beat a detector that is *told* both policies.
  - The part of the gap a learned model could plausibly claim, above what a classical Bayesian detector already
    gets, is ≈ 0 on these processes.
* **Where the gap came from.**  The practical improvement over the non-stationary study's best forgetting method
  is large.  Post-100 fraction:
  - PRIOR-EM-WIN 0.609;
  - CUSUM 0.653;
  - BOCPD **0.721** (+0.112 [0.074, 0.161] vs the window);
  - oracle 0.760.

**The price is small, and the Bayesian detector also handles drift best.**
* **Stationary cost vs full-history PRIOR-EM (40 stationary opponents):**
  - CUSUM 0.007;
  - BOCPD 0.011;
  - 50-hand window 0.074;
  - discount 0.98 0.050.
* **Drift:** BOCPD has the highest mean fraction over hands 100–500 (0.850).
  - vs the 50-hand window: +0.027 [0.016, 0.038];
  - vs CUSUM: +0.021 [0.012, 0.031].

**Recommendation.**
* **Do not prioritize a learned switching model** on the strength of this benchmark.  The rule is inconclusive,
  but what remains is essentially switch-time information that no detector can recover.
* **Adopt BOCPD + PRIOR-EM as the non-stationary baseline.**
* **The remaining room is in the oracle's own 10 hands:** how fast the *new* opponent is identified from a
  handful of hands.  That is the small-sample representation problem, which a better generative opponent space
  would improve for every method at once.

**Pre-registered expectations:**
* **C2, C3 and C5 hold.**
* **C1 fails narrowly.**  CUSUM's post-100 gain over the window is +0.044 [−0.005, +0.101].
* **C4 fails:** the detectors are *not* worse than windows on drift.
* **Safety:** 37 016 deployed test strategies audited, max Expl − ε = 9.3e−11, 0 violations, 0 LP failures.

## 0. Pre-registration

Written and committed before any calibration process was generated and before any detector was run on any
process.  It is not edited afterwards; deviations go in the deviations section.

### Question
The non-stationary study found a 3.2× gap on contrasting SWITCH pairs:
* the best practical forgetting methods need about 33 hands to regain 80% of their pre-switch fraction
  (R₈₀);
* POST-PRIOR-EM, which is *told* the switch time, needs 10.3 hands.

Can an explicit classical change detector that resets PRIOR-EM close most of that gap?
* If it can, there is little room for a learned switching model.
* If it cannot, learning to switch is a genuine problem.

### Processes
* **Test (unchanged from the non-stationary study; `outputs/nonstat/processes.npz`):**
  - 60 SWITCH processes (40 contrasting + 20 random pairs, switch after hand 200);
  - 40 DRIFT processes;
  - 2 streams × 500 hands each.
* **TEST-STAT (new stationary control):** 40 opponents drawn without replacement from the same test pool
  (300 test opponents + 100 FAR-ARCH archetypes) with `default_rng([2027, 3])`.  2 streams × 500 hands,
  split id 34.
* **Calibration (never test):**
  - **Pool:** the 150 validation-split opponents plus 48 fresh archetypes (12 per kind, generator indices
    k = 100–111; the test pool uses k = 0–24).  This keeps the test pool's 3 : 1 mix.
  - **CAL-SWITCH:** 40 contrasting + 20 random pairs, drawn with `default_rng([2027, 2])`.  "Contrasting"
    means above the median pairwise ‖g_A − g_B‖ of the calibration pool.  Switch after hand 200; split id 32.
  - **CAL-STAT:** 40 opponents, split id 33.
  - 2 streams × 500 hands each.
* **Simulation:** the mixture simulator of the non-stationary study, with the learner playing its blueprint.

### Estimator shared by every reset method
PRIOR-EM (κ = 3) on a *segment* of hands [s, t]:
* the prior is the JAC-opp (`runs_jacopp/a3_s0`) reconstruction head applied to the segment's hands;
* the counts come from the same hands.

The methods differ only in s:

| method | segment start s at checkpoint t | status |
|---|---|---|
| PRIOR-EM (full) | 1 | recomputed (non-stationary study) |
| PRIOR-EM-WIN (W = 50) | t − 49 | recomputed |
| POST-PRIOR-EM (**oracle reset**) | 201 if t > 200, else 1 | recomputed |
| CPD-PRIOR-EM (**change-detect + reset**) | change point estimated at the detector's most recent alarm; 1 before any alarm | new |
| KNOWN-CUSUM-RESET (**oracle detector**) | as CPD, but the detector knows q_A and q_B exactly | new |

**DISC-PRIOR-EM (new; "discounted PRIOR-EM"):**
* counts weighted γ^(t−h) over all hands;
* prior from JAC-opp on the last 1/(1−γ) hands;
* γ = 0.98 (50 hands) is the pre-declared default, and γ = 0.95 (20 hands) is reported as grid.

### Detector (CPD): CUSUM on the one-step-ahead predictive likelihood ratio

S_t = max(0, S_{t−1} + clip(log p(h_t | alt) − log p(h_t | cur), ±30) − c).
* **cur:** the JAC-opp amortized posterior predictive (the behavioural policy from the reconstruction head)
  computed from the current segment's hands before t.
  - The mixture-consistency result of the JAC-opp study is why this approximates the Bayesian posterior
    predictive.
* **alt:** the same predictive computed from the W_a hands before t.
  - W_a = 0 means the exact population prior predictive (a mixture over the 1 200 training opponents), i.e.
    "a fresh opponent".
* **Likelihood:** the exact hidden-card-marginalized hand likelihood (`baselines/likelihood.py`).  Factors
  that do not depend on the opponent cancel in the ratio.
* **Refit schedule:** predictive models are rebuilt every 5 hands.  Each hand in a 5-hand block is scored
  with models built from hands before the block.  Empty segments use the population predictive.
* **Alarm:** S_t > τ triggers an alarm.
  - The new segment starts at the hand after the last time S was 0 (the CUSUM change-point estimate).
  - S restarts at 0 at the next block, once the new segment's model exists.
* **Monitoring:** detection continues after an alarm, so later switches or false alarms can trigger again.

### KNOWN-CUSUM-RESET (oracle detector; a floor for detect-then-reset, not a practical method)
* **Statistic:** the same CUSUM on the exact per-hand log-likelihood ratio log p(h | q_B) / p(h | q_A),
  with c = 0.
* **Threshold:** h = log 1000, so the mean time between false alarms is ≥ 1 000 hands (Lorden).
* **Resets:** it resets to its change-point estimate.  After a pre-switch false alarm it restarts; it stops
  monitoring after its first alarm after hand 200.

### Calibration (CAL processes only)
* **Grid:** W_a ∈ {0, 10, 25} × c ∈ {0, 0.05, 0.15} × τ ∈ {2, 3, 4, 6, 8}, 45 configurations.
* **Selection:** maximize F_switch, the mean pooled fraction over the 9 SWITCH checkpoints on all 60
  CAL-SWITCH processes.
  - Constraint: the stationary cost F_stat(PRIOR-EM) − F_stat(CPD) must be ≤ 0.02 on CAL-STAT, where
    F_stat is the mean pooled fraction over checkpoints 100–500.
  - If no configuration qualifies, take the one with the smallest stationary cost.
* **Calibration LPs are exact but not audited:** they are never deployed in reported results.

### Test evaluation
* **Deployment:** ε = 0.10, exact ε-safe LP and OpenSpiel audit of every deployed test strategy.
* **Checkpoints:**
  - SWITCH: the original t ∈ {100, 200, 205, 210, 220, 250, 300, 400, 500} (primary), plus
    {215, 225, 230, 240, 260, 280} for a fine-grid robustness check;
  - TEST-STAT: {100, 200, 300, 400, 500};
  - DRIFT: the original {20, 50, 100, 200, 300, 400, 500}.
* **Metrics (as in the non-stationary study):**
  - pre, post-100, post-all, R₅₀ and R₈₀;
  - ratio-of-means fractions with 2 000 bootstrap resamples over processes;
  - paired differences.
* **Detection metrics:**
  - delay (first alarm after hand 200, minus 200) and the share detected within 25 / 50 / 100 hands;
  - change-point error;
  - false alarms per 1 000 stationary hands (pre-switch hands and TEST-STAT).
* **Information floor:**
  - per pair, the exact per-hand KL(p_B ‖ p_A) between hand-type distributions under the blueprint;
  - Lorden's approximate detection delay log(1000) / KL;
  - KNOWN-CUSUM's observed delays.
* **Hindsight:** the whole grid is also evaluated on test SWITCH and TEST-STAT, labelled hindsight.  It does
  not decide.
* **Verification:**
  - The recomputed full, WIN and oracle estimates at the original checkpoints must reproduce the
    non-stationary study's g_hat (maximum absolute difference reported) and its R₈₀.
  - The decision uses the recomputed oracle.

### Expectations (test, contrasting pairs unless stated)
* **C1:** CPD-PRIOR-EM has a lower R₈₀ than PRIOR-EM-WIN and a higher post-100 (paired CI > 0).
* **C2:** CPD's pre (t = 200) is ≥ full PRIOR-EM's − 0.02, and above PRIOR-EM-WIN's (paired CI > 0).
* **C3:** on TEST-STAT, CPD's mean fraction over checkpoints is within 0.02 of full PRIOR-EM's.
* **C4:** on DRIFT, CPD's mean fraction over t = 100–500 is below PRIOR-EM-WIN's (point estimate).  Detectors
  built for abrupt change are the wrong tool for drift.
* **C5:** 0 audit violations.
* **Prior guess (not a test):** ratio 1.3–2.0.

### Decision rule (calibrated configuration, test contrasting pairs, original checkpoint grid)
The ratio is R₈₀(CPD-PRIOR-EM) / R₈₀(POST-PRIOR-EM).
* **≤ 1.5 (about ≤ 15 hands): no-go.**  Classical detection closes most of the gap, so there is little room
  for a learned switching model.
* **> 2.0 (about > 21 hands), or CPD does not recover: go.**  Learning to switch is a genuine problem.
* **1.5 < ratio ≤ 2.0: inconclusive.**  Then run one stronger classical arm, BOCPD, and apply the same bands
  to the better of the two detectors.
  - BOCPD is the Adams–MacKay run-length posterior.  Its constant hazard is calibrated on CAL over
    {1/100, 1/250, 1/1000}.
  - Its run-length predictive is the same encoder predictive on candidate segments starting on a 5-hand
    grid, keeping the 20 most probable run lengths.
  - It deploys the posterior mixture of g over run lengths, which is exact because g is linear in the
    realization plan.
  - If the better detector is still inconclusive, the result is reported as inconclusive.
* **Qualifier:** if CPD's TEST-STAT cost exceeds 0.03, the verdict is reported as qualified: its R₈₀ was
  bought with stationary loss.
* **Floor note (reported, not deciding):** if KNOWN-CUSUM-RESET's R₈₀ itself exceeds 1.5× the oracle's, part
  of the oracle gap is unreachable by any detect-then-reset method.  The room left for a learned model is
  then also stated against KNOWN-CUSUM-RESET.

### Guardrails
* Nothing is tuned on test processes: the detector is calibrated only on CAL processes.
* Every deployed test strategy is audited (Expl ≤ ε + 1e−7), with 0 violations required and LP failures
  reported.
* Wall-clock per stage is reported.
* If running long, thin the τ grid, never processes or methods.

## 1. Processes, verification and calibration (`cpd_eval.py`)

### Processes
* **Test processes:** the 60 SWITCH and 40 DRIFT processes of the non-stationary study, unchanged.
* **TEST-STAT:** 40 stationary opponents from the test pool.
* **Calibration processes:**
  - Pool: 150 validation opponents + 48 fresh archetypes (k = 100–111); median pairwise ‖g_A − g_B‖ 1.506.
  - CAL-SWITCH: 40 contrasting pairs (median distance 1.87, range 1.52–4.17) + 20 random pairs (median 1.51,
    range 0.67–2.72).
  - CAL-STAT: 40 opponents.
* **Exclusions:** no checkpoint of any process had headroom V_ε − V_0 < 0.01, so nothing was excluded.

### Verification
* **Reproduction.**  The recomputed full-history, window-50 and oracle-reset estimates reproduce the
  non-stationary study's g_hat to a maximum absolute difference of 4.1e−7, which is float32 storage precision.
  They also reproduce its R₈₀ (oracle 10.3, window 32.7).
* **Information floor.**  The exact hand-type distributions agree with the hidden-card likelihood's log-ratios
  to 7e−15.

### Calibration: CUSUM (45 configurations; CAL only)
* **Full-history reference:** F_switch = 0.588 and F_stat = 0.856.
* **The grid is flat on switch processes:** every configuration scores F_switch 0.68–0.74.  Their stationary costs
  range from −0.000 to +0.098.
* **Constraint:** 28 configurations meet the stationary-cost limit (≤ 0.02).
* **Selected: W_a = 10, c = 0.15, τ = 3.**
  - F_switch 0.732 and stationary cost +0.018.
  - 44 alarms in 40 000 CAL-STAT hands.
* **Why nothing better was available:** the only configurations with higher F_switch (up to 0.742) cost
  0.022–0.098 on stationary opponents.

### Calibration: BOCPD (3 hazards; CAL only)

| hazard | F_switch | F_stat cost |
|---|---|---|
| **1/100 (selected)** | **0.785** | **+0.009** |
| 1/250 | 0.780 | +0.002 |
| 1/1000 | 0.768 | −0.000 |

All three hazards satisfy the constraint and beat every CUSUM configuration on F_switch.

## 2. Main result: recovery after a switch (test SWITCH, `fig1_switch.png`, `fig2_recovery.png`)

**Definitions:**
* **pre** = fraction at t = 200;
* **post-100** = mean fraction over t ∈ {205, 210, 220, 250, 300};
* **post-all** = mean over t = 205–500;
* **R₅₀ / R₈₀** = hands after the switch to regain 50% / 80% of the method's *own* pre, on the original
  checkpoint grid.

95% bootstrap CIs over processes.

**Contrasting pairs (40, primary):**

| method | pre | post-100 | post-all | R₅₀ | R₈₀ [95% CI] | ratio to oracle |
|---|---|---|---|---|---|---|
| PRIOR-EM (full history) | 0.888 | 0.397 | 0.492 | 41 | 225 [125, ∞] | 21.9 |
| PRIOR-EM-WIN (W = 50) | 0.848 | 0.609 | 0.663 | 7 | 32.7 [19.1, 42.8] | 3.18 |
| DISC-PRIOR-EM γ = 0.98 | 0.863 | 0.608 | 0.668 | 7 | 37.3 [21.2, 48.2] | 3.63 |
| DISC-PRIOR-EM γ = 0.95 (grid) | 0.813 | 0.675 | 0.704 | 5 | 14.6 [6.7, 25.6] | 1.42 |
| **CPD-PRIOR-EM (CUSUM)** | 0.878 | 0.653 | 0.708 | 6 | **19.4** [14.6, 49.2] | **1.88** |
| **BOCPD-PRIOR-EM** | 0.887 | **0.721** | **0.762** | 5 | **15.6** [8.1, 36.4] | **1.52** |
| *BOCPD, grid offset 2 (post-hoc)* | 0.885 | 0.709 | 0.753 | 5 | 16.2 [9.1, 36.6] | 1.57 |
| *KNOWN-CUSUM-RESET (oracle detector)* | 0.888 | 0.721 | 0.764 | 5 | 14.9 [8.8, 24.4] | 1.45 |
| *POST-PRIOR-EM (oracle reset)* | 0.888 | 0.760 | 0.792 | 5 | 10.3 [5.0, 26.7] | 1 |

**Fine checkpoint grid (robustness; adds 215, 225, 230, 240, 260, 280), R₈₀:**

| method | R₈₀ | ratio |
|---|---|---|
| CUSUM | 19.0 | 1.84 |
| BOCPD | 15.2 | **1.48** |
| offset BOCPD | 16.0 | 1.55 |
| known-model detector | 13.3 | 1.29 |
| window | 31.1 | |
| oracle | 10.3 | |

**Random pairs (20):**

| method | R₈₀ | ratio |
|---|---|---|
| CUSUM | 45.2 | 2.07 |
| BOCPD | 32.5 | 1.49 |
| known-model detector | 20.9 | 0.96 |
| oracle | 21.8 | 1 |

* **Post-all on random pairs:** BOCPD 0.721 vs oracle 0.751.
* **Subtle switches hurt CUSUM most:** its first alarm comes late or never (see §3).

**Paired comparisons (contrasting, post-100 fraction):**

| comparison | difference [95% CI] |
|---|---|
| BOCPD − CUSUM | +0.068 [0.033, 0.105] |
| BOCPD − window 50 | +0.112 [0.074, 0.161] |
| BOCPD − discount 0.95 | +0.046 [0.019, 0.081] |
| **BOCPD − known-model detector** | **−0.000 [−0.018, +0.017]**; post-all −0.002 [−0.015, +0.011] |
| BOCPD − oracle reset | −0.039 [−0.058, −0.023]; R₈₀ +5.3 hands [1.2, 10.5] |
| known-model detector − oracle reset | −0.039 [−0.057, −0.022] |
| CUSUM − oracle reset | −0.108 [−0.153, −0.065] |
| offset BOCPD − aligned BOCPD | −0.011 [−0.025, −0.000] |
| offset BOCPD − known-model detector | −0.012 [−0.034, +0.010] |

**R₈₀ flatters methods with a low pre.**
* **The case in point:** DISC-PRIOR-EM γ = 0.95 has the lowest R₈₀ among practical methods (14.6).
* **Why:** R₈₀ is measured against the method's own pre, and γ = 0.95 has a low one (0.813).
* **On absolute measures it loses:**
  - it is 0.046 below BOCPD on post-100;
  - it pays 0.093 on stationary opponents (§4).
* **Status:** it is a grid point, not a pre-declared default, and not a candidate for the decision.

## 3. Detection and the information floor (`fig3_detection.png`)

### Detection statistics (test SWITCH)
Delay = hands after the switch to the first alarm; BOCPD "detects" when its posterior mass on segments starting
at or after hand 196 exceeds 0.5 (5-hand resolution).

| detector | pairs | median delay [IQR] | ≤ 10 | ≤ 25 | ≤ 50 | not detected by 500 | change-point error (median / median abs) |
|---|---|---|---|---|---|---|---|
| CUSUM (calibrated) | contrasting | 10 [5, 17] | 53% | 88% | 94% | 4% | +3 / 4 hands (12% early) |
| CUSUM (calibrated) | random | 13 [6, 39] | 43% | 65% | 78% | 10% | +4 / 5.5 |
| BOCPD (H = 1/100) | contrasting | 5 [5, 10] | 79% | 98% | 100% | 0% | — (soft) |
| BOCPD (H = 1/100) | random | 12.5 [5, 20] | 50% | 80% | 90% | 2.5% | — |
| known-model CUSUM | contrasting | 5 [3, 7] | 84% | 100% | 100% | 0% | 0 / 0 |
| known-model CUSUM | random | 5 [3, 8] | 78% | 93% | 95% | 0% | 0 / 1 |

### False alarms
* **CUSUM:**
  - 1.25 per 1 000 pre-switch hands (30 in 24 000);
  - 0.85 per 1 000 TEST-STAT hands (34 alarms; 22 of 80 histories);
  - on DRIFT, 1.29 alarms per history.
* **Known-model CUSUM:** 0.08 per 1 000 pre-switch hands (2 in 24 000), consistent with its guaranteed rate of
  at most one per 1 000.
* **BOCPD** has no hard alarms.  Its stationary price is the fraction cost in §4.

### The information floor
* **Per-hand KL(p_B ‖ p_A)**, exact hand-type distributions under the learner's blueprint:
  - contrasting median 1.38 nats (10th–90th percentile 0.74–2.93);
  - random median 1.13 (0.56–2.21).
* **Monte-Carlo check:** the mean log-likelihood ratio over post-switch hands agrees with the exact KL to a median
  relative difference of 4% (contrasting) and 5% (random).
* **Lorden's delay** log(1000)/KL has median **5.0 hands** (contrasting) and 6.1 (random).  The known-model
  CUSUM achieves exactly that median.
* **Delay tracks information.**
  - Spearman ρ between delay and KL: −0.53 for CUSUM and −0.62 for the known model (contrasting).
  - Pairs below the median KL take 17 hands (CUSUM) vs 7 (known model).
  - Pairs above it take 7 vs 3.
* **Leduc hands are very informative.**  About 1.3 nats per hand is why even the ideal detector needs only
  about 5 hands.

### Where the gap to the oracle goes (contrasting, post-100 fraction)

| step | from → to | change |
|---|---|---|
| Full history → 50-hand window (non-stationary study) | 0.397 → 0.609 | +0.21 |
| Window → CUSUM | 0.609 → 0.653 | +0.04 |
| CUSUM → BOCPD | 0.653 → 0.721 | +0.07 |
| BOCPD → known-model detector | 0.721 → 0.721 | +0.00 |
| Known-model detector → oracle reset | 0.721 → 0.760 | +0.04 |

The last step is the cost of not being told the switch time.

**What makes BOCPD better than CUSUM:**
* it detects faster (median 5 vs 10 hands);
* it has no hard false resets;
* it hedges across candidate change points instead of committing to one estimate (10.3 mixture components per
  deployment on average).

## 4. Stationary control and drift (`fig4_stationary_drift.png`)

**TEST-STAT, 40 stationary opponents.**  Mean pooled fraction over t = 100–500:

| method | mean fraction | cost vs full PRIOR-EM [95% CI] |
|---|---|---|
| PRIOR-EM (full) | 0.859 | — |
| **CPD-PRIOR-EM** | 0.852 | **+0.007** [0.000, 0.016] |
| **BOCPD-PRIOR-EM** | 0.848 | **+0.011** [0.005, 0.021] |
| DISC-PRIOR-EM γ = 0.98 | 0.809 | +0.050 [0.029, 0.082] |
| PRIOR-EM-WIN | 0.785 | +0.074 [0.048, 0.114] |
| DISC-PRIOR-EM γ = 0.95 | 0.766 | +0.093 [0.055, 0.148] |

* **The detectors keep nearly all of full-history's stationary value.**  Windows and discounting give up 5–9
  points.
* **Hindsight grid.**  Among all 45 CUSUM configurations on test, the only ones with ratio < 1.6 cost 0.057–0.074
  on stationary opponents.  No CUSUM configuration is both fast and cheap.
  - BOCPD hazards on test: 1/100 → 1.52, 1/250 → 1.62, 1/1000 → 1.76.
  - Their stationary costs are 0.011, 0.005 and 0.002.

**DRIFT, 40 processes.**  Mean pooled fraction over t = 100–500:

| method | mean fraction |
|---|---|
| **BOCPD** | **0.850** |
| DISC 0.98 | 0.841 |
| CPD | 0.829 |
| WIN | 0.823 |
| DISC 0.95 | 0.806 |
| full PRIOR-EM | 0.796 |

* **Paired differences for BOCPD:**
  - vs WIN: +0.027 [0.016, 0.038];
  - vs CPD: +0.021 [0.012, 0.031];
  - vs DISC 0.98: +0.009 [−0.002, +0.020].
* **Under drift, BOCPD's run-length posterior acts as an adaptive window.**
* **C4 expected detectors to be worse than windows on drift; they are not.**  CPD − WIN is +0.006
  [−0.009, +0.020].

## 5. Expectations and decision

| test | result | holds? |
|---|---|---|
| **C1:** CUSUM R₈₀ < window R₈₀ and post-100 gain CI > 0 | R₈₀ 19.4 < 32.7, but post-100 +0.044 [−0.005, +0.101] | **no** (narrowly) |
| **C2:** CUSUM pre ≥ full − 0.02 and > window pre | −0.010 [−0.029, +0.003]; vs window +0.030 [0.011, 0.051] | **yes** |
| **C3:** TEST-STAT cost within 0.02 | +0.007 [0.000, 0.016] | **yes** |
| **C4:** CUSUM below window on drift | +0.006 [−0.009, +0.020] | **no** |
| **C5:** 0 audit violations | 0 of 37 016 | **yes** |
| **Prior guess:** ratio 1.3–2.0 | CUSUM 1.88, BOCPD 1.52 | inside |

**Decision rule.**
* **CUSUM:** ratio 1.88 → inconclusive, so the BOCPD arm was run.
* **The better detector is BOCPD** (lower R₈₀): ratio **1.518** [1.07, 2.56] → **inconclusive.**
* **Not qualified:** its TEST-STAT cost is 0.011 ≤ 0.03.
* **Floor note:** KNOWN-CUSUM-RESET's ratio is 1.45 (≤ 1.5), so the pre-registered floor condition is not
  triggered.  But BOCPD is statistically indistinguishable from it.  The room left for a learned model, measured
  against the known-model detector, is −0.000 [−0.018, +0.017] of post-100 fraction.

## 6. Audits and wall-clock

**Audits.**  Every deployed test strategy went through the exact ε-safe LP (ε = 0.10) and an OpenSpiel exact
best-response audit:
* 25 856 unique strategies (CUSUM stage);
* 9 360 BOCPD mixtures;
* 1 800 offset-BOCPD mixtures.

Total **37 016**:
* max Expl − ε = 9.3e−11;
* **0 violations**;
* 0 LP failures.

**Calibration LPs** (10 438 CUSUM estimates, 4 440 BOCPD mixtures) were exact but not audited, as pre-registered.

**Wall-clock:**

| stage | wall-clock |
|---|---|
| Build (processes, oracle values) | 0.6 min |
| CUSUM calibration: 45-configuration lockstep detector on 200 histories + 10 438 LPs | 7.9 min |
| CUSUM test: grid on 200 histories, DRIFT, known-model CUSUM + 25 856 LPs with audit | 13.7 min |
| BOCPD calibration: 3 hazards, 18 098 segment fits + 4 440 LPs | 12.8 min |
| BOCPD test: 32 614 segment fits + 9 360 audited LPs | 18.1 min |
| Offset-BOCPD check | 7.9 min |
| Analysis (2 000 bootstrap resamples) + figures | ~1 min |
| **Total** | **≈ 62 min** |

## 7. Deviations and notes

1. **Conditional arm run as pre-registered.**  CUSUM's ratio fell in the inconclusive band, so the BOCPD arm ran.
2. **Implementation details the pre-registration left open.**  All were fixed in `cpd_bocpd.py`, committed before
   the BOCPD test run.
   - **Pruning:** starts below 1e-10 of the most probable are dropped, in addition to the top-20 cap.
   - **Mixture:** components below 1e-3 posterior mass are dropped and the rest renormalized.
   - **What is deployed:** the posterior over which segment produced the hands seen so far.  It is scored against
     the current opponent, so no fresh-start term is included.
   - **Hazard:** the per-block hazard 1 − (1 − H)^5 is placed at the 5-hand grid points.
3. **Grid alignment (post-hoc check).**
   - The true switch (after hand 200) lies on BOCPD's 5-hand change-point grid; CUSUM's change points are not
     gridded.
   - A post-hoc check moved the grid by 2 hands, so the switch falls between grid points.  It cost 0.011 of
     post-100 and 0.6 hands of R₈₀ (ratio 1.57).  The result stays indistinguishable from the known-model detector.
   - It does not change the verdict: inconclusive either way.
4. **"Better detector" was chosen on test by R₈₀,** as the pre-registration specifies.  This choice between two
   classical detectors favours the no-go side.  It did not matter: both detectors land in the inconclusive band.
5. **CAL-STAT opponents** were drawn with `default_rng([2027, 4])`; the pre-registration fixed only the
   CAL-SWITCH seed.
6. **BOCPD detection delay** (posterior mass > 0.5 on a segment starting at or after hand 196) is a descriptive
   metric added for §3.  It was not pre-registered.
7. **R₈₀ is relative to each method's own pre-switch level.**  This is the pre-registered decision metric, but it
   rewards methods with low stationary performance (see DISC γ = 0.95).  Absolute post-100 and post-all fractions
   are reported alongside.
8. **Scope.**
   - One switch per process, after 200 hands, between fairly distinct opponents.
   - Hands carry about 1.1–1.4 nats of information about the switch.
   - Learned switching models could matter more when these conditions fail: frequent or structured switching,
     partial switches, much less informative observations, or opponents that switch *in response to us*.  None
     of these were tested.

## 8. What this means for the learned-switching idea

1. **On abrupt switches of the kind tested, a classical Bayesian change-point filter is about as good as one that
   knows both opponents exactly.**  It uses the learned amortized predictive (JAC-opp) and PRIOR-EM.  Its gap to
   the oracle reset is roughly the size switch-time uncertainty alone imposes.
   - Training a sequence model on switching opponents could at best win back a few hands.  To do so it would have
     to beat an oracle detector.
   - The pre-registered rule calls this inconclusive, but the substance points to little room.
2. **BOCPD + PRIOR-EM should be the non-stationary baseline from now on:**
   - within 0.04 of the oracle reset after a switch;
   - 0.011 stationary cost;
   - best on drift.
3. **The remaining large term is the oracle's own recovery: about 10 hands to identify a *new* opponent.**  That
   is the small-sample identification problem.  A better opponent representation or prior speeds it up for every
   method, stationary or not.  This includes the generative opponent-space decoder and teacher-curriculum ideas
   discussed earlier.  That is where further effort should go.
