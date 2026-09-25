# Change detection + reset vs the oracle reset (Leduc, evaluation only)

*(Results sections are appended below the pre-registration after the run.)*

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
