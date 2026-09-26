# Predicting a decision code from hands (Leduc)

*(Results are appended below the pre-registration after the run.)*

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
