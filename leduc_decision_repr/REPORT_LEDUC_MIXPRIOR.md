# Mixture-prior Bayesian opponent modelling: is the neural prior needed? (Leduc)

*Results are added below the pre-registration after the test run.  The pre-registration section is not edited after the
selection and test runs start.*

---

## Pre-registration (written before κ selection and before any test history is scored)

### Question

PRIOR-EM (tabular EM whose Dirichlet prior is centred on the JAC-opp network's decoded policy) is the best method so far.
Is the neural prior needed, or does a classical mixture prior over the training opponents do as well?

Prior: q ~ Σ_k π_k Dir(κ a_k), with π_k uniform over the 1,200 training opponents.

### Arms (no training)

| Arm | Anchors a_k (1,200) |
|---|---|
| **MIX-BANK** | the true policies of the 1,200 training opponents |
| **MIX-LATENT** | JAC-opp (seed 0, `runs_jacopp/a3_s0`) decoded q̂ of each training opponent, from training stream 0 at N = 500 (built: `outputs/mixprior/anchors_latent.npy`; mean per-infoset TV to the true policy 0.161, median 0.157) |

**Baselines, reused as they are (not re-run):**

| Baseline | In distribution | Off distribution |
|---|---|---|
| BANK | `outputs/eval/test/solve_BANK_POSTERIOR.npz` | `outputs/gen/solve.npz` "BANK" |
| TAB-EM | `outputs/eval/test/solve_TABULAR_EM_UNIFORM.npz` | "TAB-EM" |
| JAC-opp (seed 0) | `outputs/eval/test/solve_NEURAL_JO_A3_s0.npz` | "JAC-opp" |
| PRIOR-EM (JAC-opp prior) | `outputs/eval/test/solve_HYB_PRIOR_EM_JOBEST.npz` | "PRIOR-EM (JAC-opp)" |

In distribution, PRIOR-EM uses the 3-seed mean JAC-opp q̂ as prior, with κ_N ∈ {1, 3, 10, 30, 100} selected on
validation (the JAC-opp study's EM-prior arm).  Off distribution, it uses the seed-0 q̂ prior at κ = 3 (the GEN study).
The in-distribution PRIOR-EM therefore has a slightly stronger prior than MIX-LATENT (one seed).  This favours PRIOR-EM
against MIX-BANK, so a "not needed" verdict is conservative.

### Inference per history (exact choices)

1. **Score and keep.**  Score all 1,200 anchors by the exact hidden-card hand likelihood log p(H | a_k).  Keep the top
   **K(N)** anchors (see "Truncation" below).
2. **Fit each kept anchor.**  Run tabular EM with the prior Dir(κ a_k):
   * a_k is floored at 1e-6 and renormalized over the legal actions;
   * EM starts at a_k and runs until max |Δq| < 1e-7, or for at most 200 iterations;
   * M-step: q_k = (n̂ + κ a_k) / Σ.  This is the posterior mean under Dir(κ a_k), the same update as PRIOR-EM.  TAB-EM
     is the special case κ = 1, a = uniform.
3. **Weight the components.**  w_k ∝ π_k exp(ℓ_k), using the **"evidence" formula**:

   ℓ_k = log p(H | q_k) − Σ n̂_k log q_k + Σ_I [log B(κ a_{k,I} + n̂_{k,I}) − log B(κ a_{k,I})],

   where n̂_k are the expected completed-data counts at q_k (the E-step).  The formula takes the EM lower bound and
   replaces its completed-data term with the exact Dirichlet-multinomial marginal at n̂_k.  It is exact when no card is
   hidden, and tends to log p(H | a_k) as κ → ∞.

   The parameterization is Dir(κ a_k) exactly.  The +1 shift is not needed, because the evidence, unlike the density,
   is finite for any α > 0.

   **Declared deviation from the default.**  The spec's default weight is the EM objective, log p(H | q_k) +
   log Dir(q_k | κ a_k + 1).  That weight **fails S-a** (below).  At large κ its Dirichlet peak heights differ between
   anchors by far more than the data log-likelihood does, so the weights collapse onto one component.
4. **Deploy.**  Compute g_hat = Σ_k w_k g(q_k) and solve the exact ε-safe LP at ε = 0.10.

**Truncation: a declared deviation from K = 16.**  With K = 16, S-a failed against the full bank.  The top 16 anchors
hold a median of only 22% of the bank posterior at N = 5 (it takes a median of 690 anchors to cover 99% at N = 5).  The
truncated bank then loses 0.017 chips at N = 5 and 0.0035 at N = 20, which would make M1 fail by construction.

K(N) is therefore chosen on validation by a fixed rule: the smallest K in {16, 64, 128, 256} whose truncated-bank
regret is within 0.002 of the full bank's.  The rule was set after seeing the K = 16 failure, but before any mixture was
fitted at κ < 10⁴.  It gives:

| N | 5 | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|
| K(N) | 128 | 128 | 64 | 16 | 16 | 16 | 16 |

The same K(N) is used for both arms.

### Sanity checks: all pass (validation, 150 opponents × stream 0; `outputs/mixprior/sanity.json`)

**S-a: MIX-BANK at κ = 10⁴ reproduces BANK.**  Pass criterion: |regret − full-BANK regret| ≤ 0.003, and mean weight TV
to the bank posterior on the kept anchors ≤ 0.01.

| N | 5 | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|
| full BANK regret | 0.1552 | 0.1458 | 0.1375 | 0.1358 | 0.1292 | 0.1473 | 0.1446 |
| MIX-BANK (evidence) − BANK | −0.0026 | +0.0011 | −0.0003 | +0.0011 | +0.0005 | +0.0001 | +0.0003 |
| weight TV vs bank posterior | 3e-5 | 6e-5 | 1e-4 | 3e-4 | 5e-4 | 1e-3 | 4e-3 |
| *'map' weight (EM objective) − BANK* | *+0.124* | *+0.115* | *+0.095* | *+0.059* | *+0.058* | *+0.024* | *+0.009* |
| *'map' weight TV* | *0.99* | *0.98* | *0.96* | *0.90* | *0.95* | *0.82* | *0.80* |

The remaining ±0.003 differences are truncation noise on 150 histories, of mixed sign.  They are below the M1 margin.

**S-b: K = 1 with a flat anchor (κ = 1) reproduces TAB-EM.**  Max |Δq| is 4e-16 and max |Δg| is 1e-16 at N = 5, 100
and 500.  The regrets are identical.

**S-c:** the weights are finite and sum to 1 in every run.  This is re-checked in every selection and test run.

### κ selection

* κ ∈ {3, 10, 30, 100}, selected per N and separately per arm.
* Criterion: mean validation regret at ε = 0.10 over 150 validation opponents × stream 0.  This is the T5 protocol
  that selected PRIOR-EM's κ.  Ties go to the smaller κ.
* Nothing is tuned on test or on the new families.

### Data

* **In distribution:** 300 test opponents × streams 0–3 (1,200 histories).
* **Off distribution:** NEAR and FAR-EXPL, 100 opponents × 2 streams each (200 histories per family).
* N ∈ {5, 10, 20, 50, 100, 200, 500}; ε = 0.10 only.
* Baseline numbers are restricted to exactly these histories.

### Metrics

* **Regret** = V_ε − u (chips per hand).
* **Fraction** = (u − V₀)/(V_ε − V₀), pooled as a ratio of means, excluding opponents with headroom < 0.01 (counts
  reported).
* **Paired differences:** per-opponent means over streams, with 95% percentile bootstrap CIs (2,000 resamples) over
  opponents (300 in distribution; 100 per off-distribution family).
* **Effective number of components** 1/Σw² versus N.

### Hypotheses

All margins are in regret (chips per hand) unless stated otherwise.  "Within x of Y" means the point estimate of the
paired mean difference is ≤ x.

* **M1:** at N ∈ {5, 10} in distribution, MIX-BANK − BANK ≤ 0.005.
* **M2:** at N = 500, MIX-BANK's fraction is ≥ TAB-EM's − 0.02 on ID, NEAR and FAR-EXPL.  Also, MIX-BANK's fraction is
  above BANK's on NEAR and on FAR-EXPL, with the paired 95% CI excluding 0.
* **M3:** at N ∈ {5, 10, 20} in distribution, MIX-BANK − PRIOR-EM ≤ 0.005.
* **M4 (exploratory):** MIX-LATENT − MIX-BANK, per N and per family, with CIs.

**Decision rule.**

* If MIX-BANK − PRIOR-EM ≤ 0.005 at every N in all three groups (ID, NEAR, FAR-EXPL; 21 cells), the headline is
  **"the neural prior is NOT needed in Leduc"**.
* If PRIOR-EM or MIX-LATENT beats MIX-BANK by ≥ 0.01 with the paired 95% CI excluding 0 at some N in some group, the
  learned representation adds value; report where.
* If both hold (possible via MIX-LATENT), report both.  If neither holds, report the per-cell pattern without a
  headline claim.

### Budget and guardrails

**Budget.**  The benchmark (parallel EM, 4 workers, 150 validation histories) gives:

| Stage | Projected |
|---|---|
| Sanity checks (done) | 4.4 min |
| Validation κ selection (both arms × 4 κ × 7 N, EM + LP) | ≈ 20 min |
| Test EM (1,600 histories × 7 N × 2 arms at the selected κ) | ≈ 50 min (≤ 70 min if κ = 3 everywhere) |
| Test LP + audit (22,400) | ≈ 5 min |
| **Total** | **≈ 80 min (≤ 100 min worst case)** |

This is under the 1 h 45 m threshold, so **no cuts**: both arms, the full κ grid, 4 in-distribution streams, and both
off-distribution families are kept.  K is larger than the spec's 16, as declared above.

**Guardrails.**  Every deployed test strategy is audited in OpenSpiel.  0 violations (exploitability ≤ ε + 1e-7) are
required.  Validation LPs used only for κ selection are not audited, as in earlier studies.
