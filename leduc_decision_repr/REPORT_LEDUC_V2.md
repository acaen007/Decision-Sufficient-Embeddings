# REPORT_LEDUC_V2 — End-to-end safe-response training (SAFE_REGRET) vs G_MSE vs RECON

*Extension of REPORT_LEDUC_V1: a third neural method trained directly on exact-oracle safe-response regret
through a differentiable regularized safe solver, evaluated with the unchanged exact ε-safe LP and OpenSpiel
audit on the same 300 held-out opponents.*

## 1. Question

V1 found that predicting the decision vector g(q) = A y_q (G_MSE) beats reconstructing the policy (RECON) at
identical certified safety, and that the advantage tracked the accuracy of ĝ.  V2 asks the sharper question:
if the encoder and head are trained on the *actual* safe-response regret rather than on ĝ accuracy, does the
resulting method outperform G_MSE **even when its predicted g is numerically less accurate**?  A positive
answer shows directly that predicting g accurately is not the same as learning exactly the information needed
to choose the correct safe response.

Everything from V1 is kept frozen: population, splits, observation streams, the hierarchical Transformer
encoder (z ∈ R¹²⁸), the decision head architecture, 6 000 steps × batch 32, AdamW 3e−4 with warm-up and cosine
decay, seeds {0, 1, 2}, ε ∈ {0.05, 0.10, 0.20}, the exact safe LP and the OpenSpiel best-response audit at
evaluation.  RECON and G_MSE are **not** retrained: their V1 test predictions and LP solutions are reused
verbatim, so the three-way comparison uses identical histories, opponents and solver.

## 2. Method 3: SAFE_REGRET

```
H_N → encoder → z → decision head → ĝ → differentiable regularized ε-safe solver → x̂_ε
L_response = V_ε(q) − x̂_εᵀ g(q)          (V_ε(q): exact oracle safe value from V1)
```
ε is sampled per training sample uniformly from {0.05, 0.10, 0.20}.  Nothing about safety is learned: the
solver's feasible set is exactly the LP's, and **the deployed / evaluated strategy always comes from the
original exact unregularized safe LP applied to ĝ**, exactly as for G_MSE.

### 2.1 Differentiable regularized safe solver (training only)

For predicted ĝ, budget ε and regularization τ > 0 we solve

```
max_{x, v}   ĝᵀx − τ/2 ‖x‖² − τ_v/2 ‖v‖²
s.t.         E x = e,  x ≥ 0,  Fᵀv − Aᵀx ≤ 0,  v₀ ≥ v* − ε
```

i.e. the V1 ε-safe LP (same E, F, A, v*) with a strictly concave quadratic term on the realization plan x.
The negligible term on the dual-side variables v (τ_v = 10⁻⁴ τ) makes the optimum unique in (x, v) and the
KKT system non-singular; it moves x* by ≤ 3e−4 in norm (measured) and does not change the feasible set, so
every regularized solution is exactly ε-safe (verified by the OpenSpiel audit in the tests: e(x̂) ≤ ε + 1e−6
for τ ∈ {0.1, 0.01, 1e−4}).  As τ → 0 the solution converges to the LP optimum (value within 2e−3 at
τ = 1e−4; identical at 1e−4 for most instances).

*Forward*: Clarabel (interior point; tolerances 1e−8; 15–25 iterations; ≈ 21 ms per solve).  OSQP (ADMM) and
HiGHS-QP were benchmarked and rejected (0.7 s per solve with 1e−5 feasibility violations; wrong solutions,
respectively).

*Backward*: implicit differentiation of the KKT conditions on the active set.  With H = diag(τI, τ_vI),
G_a the rows of the equality constraints plus the inequality rows classified active (dual larger than slack,
the interior-point heuristic), and K = G_a H⁻¹ G_aᵀ (+ 1e−9/τ · I against degeneracy), the gradient of
L = ⟨w, x*(ĝ)⟩ is
```
dL/dĝ = [ H⁻¹w̃ − H⁻¹G_aᵀ K⁻¹ G_a H⁻¹ w̃ ]_x ,   w̃ = (w, 0),
```
solved with a sparse LU (≈ 6.5 ms per sample).  Central finite differences agree to 1e−3–1e−5 relative
error away from active-set kinks (a minority of probes straddle a kink, where the piecewise-linear map has no
single derivative).  The map ĝ ↦ x*(ĝ) is piecewise linear with slope ∝ 1/τ inside each active-set region:
at τ = 0 (the LP) it is piecewise constant and the gradient vanishes almost everywhere — the vertex-seeking
problem — which is why τ > 0 is needed during training.

### 2.2 Annealing

τ_t decays **linearly** from τ_start to τ_end = 1e−4 over the 6 000 steps (`tau_schedule = linear`), so most
of training happens at moderate τ and the last ~5 % of steps at τ < 0.005.  The validation-only sweep
(seed 0, everything else identical) covers τ_start ∈ {0.01, 0.05, 0.10} plus the τ ≈ 0 control
(constant τ = 1e−4 from step 0: as close to the raw LP as the solver allows).  One schedule is selected by
validation exact-LP safe regret (§4) and then frozen for seeds 1 and 2.  For scale, τ/2 ‖x‖² is ≈ 2.5 τ chips
(‖x‖² ≈ 5 for a realization plan) against ĝᵀx ≈ 0.3–1.3 chips, so τ = 0.1 is a strong regularizer,
0.01 moderate and 1e−4 negligible.

### 2.3 Tracking (every step / every validation)

Per step: training regret, τ_t, ‖∂L/∂z‖ (gradient norm into the encoder, averaged over the batch), encoder
parameter gradient norm before clipping, and every 50 steps the behavioral determinism / entropy / fraction of
zero realization-plan entries of the regularized solutions.  Every 250 steps on validation opponents: exact-LP
safe regret and regularized-QP regret on a fixed subset (150 opponents × 1 stream × N ∈ {5, 20, 100, 500},
ε = 0.10; 600 exact LPs), validation g-NMSE at all N, determinism/entropy/active-set size of the QP solutions,
and on 32 fixed probe samples (N = 100, ε = 0.10) the fraction of QP active-set entries and exact-LP support
entries that changed since the previous validation.  **Checkpoint selection uses the exact-LP validation
regret** (the method's own objective, computed exactly), whereas V1's G_MSE and RECON used their own losses;
their best checkpoints were at or within 250 steps of the end of training, so this asymmetry is immaterial.

### 2.4 Cost

≈ 1.8–2.0 s per step (0.7 s Clarabel forward + 0.2 s KKT backward + encoder) against 0.5 s for G_MSE; each
6 000-step run takes ≈ 3 h on one CPU core plus ≈ 20 min of validation.

## 3. Protocol

1. Validation-only τ sweep (seed 0) → select one schedule by validation exact-LP regret → freeze.
2. Train seeds 1 and 2 with the frozen schedule (seed 0 = the sweep run).
3. Evaluate the three SAFE_REGRET seeds once on the 300 test opponents with the unchanged exact LP + OpenSpiel audit; reuse V1's test results for G_MSE, RECON and the classical baselines.
4. Three-way analysis (regret, F, paired CIs, g NMSE, per family, latent dimension, geometry).
No test result was inspected before step 3 completed for all methods.

## 4. Annealing / saturation: the validation-only sweep (Figure 12, `outputs/runs_v2/sweep_diagnostics.json`)

All runs: seed 0, 6 000 steps, identical data and optimizer; validation exact-LP regret on the fixed subset
(ε = 0.10, N ∈ {5, 20, 100, 500}) and, for the selection, on the full validation split
(150 opponents × 4 streams × 7 N × ε ∈ {0.05, 0.10, 0.20}; 12 600 exact LPs per run, 0 failures, audited).
Reference on the same subset: G_MSE seeds 0.133 / 0.135 / 0.136, RECON seeds 0.145 / 0.147 / 0.145.

| run | schedule | best step | best subset regret | final subset regret | full-val regret (ε>0 mean) | ‖ĝ‖/‖g‖ at best | τ at best → effective τ | g-NMSE (N=100) at best |
|---|---|---|---|---|---|---|---|---|
| SR_t0.01 **(selected)** | 0.01 → 1e−4 linear | 5 750 | **0.1605** | 0.1626 | **0.1727** | 1.7 | 5e−4 → 3e−4 | 2.0 |
| SR_t0.05 | 0.05 → 1e−4 linear | 3 250 | 0.1774 | 0.1963 | 0.1959 | 57 | 0.023 → 4e−4 | 1 245 |
| SR_t0.10 | 0.10 → 1e−4 linear | 1 000 | 0.2021 | 0.3606 (terminated at 4 300) | 0.2191 | 10 | 0.083 → 8e−3 | 87 |
| SR_const1e−4 (τ≈0 control) | 1e−4 constant | 2 000 | 0.1992 | 0.2104 | 0.2119 | 1.1 | 1e−4 → 1e−4 | 1.2 |
| SR_t0.05_fixednorm (diagnostic, outside the sweep) | 0.05 → 1e−4 linear, ‖ĝ‖ pinned | 5 750 | 0.1632 | 0.1632 | 0.1776 | 1.0 (pinned) | 2.2e−3 → 2.2e−3 | 4.4 |

**Finding A — the free-scale head anneals itself.**  With the objective ĝᵀx − τ/2‖x‖², scaling ĝ by c is
equivalent to dividing τ by c, and the loss (regret) is invariant to the scale of ĝ at τ = 0.  The network
exploits this: the effective regularization τ_eff = τ·‖g‖/‖ĝ‖ at the best checkpoints of the three annealed
runs is 3e−4, 4e−4 and 8e−3 although their scheduled τ differ by 200×, because ĝ inflated by 1.7×, 57× and
10×.  The inflation does not stop once τ has been neutralized: with nothing restoring the scale, ‖ĝ‖ keeps
drifting (g-NMSE 3e7 and 3e8 by the end of SR_t0.05 and the control; 1.6e10 in SR_t0.10, which we
terminated at step 4 300 after its validation regret had risen from 0.20 to 0.36 with the QP running at the
solver's numerical limit).  The exact LP used at deployment is scale-invariant, so this does not affect the
evaluated strategies, but it means the schedule alone does not control the smoothing.  Figure 12(b)–(c) show
the corresponding signatures: ‖∂L/∂z‖ falls by two orders of magnitude while ĝ inflates (larger τ_start →
smaller gradient into the encoder, ∝ 1/τ_eff) and then explodes late in the control and SR_t0.05 runs.

**Finding B — annealing matters; the τ ≈ 0 control stalls.**  The constant-τ control is the vertex-seeking
failure mode the RPS entropy annealing was designed to avoid: its validation regret reaches 0.199 by step
2 000 and never improves (0.210 at the end), whereas both runs that start smooth and end sharp reach
0.160–0.163.  The QP active set and LP support of the probe samples keep changing at 4–6 % of entries between
validations in every run (Figure 12(e)), so the stall is not an absence of movement between vertices but the
absence of a useful gradient direction: at τ_eff ≈ 1e−4 the map ĝ ↦ x* is flat almost everywhere.

**Finding C — closing the scale loophole gives the same result.**  The fixed-norm diagnostic (‖ĝ‖ pinned to
the mean training ‖g‖, so the scheduled τ is the effective τ) ends at 0.1632 on the subset and 0.1776 on the
full validation set, statistically indistinguishable from the selected free-scale run (0.1605 / 0.1727) and
far better than the control.  Its regularized-QP regret sits *below* its exact-LP regret throughout
(e.g. 0.183 vs 0.193 at step 1 000): the smoothed safe strategy from the same ĝ is more robust to errors in ĝ
than the LP vertex, a point taken up in §7.

**Finding D — determinism is set by ε, not by τ.**  The deployed-policy determinism of the validation
solutions is 0.76–0.80 for every run and every τ (Figure 12(d)); the regularizer changes the mixing at a
few dozen reachable infosets, not the overall shape of the safe strategy, which is dictated by the ε-safety
constraint.

**Selection.**  Lowest full-validation regret: SR_t0.01 (0.1727).  Frozen for seeds 1 and 2:
τ_start = 0.01, linear to 1e−4, no norm fixing (the fixed-norm run is outside the pre-specified sweep and is
reported only as a diagnostic).  The selected run's own validation regret at ε = 0.10 (0.164) is, however,
already well above G_MSE's (≈ 0.13): the answer to the critical question is foreshadowed on validation and
is confirmed on the test split in §5.


RESULTS_PLACEHOLDER
