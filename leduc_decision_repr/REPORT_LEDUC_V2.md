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


## 5. Main comparison on the held-out test opponents (Figures 1–3, 6; Appendix B)

Three SAFE_REGRET seeds (τ_start 0.01 → 1e−4 linear), evaluated once; RECON and G_MSE numbers are V1's.
All 201 600 new deployed strategies were audited: 0 LP failures, max exploitability − ε = 9.4e−11,
0 strategies above ε + 1e−7 (Appendix B, Table R7).

**Safe response regret** (chips/hand; mean over 300 opponents × 8 streams; neural = mean of 3 seeds):

| ε | method | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 | AUC(log N) [95 % CI] |
|---|---|---|---|---|---|---|---|---|---|
| 0.05 | SAFE_REGRET | 0.134 | 0.126 | 0.119 | 0.114 | 0.111 | 0.110 | 0.109 | 0.117 [0.106, 0.127] |
| | G_MSE | 0.126 | 0.112 | 0.102 | 0.091 | 0.086 | 0.084 | 0.081 | 0.096 [0.086, 0.105] |
| | RECON | 0.132 | 0.117 | 0.107 | 0.097 | 0.093 | 0.091 | 0.089 | 0.102 [0.092, 0.112] |
| | BANK_POSTERIOR | 0.119 | 0.107 | 0.101 | 0.096 | 0.096 | 0.094 | 0.094 | 0.100 [0.091, 0.110] |
| 0.10 | SAFE_REGRET | 0.177 | 0.165 | 0.155 | 0.148 | 0.144 | 0.142 | 0.140 | 0.151 [0.138, 0.165] |
| | G_MSE | 0.167 | 0.149 | 0.135 | 0.120 | 0.112 | 0.108 | 0.105 | 0.126 [0.114, 0.137] |
| | RECON | 0.177 | 0.157 | 0.142 | 0.128 | 0.124 | 0.120 | 0.118 | 0.136 [0.123, 0.149] |
| | BANK_POSTERIOR | 0.158 | 0.144 | 0.136 | 0.128 | 0.127 | 0.123 | 0.122 | 0.133 [0.122, 0.145] |
| 0.20 | SAFE_REGRET | 0.261 | 0.243 | 0.225 | 0.213 | 0.206 | 0.202 | 0.199 | 0.219 [0.201, 0.237] |
| | G_MSE | 0.238 | 0.210 | 0.188 | 0.165 | 0.154 | 0.147 | 0.142 | 0.174 [0.159, 0.189] |
| | RECON | 0.251 | 0.224 | 0.200 | 0.180 | 0.172 | 0.165 | 0.162 | 0.190 [0.173, 0.208] |
| | BANK_POSTERIOR | 0.225 | 0.202 | 0.190 | 0.176 | 0.174 | 0.170 | 0.171 | 0.184 [0.169, 0.201] |

SAFE_REGRET seeds at ε = 0.10 (N=5 … 500): 0.177→0.141, 0.175→0.138, 0.178→0.142 — the seed spread is
≤ 0.004, far smaller than the differences between methods.

**Paired differences SAFE_REGRET − G_MSE (positive = SAFE_REGRET worse; 95 % paired-bootstrap CI):**

| ε | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 |
|---|---|---|---|---|---|---|---|
| 0.05 | +0.008 [+0.006, +0.011] | +0.014 [+0.011, +0.018] | +0.017 [+0.013, +0.022] | +0.023 [+0.018, +0.028] | +0.025 [+0.020, +0.031] | +0.026 [+0.021, +0.032] | +0.028 [+0.022, +0.033] |
| 0.10 | +0.010 [+0.006, +0.013] | +0.016 [+0.012, +0.020] | +0.021 [+0.015, +0.027] | +0.028 [+0.021, +0.035] | +0.032 [+0.025, +0.040] | +0.034 [+0.026, +0.043] | +0.035 [+0.027, +0.044] |
| 0.20 | +0.022 [+0.015, +0.030] | +0.033 [+0.026, +0.041] | +0.037 [+0.028, +0.046] | +0.047 [+0.038, +0.059] | +0.053 [+0.042, +0.064] | +0.055 [+0.044, +0.067] | +0.057 [+0.045, +0.070] |

SAFE_REGRET − RECON: −0.000 [−0.007, +0.006] at N=5 (ε=0.10), then +0.007 … +0.022 with CIs excluding 0
for every N ≥ 10 at every ε (+0.003 [−0.001, +0.007] at N=5, ε=0.05; +0.009 [−0.001, +0.019] at N=5,
ε=0.20).  SAFE_REGRET − BANK_POSTERIOR: +0.015 to +0.041, CIs excluding 0 everywhere.

**Fraction of oracle-safe gain recovered** at ε = 0.10: SAFE_REGRET 0.49 → 0.55 (N=5 → 500), G_MSE
0.54 → 0.71, RECON 0.51 → 0.71, bank 0.56 → 0.71.  N50 = 10 for SAFE_REGRET (5 for the others; 50 at
ε = 0.20); N80 and N90 are not reached by any method.  AUC of the regret curve: SAFE_REGRET is 20 % (ε=0.05),
21 % (0.10) and 26 % (0.20) above G_MSE, and 11–15 % above RECON.

**Reading.**  (i) The end-to-end method is worse than G_MSE at every budget and every safety level, and the
gap *widens* with N: SAFE_REGRET's curve is nearly flat (it recovers 0.037 chips of regret between N=5 and
N=500 at ε=0.10, against 0.062 for G_MSE and 0.059 for RECON), i.e. it extracts less from additional
hands.  (ii) At N=5 it is statistically tied with RECON and only 0.01 behind G_MSE: what it learns is close to
a good population-level safe exploit; what it fails to learn is the opponent-specific refinement.
(iii) The ranking is the same at all three ε and the deficit scales roughly with ε (the size of the safe
strategy space): +0.03 / +0.035 / +0.057 chips at N=500.  (iv) At ε = 0 (a pure equilibrium-selection check)
SAFE_REGRET's ĝ also selects slightly worse equilibria (regret 0.011 vs 0.004–0.005 for the other methods;
oracle gain 0.005).

**Per family (ε = 0.10; SAFE_REGRET − G_MSE paired, 75 opponents each):**

| family (oracle gain) | method | N=5 | N=20 | N=100 | N=500 | Δ vs G_MSE at N=5 / N=500 |
|---|---|---|---|---|---|---|
| NASH_LOGIT_PERTURB (0.194) | SAFE_REGRET / G_MSE / RECON | 0.125 / 0.113 / 0.125 | 0.123 / 0.094 / 0.099 | 0.119 / 0.078 / 0.084 | 0.118 / 0.072 / 0.074 | +0.012 [+0.008, +0.017] / +0.046 [+0.031, +0.063] |
| NASH_RANDOM_MIX (0.286) | | 0.099 / 0.092 / 0.087 | 0.089 / 0.086 / 0.081 | 0.082 / 0.074 / 0.074 | 0.078 / 0.069 / 0.070 | +0.007 [+0.003, +0.010] / +0.010 [+0.003, +0.017] |
| STRUCTURED_CORRELATED (0.799) | | 0.227 / 0.202 / 0.224 | 0.159 / 0.120 / 0.139 | 0.138 / 0.082 / 0.098 | 0.135 / 0.073 / 0.091 | +0.025 [+0.015, +0.036] / +0.063 [+0.040, +0.088] |
| UNSTRUCTURED_DIRICHLET (0.684) | | 0.256 / 0.261 / 0.272 | 0.251 / 0.239 / 0.251 | 0.237 / 0.213 / 0.240 | 0.229 / 0.207 / 0.237 | −0.005 [−0.012, +0.002] / +0.022 [+0.007, +0.039] |

SAFE_REGRET is essentially flat in N for the near-equilibrium family (0.125 → 0.118) where G_MSE halves
its regret (0.113 → 0.072); it is the only method that is (insignificantly) *better* than G_MSE at N=5 on the
unstructured family, and it tracks RECON there at all N.  The pattern is consistent with a representation that
identifies the family and a coarse exploit quickly but does not refine it with more hands.

## 6. Prediction accuracy, latent dimension and strategic geometry (test split)

**g accuracy (Table R8 in Appendix B).**  Standardized NMSE of ĝ against the true g(q) (1.0 = predicting the
training mean), mean over the 300 test opponents:

| method (seed) | N=5 | 10 | 20 | 50 | 100 | 200 | 500 | ‖ĝ‖/‖g‖ (N=100, median) |
|---|---|---|---|---|---|---|---|---|
| G_MSE s0 / s1 / s2 | 0.785 / 0.791 / 0.782 | 0.656 / 0.666 / 0.665 | 0.555 / 0.559 / 0.562 | 0.457 / 0.467 / 0.464 | 0.415 / 0.424 / 0.422 | 0.389 / 0.398 / 0.396 | 0.372 / 0.381 / 0.380 | 0.93 / 0.93 / 0.95 |
| RECON s0 / s1 / s2 (implied A ŷ) | 0.819 / 0.817 / 0.810 | 0.714 / 0.720 / 0.709 | 0.634 / 0.637 / 0.629 | 0.563 / 0.561 / 0.555 | 0.532 / 0.530 / 0.525 | 0.516 / 0.511 / 0.507 | 0.503 / 0.497 / 0.495 | – |
| SAFE_REGRET s0 / s1 / s2 | 1.92 / 2.94 / 2.81 | 1.96 / 2.95 / 3.08 | 1.98 / 3.04 / 3.45 | 2.01 / 3.14 / 3.50 | 2.01 / 3.17 / 3.56 | 2.01 / 3.20 / 3.64 | 2.01 / 3.21 / 3.64 | 1.76 / 2.39 / 2.32 |

SAFE_REGRET's ĝ is far from g(q) — 2–4× the error of predicting the mean, inflated in norm by 1.8–2.4×, and,
most tellingly, **its error does not decrease with N** (it even rises slightly): the end-to-end objective
never rewards ĝ for being close to g, only for inducing a good safe response, and after 6 000 steps the
information in additional hands is not being written into ĝ in a way that the standardized metric
detects.  Raw ‖ĝ − g‖₂ tells the same story (2.0–2.9 vs 0.54–0.87 at N=500).

**Effective latent dimension (Table R9).**  Participation ratio of z across the test opponents:
SAFE_REGRET 2.1–2.6 (all N, three seeds), G_MSE 3.3–4.3, RECON 2.7–3.2.  All three use a tiny fraction of
the 128 nominal dimensions; the end-to-end objective yields the most compact latent by this (linear)
diagnostic, consistent with the RPS finding that response-aware objectives use fewer effective dimensions,
but the differences are small in absolute terms and the same caveats as in V1 §13 apply.

**Strategic geometry (Table R10; Figure 7 regenerated with the third method).**  ε = 0.10, 20 000 fixed
test pairs, z averaged over the 8 streams:

| representation | N | ρ(d_z, d_beh) | ρ(d_z, d_resp) | behavior-matched separation far/near [95 % CI] |
|---|---|---|---|---|
| G_MSE s0 / s1 / s2 | 100 | 0.797 / 0.785 / 0.787 | 0.563 / 0.553 / 0.541 | 1.005 / 1.007 / 0.986 (CIs ≈ ±0.014) |
| RECON s0 / s1 / s2 | 100 | 0.793 / 0.782 / 0.785 | 0.479 / 0.451 / 0.455 | 0.894 / 0.875 / 0.879 |
| SAFE_REGRET s0 / s1 / s2 | 100 | 0.622 / 0.661 / 0.627 | 0.548 / 0.596 / 0.557 | **1.279 / 1.317 / 1.277** [1.247, 1.310] … |
| SAFE_REGRET s0 / s1 / s2 | 20 / 500 | – | – | 1.259 / 1.309 / 1.256 at N=20; 1.281 / 1.320 / 1.286 at N=500 |
| true g(q) | – | – | – | 1.270 [1.250, 1.289] |
| d_beh itself (control) | – | – | – | 1.013 [1.002, 1.024] |

This is the one place where SAFE_REGRET is qualitatively different — and it reproduces the memory-2 RPS
signature that V1 failed to find.  At matched behavioral distance, the end-to-end latent places strategically
far opponents 28–32 % further apart than strategically near ones, exactly as the true decision vectors do
(27 %), while the G_MSE latent does not separate them at all (≈ 1.00) and the RECON latent slightly inverts
the order (0.88).  Correspondingly the SAFE_REGRET latent is *less* tied to behavior (ρ(d_z, d_beh) 0.62–0.66
vs 0.79) and equally or more tied to response distance (0.55–0.60 vs 0.54–0.56).  Training through the safe
solver therefore does organize the representation by "what matters for the response" — the encoder learns
strategic similarity — even though (§5) the resulting deployed responses are worse than G_MSE's.  The
representation-level claim of the project (a response-trained encoder organizes opponents by response
consequences rather than by behavior) is supported in Leduc by SAFE_REGRET and not by G_MSE.

## 7. Supplementary: deploying the regularized solution instead of the LP vertex

DEPLOY_SECTION_PLACEHOLDER

## 8. Conclusions of V2

**Answer to the critical question.**  No.  Training end-to-end on the exact oracle safe-response regret
through a differentiable regularized safe solver, with everything else frozen, produces a method whose
predicted g is far less accurate (standardized NMSE 2–3.6 vs 0.37–0.79) **and** whose deployed safe responses
are worse than G_MSE's at every observation budget and every ε (paired 95 % CIs exclude zero throughout;
regret 20–26 % higher in area under the curve; the deficit grows with N).  On this evidence, "predicting g
accurately" is not merely a proxy that happens to work: within the identical encoder, head, data and budget,
it is the better route to the information a certified safe response needs.  The experiment therefore does
*not* establish that accurate g prediction ≠ decision-sufficient learning; it establishes the opposite for
this game, architecture and budget.

**Why, as far as the diagnostics show.**  (1) The learning signal is weak and sparse: the map ĝ ↦ x*(ĝ) is
piecewise linear with support only on the active face of the safe polytope, the gradient into the encoder is
10–100× smaller than under the MSE objective, and the constant-τ control shows that without smoothing the
objective barely moves at all (0.199 → 0.210).  (2) Annealing helps (0.16 vs 0.20) but the free-scale head
undoes most of it by inflating ĝ, so the effective τ collapses to ≈ 3e−4 early regardless of the schedule;
pinning the scale (fixed-norm diagnostic) gives the same final regret, so the ceiling is not a τ-tuning
artefact.  (3) SAFE_REGRET's ĝ error does not fall with N: the objective rewards a good *average* response
and provides little pressure to encode the opponent-specific residual that more hands reveal, which is
exactly where G_MSE keeps improving.  (4) Six thousand steps at 32 samples is a small budget for a
piecewise-linear objective; the validation curves were still descending slowly, so a longer budget or a
hybrid objective (regret + a small g-MSE anchor) might close part of the gap — not tested here.

**What SAFE_REGRET does uniquely.**  Its latent space is organized by response consequences: at matched
behavioral distance, strategically far opponents are 28–32 % further apart in z than strategically near
ones, matching the true g vectors (27 %), whereas the G_MSE latent shows no such separation (≈ 1.00) and the
RECON latent inverts it (0.88).  The encoder trained through the solver learns "what matters for the
decision" as a *geometry*; it just does not turn that geometry into better ĝ or better responses under this
budget.  Effective latent dimension is also lowest (2.1–2.6 vs 3.3–4.3 for G_MSE).

**Supplementary (§7).**  DEPLOY_PLACEHOLDER

**Updated ranking at identical exact safety (ε = 0.10, N = 5 → 500):** bank posterior best at N ≤ 10;
G_MSE best among neural methods at every N and best overall for 20 ≤ N ≤ 100; per-opponent tabular EM best at
N ≥ 200; RECON and SAFE_REGRET behind, with SAFE_REGRET last from N = 10 on.

**What V2 does not establish.**  It does not rule out that a differently regularized or longer-trained
end-to-end objective could match G_MSE (only the specified sweep was run, one schedule was frozen); it does
not test deploying the regularized solution as the official strategy (only as a supplementary comparison); it
uses the same single population, single blueprint and small game as V1.

## 9. Runtime, reproduction, artifacts (V2)

Same container as V1 (4 CPU cores, no GPU).  Session 2026-09-23 08:10 → ≈ 19:00 UTC.

| stage | wall time | notes |
|---|---|---|
| QP layer tests (`tests/test_safe_qp.py`, 4 tests) | 65 s | all passing (35 tests in total now) |
| τ sweep: 4 runs × 6 000 steps, seed 0, 1 thread each | 4.8 h (τ = 0.10 run terminated at step 4 300 after divergence) | 4.0–4.8 h per run; 28.9 core-hours for all 7 SAFE_REGRET runs |
| fixed-norm diagnostic run (concurrent, 5th process) | 4.0 h | validation-only |
| validation selection: predict + 4 × 12 600 exact LPs | 10 min (4 workers) | 0 failures |
| seeds 1–2 of the frozen schedule, 2 threads each | 3.4 h | |
| test prediction (3 seeds) | 1 min | |
| test LP + audit, 3 × 67 200 | 50 min (4 workers) | 0 failures, max violation 9.4e−11 |
| analysis + figures | 52 s | |
| supplementary regularized deployment (6 methods × 50 400 QPs + audits) | DEPLOY_RUNTIME_PLACEHOLDER | |

**Reproduce** (after the V1 pipeline; from the repository root):
```bash
pip install clarabel                                   # in addition to the V1 requirements
python -m pytest leduc_decision_repr/tests/test_safe_qp.py -q
leduc_decision_repr/run_sweep_v2.sh                    # 4 sweep runs (validation only)
python -m leduc_decision_repr.train --method safe_regret --seed 0 --steps 6000 --tau_start 0.05 --g_norm_fixed 1 \
   --out leduc_decision_repr/outputs/runs_v2/safe_regret_t0.05_fixednorm_s0    # diagnostic (optional)
leduc_decision_repr/run_pipeline_v2.sh                 # selection -> seeds 1,2 -> test evaluation -> analysis
python -m leduc_decision_repr.deploy_compare_v2 --methods NEURAL_SAFE_REGRET_s0,NEURAL_SAFE_REGRET_s1,NEURAL_SAFE_REGRET_s2,NEURAL_DECISION_s0,NEURAL_DECISION_s1,NEURAL_DECISION_s2 --tau 0.01
python -m leduc_decision_repr.report_tables leduc_decision_repr/outputs/eval/test > leduc_decision_repr/outputs/eval/test/report_tables.md
```

**Artifacts.**  Solver: `game/safe_qp.py` (+ `tests/test_safe_qp.py`).  Runs: `outputs/runs_v2/*`
(`config.json`, `log.jsonl` with per-step τ, gradient norms, policy statistics; `best.pt`; `result.json`;
`sweep_diagnostics.json`).  Validation selection: `outputs/eval/val_sweep_v2/sweep_selection.json` (+ solve
arrays); fixed-norm diagnostic: `outputs/eval/val_fixednorm_v2/`.  Test: `outputs/eval/test/` now also holds
`pred_/ghat_/solve_NEURAL_SAFE_REGRET_s{0,1,2}` and the regenerated `summary.json`, `geometry.json`,
`per_opponent_metrics.csv`, `report_tables.md`, plus `deployqp_tau0.01_*.npz` for §7.  Figures:
`figures/fig1…fig11` regenerated with the third method, `figures/fig12_annealing.*` (+ `_data.json`).


## Appendix B. Full three-way tables (generated by `report_tables.py`; also covers V1 methods)

### Table R1. Safe response regret (chips/hand, mean over 300 held-out opponents, 8 streams each; 95% paired-bootstrap CI)

**ε = 0.0** (Nash regret = oracle gain = 0.005 [0.004, 0.007])

| method | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 | AUC(logN) |
|---|---|---|---|---|---|---|---|---|
| NEURAL_SAFE_REGRET | 0.011 [0.009,0.012] | 0.011 [0.009,0.012] | 0.011 [0.009,0.012] | 0.010 [0.009,0.012] | 0.010 [0.009,0.012] | 0.010 [0.009,0.012] | 0.010 [0.009,0.012] | 0.010 [0.009,0.012] |
| NEURAL_DECISION | 0.005 [0.004,0.007] | 0.005 [0.004,0.007] | 0.005 [0.004,0.006] | 0.004 [0.003,0.006] | 0.004 [0.003,0.005] | 0.004 [0.003,0.005] | 0.004 [0.003,0.005] | 0.004 [0.003,0.006] |
| NEURAL_RECON | 0.005 [0.004,0.007] | 0.005 [0.003,0.006] | 0.004 [0.003,0.005] | 0.004 [0.002,0.005] | 0.003 [0.002,0.005] | 0.003 [0.002,0.005] | 0.003 [0.002,0.004] | 0.004 [0.003,0.005] |
| BANK_POSTERIOR | 0.005 [0.004,0.007] | 0.005 [0.004,0.006] | 0.005 [0.004,0.006] | 0.005 [0.004,0.006] | 0.005 [0.004,0.006] | 0.005 [0.004,0.007] | 0.005 [0.004,0.007] | 0.005 [0.004,0.006] |
| TABULAR_EM_NASH | 0.053 [0.048,0.059] | 0.045 [0.041,0.050] | 0.037 [0.033,0.040] | 0.025 [0.023,0.028] | 0.018 [0.017,0.020] | 0.012 [0.011,0.013] | 0.007 [0.006,0.007] | 0.027 [0.025,0.029] |
| TABULAR_EM_UNIFORM | 0.005 [0.004,0.007] | 0.005 [0.004,0.007] | 0.006 [0.004,0.007] | 0.005 [0.004,0.006] | 0.004 [0.004,0.005] | 0.003 [0.003,0.004] | 0.002 [0.002,0.002] | 0.004 [0.004,0.005] |
| NEURAL_DECISION_s0 | 0.005 [0.004,0.007] | 0.005 [0.004,0.007] | 0.005 [0.003,0.006] | 0.004 [0.003,0.006] | 0.004 [0.003,0.006] | 0.004 [0.003,0.006] | 0.004 [0.003,0.006] | 0.004 [0.003,0.006] |
| NEURAL_DECISION_s1 | 0.005 [0.004,0.007] | 0.005 [0.004,0.007] | 0.005 [0.003,0.006] | 0.004 [0.003,0.006] | 0.004 [0.003,0.005] | 0.004 [0.003,0.005] | 0.004 [0.003,0.005] | 0.004 [0.003,0.006] |
| NEURAL_DECISION_s2 | 0.005 [0.004,0.007] | 0.005 [0.004,0.007] | 0.005 [0.004,0.006] | 0.004 [0.003,0.006] | 0.004 [0.003,0.006] | 0.004 [0.003,0.005] | 0.004 [0.003,0.005] | 0.005 [0.003,0.006] |
| NEURAL_RECON_s0 | 0.005 [0.004,0.007] | 0.005 [0.003,0.006] | 0.004 [0.003,0.005] | 0.003 [0.002,0.005] | 0.003 [0.002,0.005] | 0.003 [0.002,0.005] | 0.003 [0.002,0.004] | 0.004 [0.003,0.005] |
| NEURAL_RECON_s1 | 0.005 [0.004,0.007] | 0.005 [0.003,0.006] | 0.004 [0.003,0.005] | 0.004 [0.002,0.005] | 0.003 [0.002,0.005] | 0.003 [0.002,0.005] | 0.003 [0.002,0.004] | 0.004 [0.003,0.005] |
| NEURAL_RECON_s2 | 0.005 [0.004,0.007] | 0.005 [0.003,0.006] | 0.004 [0.003,0.006] | 0.004 [0.002,0.005] | 0.003 [0.002,0.005] | 0.003 [0.002,0.005] | 0.003 [0.002,0.004] | 0.004 [0.003,0.005] |
| NEURAL_SAFE_REGRET_s0 | 0.010 [0.009,0.012] | 0.010 [0.009,0.012] | 0.010 [0.009,0.012] | 0.010 [0.009,0.012] | 0.010 [0.008,0.012] | 0.010 [0.008,0.012] | 0.010 [0.009,0.012] | 0.010 [0.009,0.012] |
| NEURAL_SAFE_REGRET_s1 | 0.011 [0.009,0.013] | 0.011 [0.010,0.013] | 0.012 [0.010,0.014] | 0.012 [0.010,0.013] | 0.011 [0.010,0.013] | 0.011 [0.010,0.013] | 0.011 [0.010,0.013] | 0.011 [0.010,0.013] |
| NEURAL_SAFE_REGRET_s2 | 0.010 [0.009,0.012] | 0.010 [0.009,0.012] | 0.010 [0.009,0.012] | 0.010 [0.008,0.011] | 0.010 [0.008,0.011] | 0.009 [0.008,0.011] | 0.009 [0.008,0.011] | 0.010 [0.008,0.012] |

**ε = 0.05** (Nash regret = oracle gain = 0.347 [0.314, 0.383])

| method | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 | AUC(logN) |
|---|---|---|---|---|---|---|---|---|
| NEURAL_SAFE_REGRET | 0.134 [0.123,0.146] | 0.126 [0.115,0.137] | 0.119 [0.109,0.130] | 0.114 [0.103,0.125] | 0.111 [0.100,0.122] | 0.110 [0.099,0.121] | 0.109 [0.098,0.120] | 0.117 [0.106,0.127] |
| NEURAL_DECISION | 0.126 [0.115,0.136] | 0.112 [0.102,0.121] | 0.102 [0.092,0.111] | 0.091 [0.082,0.100] | 0.086 [0.077,0.094] | 0.084 [0.075,0.092] | 0.081 [0.073,0.090] | 0.096 [0.086,0.105] |
| NEURAL_RECON | 0.132 [0.121,0.143] | 0.117 [0.107,0.128] | 0.107 [0.097,0.117] | 0.097 [0.087,0.107] | 0.093 [0.083,0.103] | 0.091 [0.081,0.101] | 0.089 [0.079,0.100] | 0.102 [0.092,0.112] |
| BANK_POSTERIOR | 0.119 [0.109,0.130] | 0.107 [0.098,0.117] | 0.101 [0.092,0.112] | 0.096 [0.087,0.106] | 0.096 [0.087,0.106] | 0.094 [0.085,0.103] | 0.094 [0.084,0.104] | 0.100 [0.091,0.110] |
| TABULAR_EM_NASH | 0.318 [0.291,0.345] | 0.285 [0.261,0.308] | 0.251 [0.230,0.271] | 0.200 [0.184,0.216] | 0.163 [0.148,0.176] | 0.136 [0.124,0.148] | 0.110 [0.100,0.121] | 0.205 [0.188,0.221] |
| TABULAR_EM_UNIFORM | 0.147 [0.135,0.160] | 0.135 [0.125,0.145] | 0.122 [0.114,0.131] | 0.098 [0.091,0.104] | 0.079 [0.075,0.085] | 0.064 [0.060,0.068] | 0.050 [0.047,0.053] | 0.098 [0.092,0.104] |
| NEURAL_DECISION_s0 | 0.126 [0.115,0.138] | 0.111 [0.101,0.121] | 0.101 [0.092,0.111] | 0.090 [0.082,0.100] | 0.085 [0.076,0.094] | 0.083 [0.074,0.092] | 0.081 [0.072,0.090] | 0.095 [0.086,0.104] |
| NEURAL_DECISION_s1 | 0.126 [0.116,0.137] | 0.112 [0.102,0.122] | 0.102 [0.093,0.111] | 0.092 [0.083,0.101] | 0.086 [0.078,0.095] | 0.084 [0.075,0.092] | 0.082 [0.073,0.090] | 0.096 [0.087,0.105] |
| NEURAL_DECISION_s2 | 0.125 [0.115,0.136] | 0.112 [0.102,0.122] | 0.102 [0.093,0.112] | 0.091 [0.082,0.099] | 0.086 [0.077,0.095] | 0.084 [0.075,0.093] | 0.082 [0.073,0.091] | 0.096 [0.087,0.105] |
| NEURAL_RECON_s0 | 0.131 [0.120,0.143] | 0.117 [0.106,0.127] | 0.106 [0.096,0.117] | 0.097 [0.087,0.107] | 0.093 [0.084,0.104] | 0.091 [0.081,0.102] | 0.089 [0.079,0.100] | 0.102 [0.092,0.112] |
| NEURAL_RECON_s1 | 0.133 [0.122,0.145] | 0.118 [0.108,0.129] | 0.107 [0.097,0.117] | 0.098 [0.088,0.108] | 0.094 [0.084,0.104] | 0.091 [0.081,0.102] | 0.090 [0.080,0.100] | 0.103 [0.093,0.113] |
| NEURAL_RECON_s2 | 0.131 [0.120,0.141] | 0.117 [0.107,0.127] | 0.107 [0.097,0.116] | 0.097 [0.087,0.107] | 0.093 [0.084,0.104] | 0.091 [0.081,0.101] | 0.089 [0.080,0.100] | 0.102 [0.092,0.112] |
| NEURAL_SAFE_REGRET_s0 | 0.134 [0.123,0.145] | 0.125 [0.114,0.136] | 0.118 [0.107,0.129] | 0.114 [0.103,0.125] | 0.111 [0.100,0.122] | 0.110 [0.098,0.120] | 0.109 [0.098,0.120] | 0.116 [0.105,0.127] |
| NEURAL_SAFE_REGRET_s1 | 0.133 [0.122,0.145] | 0.124 [0.114,0.135] | 0.118 [0.107,0.129] | 0.112 [0.102,0.123] | 0.109 [0.098,0.119] | 0.108 [0.098,0.119] | 0.107 [0.097,0.118] | 0.115 [0.104,0.126] |
| NEURAL_SAFE_REGRET_s2 | 0.136 [0.124,0.147] | 0.128 [0.117,0.139] | 0.122 [0.111,0.132] | 0.116 [0.105,0.126] | 0.113 [0.102,0.124] | 0.112 [0.101,0.122] | 0.111 [0.100,0.122] | 0.119 [0.108,0.129] |

**ε = 0.1** (Nash regret = oracle gain = 0.491 [0.445, 0.537])

| method | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 | AUC(logN) |
|---|---|---|---|---|---|---|---|---|
| NEURAL_SAFE_REGRET | 0.177 [0.162,0.192] | 0.165 [0.151,0.179] | 0.155 [0.142,0.169] | 0.148 [0.134,0.162] | 0.144 [0.130,0.158] | 0.142 [0.128,0.156] | 0.140 [0.126,0.154] | 0.151 [0.138,0.165] |
| NEURAL_DECISION | 0.167 [0.153,0.182] | 0.149 [0.136,0.162] | 0.135 [0.123,0.147] | 0.120 [0.109,0.132] | 0.112 [0.101,0.123] | 0.108 [0.097,0.119] | 0.105 [0.094,0.117] | 0.126 [0.114,0.137] |
| NEURAL_RECON | 0.177 [0.163,0.192] | 0.157 [0.145,0.171] | 0.142 [0.130,0.156] | 0.128 [0.116,0.142] | 0.124 [0.111,0.138] | 0.120 [0.107,0.134] | 0.118 [0.105,0.132] | 0.136 [0.123,0.149] |
| BANK_POSTERIOR | 0.158 [0.146,0.172] | 0.144 [0.132,0.156] | 0.136 [0.125,0.149] | 0.128 [0.117,0.141] | 0.127 [0.116,0.139] | 0.123 [0.112,0.135] | 0.122 [0.110,0.134] | 0.132 [0.121,0.145] |
| TABULAR_EM_NASH | 0.408 [0.375,0.440] | 0.365 [0.336,0.393] | 0.321 [0.296,0.346] | 0.257 [0.237,0.277] | 0.212 [0.195,0.229] | 0.179 [0.164,0.194] | 0.147 [0.134,0.161] | 0.264 [0.244,0.284] |
| TABULAR_EM_UNIFORM | 0.196 [0.181,0.210] | 0.180 [0.168,0.192] | 0.165 [0.154,0.176] | 0.135 [0.126,0.143] | 0.114 [0.107,0.121] | 0.093 [0.087,0.100] | 0.075 [0.070,0.080] | 0.135 [0.127,0.143] |
| NEURAL_DECISION_s0 | 0.167 [0.153,0.182] | 0.148 [0.135,0.162] | 0.133 [0.122,0.146] | 0.120 [0.108,0.132] | 0.111 [0.100,0.122] | 0.107 [0.096,0.118] | 0.103 [0.093,0.114] | 0.125 [0.114,0.137] |
| NEURAL_DECISION_s1 | 0.167 [0.154,0.182] | 0.150 [0.138,0.162] | 0.135 [0.124,0.147] | 0.121 [0.111,0.132] | 0.112 [0.102,0.123] | 0.108 [0.097,0.119] | 0.105 [0.095,0.117] | 0.126 [0.116,0.137] |
| NEURAL_DECISION_s2 | 0.167 [0.153,0.181] | 0.149 [0.136,0.162] | 0.135 [0.123,0.147] | 0.119 [0.108,0.131] | 0.112 [0.101,0.123] | 0.109 [0.098,0.120] | 0.106 [0.096,0.118] | 0.126 [0.115,0.137] |
| NEURAL_RECON_s0 | 0.177 [0.162,0.192] | 0.156 [0.143,0.170] | 0.142 [0.129,0.155] | 0.128 [0.115,0.141] | 0.123 [0.111,0.137] | 0.120 [0.107,0.134] | 0.118 [0.106,0.132] | 0.135 [0.123,0.149] |
| NEURAL_RECON_s1 | 0.179 [0.164,0.193] | 0.159 [0.146,0.173] | 0.144 [0.130,0.157] | 0.130 [0.117,0.144] | 0.125 [0.112,0.139] | 0.121 [0.108,0.135] | 0.118 [0.104,0.131] | 0.137 [0.124,0.151] |
| NEURAL_RECON_s2 | 0.176 [0.162,0.190] | 0.157 [0.144,0.170] | 0.142 [0.129,0.155] | 0.127 [0.115,0.141] | 0.123 [0.111,0.138] | 0.120 [0.107,0.134] | 0.118 [0.105,0.133] | 0.135 [0.123,0.148] |
| NEURAL_SAFE_REGRET_s0 | 0.177 [0.162,0.192] | 0.164 [0.149,0.179] | 0.155 [0.141,0.170] | 0.148 [0.134,0.163] | 0.145 [0.131,0.159] | 0.142 [0.128,0.157] | 0.141 [0.127,0.155] | 0.151 [0.138,0.166] |
| NEURAL_SAFE_REGRET_s1 | 0.175 [0.161,0.191] | 0.163 [0.149,0.178] | 0.153 [0.139,0.168] | 0.146 [0.132,0.160] | 0.141 [0.128,0.155] | 0.139 [0.126,0.153] | 0.138 [0.124,0.152] | 0.149 [0.136,0.163] |
| NEURAL_SAFE_REGRET_s2 | 0.178 [0.163,0.194] | 0.167 [0.152,0.181] | 0.157 [0.143,0.172] | 0.150 [0.135,0.164] | 0.146 [0.132,0.160] | 0.144 [0.130,0.158] | 0.142 [0.128,0.156] | 0.153 [0.140,0.168] |

**ε = 0.2** (Nash regret = oracle gain = 0.684 [0.629, 0.744])

| method | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 | AUC(logN) |
|---|---|---|---|---|---|---|---|---|
| NEURAL_SAFE_REGRET | 0.261 [0.241,0.281] | 0.243 [0.224,0.262] | 0.225 [0.206,0.244] | 0.213 [0.194,0.232] | 0.206 [0.188,0.225] | 0.202 [0.184,0.220] | 0.199 [0.181,0.218] | 0.219 [0.200,0.237] |
| NEURAL_DECISION | 0.238 [0.220,0.256] | 0.210 [0.193,0.226] | 0.188 [0.172,0.204] | 0.165 [0.151,0.180] | 0.154 [0.140,0.168] | 0.147 [0.133,0.161] | 0.142 [0.128,0.157] | 0.174 [0.159,0.189] |
| NEURAL_RECON | 0.251 [0.233,0.271] | 0.224 [0.206,0.242] | 0.200 [0.182,0.218] | 0.180 [0.162,0.197] | 0.172 [0.155,0.191] | 0.165 [0.148,0.184] | 0.162 [0.144,0.181] | 0.190 [0.173,0.208] |
| BANK_POSTERIOR | 0.225 [0.208,0.243] | 0.202 [0.186,0.219] | 0.190 [0.174,0.208] | 0.176 [0.161,0.192] | 0.174 [0.159,0.192] | 0.170 [0.154,0.187] | 0.171 [0.153,0.190] | 0.184 [0.169,0.201] |
| TABULAR_EM_NASH | 0.556 [0.516,0.600] | 0.498 [0.463,0.537] | 0.436 [0.405,0.470] | 0.352 [0.325,0.379] | 0.291 [0.268,0.316] | 0.247 [0.226,0.269] | 0.208 [0.189,0.228] | 0.362 [0.336,0.390] |
| TABULAR_EM_UNIFORM | 0.276 [0.258,0.294] | 0.257 [0.240,0.273] | 0.232 [0.219,0.245] | 0.191 [0.179,0.201] | 0.166 [0.156,0.177] | 0.139 [0.130,0.148] | 0.113 [0.105,0.121] | 0.194 [0.183,0.204] |
| NEURAL_DECISION_s0 | 0.237 [0.220,0.255] | 0.208 [0.192,0.225] | 0.187 [0.172,0.203] | 0.164 [0.150,0.180] | 0.153 [0.139,0.167] | 0.147 [0.133,0.162] | 0.141 [0.127,0.156] | 0.173 [0.159,0.189] |
| NEURAL_DECISION_s1 | 0.240 [0.222,0.259] | 0.211 [0.194,0.228] | 0.188 [0.173,0.205] | 0.166 [0.151,0.181] | 0.154 [0.139,0.169] | 0.146 [0.131,0.161] | 0.141 [0.126,0.156] | 0.174 [0.159,0.190] |
| NEURAL_DECISION_s2 | 0.238 [0.220,0.258] | 0.210 [0.193,0.228] | 0.189 [0.173,0.206] | 0.166 [0.151,0.183] | 0.155 [0.140,0.171] | 0.149 [0.134,0.165] | 0.145 [0.130,0.161] | 0.175 [0.160,0.192] |
| NEURAL_RECON_s0 | 0.251 [0.234,0.271] | 0.223 [0.206,0.242] | 0.200 [0.184,0.218] | 0.179 [0.163,0.198] | 0.171 [0.155,0.191] | 0.166 [0.149,0.185] | 0.163 [0.146,0.182] | 0.190 [0.174,0.208] |
| NEURAL_RECON_s1 | 0.253 [0.234,0.272] | 0.224 [0.206,0.242] | 0.200 [0.182,0.218] | 0.182 [0.164,0.200] | 0.173 [0.156,0.192] | 0.165 [0.148,0.183] | 0.160 [0.143,0.179] | 0.190 [0.173,0.208] |
| NEURAL_RECON_s2 | 0.250 [0.231,0.268] | 0.224 [0.206,0.242] | 0.200 [0.182,0.217] | 0.178 [0.161,0.195] | 0.172 [0.155,0.190] | 0.166 [0.148,0.183] | 0.163 [0.145,0.180] | 0.190 [0.173,0.206] |
| NEURAL_SAFE_REGRET_s0 | 0.258 [0.239,0.278] | 0.241 [0.222,0.261] | 0.224 [0.204,0.243] | 0.213 [0.194,0.233] | 0.206 [0.187,0.226] | 0.201 [0.182,0.221] | 0.199 [0.179,0.219] | 0.218 [0.200,0.238] |
| NEURAL_SAFE_REGRET_s1 | 0.260 [0.241,0.280] | 0.240 [0.222,0.259] | 0.221 [0.204,0.239] | 0.209 [0.192,0.227] | 0.202 [0.185,0.220] | 0.199 [0.182,0.218] | 0.197 [0.180,0.215] | 0.216 [0.199,0.233] |
| NEURAL_SAFE_REGRET_s2 | 0.265 [0.246,0.285] | 0.247 [0.228,0.267] | 0.231 [0.213,0.250] | 0.216 [0.198,0.235] | 0.211 [0.193,0.230] | 0.206 [0.188,0.224] | 0.202 [0.185,0.221] | 0.223 [0.205,0.242] |

### Table R2. Fraction of oracle-safe gain recovered F (opponents with gain ≥ 0.02 only)

**ε = 0.05** (291 of 300 opponents with valid denominator)

| method | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 | N50 | N80 | N90 | AUC_F |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NEURAL_SAFE_REGRET | 0.48 [0.45,0.52] | 0.50 [0.47,0.54] | 0.52 [0.48,0.56] | 0.54 [0.50,0.57] | 0.54 [0.51,0.58] | 0.55 [0.51,0.58] | 0.55 [0.51,0.59] | 10 | >500 | >500 | 0.528 [0.492,0.564] |
| NEURAL_DECISION | 0.54 [0.51,0.57] | 0.59 [0.56,0.61] | 0.62 [0.60,0.65] | 0.66 [0.63,0.68] | 0.68 [0.65,0.70] | 0.69 [0.66,0.71] | 0.70 [0.68,0.72] | 5 | >500 | >500 | 0.644 [0.622,0.666] |
| NEURAL_RECON | 0.53 [0.50,0.56] | 0.58 [0.56,0.61] | 0.62 [0.60,0.64] | 0.66 [0.63,0.68] | 0.67 [0.65,0.70] | 0.69 [0.67,0.71] | 0.70 [0.68,0.72] | 5 | >500 | >500 | 0.642 [0.619,0.664] |
| BANK_POSTERIOR | 0.56 [0.54,0.59] | 0.61 [0.59,0.64] | 0.64 [0.62,0.66] | 0.67 [0.65,0.69] | 0.67 [0.65,0.69] | 0.68 [0.66,0.70] | 0.69 [0.67,0.71] | 5 | >500 | >500 | 0.653 [0.633,0.673] |
| TABULAR_EM_NASH | 0.02 [-0.01,0.04] | 0.09 [0.06,0.12] | 0.17 [0.13,0.20] | 0.29 [0.26,0.32] | 0.39 [0.36,0.42] | 0.47 [0.44,0.50] | 0.56 [0.53,0.59] | 500 | >500 | >500 | 0.291 [0.262,0.319] |
| TABULAR_EM_UNIFORM | 0.44 [0.40,0.47] | 0.46 [0.42,0.50] | 0.49 [0.45,0.53] | 0.55 [0.51,0.59] | 0.60 [0.57,0.64] | 0.65 [0.61,0.69] | 0.69 [0.65,0.73] | 50 | >500 | >500 | 0.558 [0.521,0.593] |
| NEURAL_DECISION_s0 | 0.54 [0.51,0.56] | 0.59 [0.56,0.61] | 0.62 [0.60,0.65] | 0.66 [0.64,0.68] | 0.68 [0.66,0.70] | 0.69 [0.67,0.71] | 0.71 [0.68,0.73] | 5 | >500 | >500 | 0.646 [0.624,0.669] |
| NEURAL_DECISION_s1 | 0.54 [0.51,0.57] | 0.58 [0.56,0.61] | 0.62 [0.59,0.64] | 0.65 [0.63,0.68] | 0.67 [0.65,0.69] | 0.68 [0.66,0.71] | 0.70 [0.67,0.72] | 5 | >500 | >500 | 0.640 [0.616,0.664] |
| NEURAL_DECISION_s2 | 0.55 [0.52,0.58] | 0.59 [0.57,0.62] | 0.62 [0.60,0.65] | 0.66 [0.64,0.69] | 0.68 [0.65,0.70] | 0.69 [0.66,0.71] | 0.69 [0.67,0.72] | 5 | >500 | >500 | 0.646 [0.623,0.670] |
| NEURAL_RECON_s0 | 0.53 [0.50,0.55] | 0.58 [0.56,0.61] | 0.62 [0.60,0.64] | 0.66 [0.63,0.68] | 0.67 [0.65,0.69] | 0.68 [0.66,0.71] | 0.70 [0.67,0.72] | 5 | >500 | >500 | 0.641 [0.619,0.662] |
| NEURAL_RECON_s1 | 0.52 [0.49,0.55] | 0.58 [0.55,0.60] | 0.61 [0.59,0.64] | 0.65 [0.63,0.68] | 0.67 [0.65,0.70] | 0.69 [0.66,0.71] | 0.70 [0.68,0.72] | 5 | >500 | >500 | 0.638 [0.615,0.660] |
| NEURAL_RECON_s2 | 0.54 [0.51,0.56] | 0.59 [0.56,0.61] | 0.62 [0.60,0.65] | 0.66 [0.64,0.68] | 0.68 [0.66,0.70] | 0.69 [0.67,0.71] | 0.70 [0.68,0.72] | 5 | >500 | >500 | 0.646 [0.624,0.667] |
| NEURAL_SAFE_REGRET_s0 | 0.48 [0.45,0.52] | 0.50 [0.47,0.54] | 0.52 [0.49,0.56] | 0.54 [0.50,0.57] | 0.55 [0.51,0.58] | 0.55 [0.51,0.59] | 0.55 [0.52,0.59] | 10 | >500 | >500 | 0.531 [0.495,0.567] |
| NEURAL_SAFE_REGRET_s1 | 0.49 [0.45,0.52] | 0.51 [0.48,0.54] | 0.53 [0.49,0.56] | 0.54 [0.51,0.58] | 0.55 [0.52,0.59] | 0.55 [0.52,0.59] | 0.55 [0.51,0.59] | 10 | >500 | >500 | 0.534 [0.500,0.569] |
| NEURAL_SAFE_REGRET_s2 | 0.48 [0.44,0.51] | 0.50 [0.46,0.53] | 0.51 [0.48,0.55] | 0.53 [0.49,0.56] | 0.53 [0.50,0.57] | 0.54 [0.50,0.57] | 0.54 [0.50,0.58] | 20 | >500 | >500 | 0.520 [0.484,0.554] |

**ε = 0.1** (299 of 300 opponents with valid denominator)

| method | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 | N50 | N80 | N90 | AUC_F |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NEURAL_SAFE_REGRET | 0.49 [0.45,0.53] | 0.51 [0.47,0.55] | 0.53 [0.49,0.57] | 0.54 [0.50,0.58] | 0.55 [0.51,0.59] | 0.55 [0.51,0.59] | 0.55 [0.51,0.60] | 10 | >500 | >500 | 0.535 [0.494,0.576] |
| NEURAL_DECISION | 0.54 [0.51,0.57] | 0.59 [0.56,0.62] | 0.63 [0.60,0.65] | 0.66 [0.64,0.69] | 0.68 [0.66,0.71] | 0.70 [0.67,0.72] | 0.71 [0.68,0.73] | 5 | >500 | >500 | 0.650 [0.624,0.675] |
| NEURAL_RECON | 0.51 [0.47,0.55] | 0.58 [0.55,0.61] | 0.62 [0.59,0.65] | 0.66 [0.64,0.69] | 0.68 [0.66,0.70] | 0.70 [0.68,0.72] | 0.71 [0.69,0.73] | 5 | >500 | >500 | 0.644 [0.620,0.669] |
| BANK_POSTERIOR | 0.56 [0.53,0.59] | 0.61 [0.58,0.64] | 0.64 [0.62,0.67] | 0.67 [0.65,0.70] | 0.68 [0.66,0.70] | 0.70 [0.68,0.71] | 0.71 [0.69,0.72] | 5 | >500 | >500 | 0.659 [0.639,0.679] |
| TABULAR_EM_NASH | 0.09 [0.06,0.11] | 0.16 [0.13,0.18] | 0.22 [0.19,0.25] | 0.33 [0.30,0.37] | 0.42 [0.39,0.46] | 0.49 [0.46,0.52] | 0.57 [0.53,0.60] | 500 | >500 | >500 | 0.333 [0.304,0.361] |
| TABULAR_EM_UNIFORM | 0.40 [0.35,0.45] | 0.42 [0.37,0.47] | 0.44 [0.39,0.49] | 0.50 [0.45,0.55] | 0.55 [0.50,0.60] | 0.59 [0.55,0.64] | 0.65 [0.61,0.69] | 100 | >500 | >500 | 0.508 [0.459,0.556] |
| NEURAL_DECISION_s0 | 0.54 [0.50,0.57] | 0.59 [0.56,0.62] | 0.63 [0.60,0.66] | 0.66 [0.64,0.69] | 0.69 [0.66,0.71] | 0.70 [0.68,0.72] | 0.71 [0.69,0.74] | 5 | >500 | >500 | 0.652 [0.626,0.677] |
| NEURAL_DECISION_s1 | 0.54 [0.51,0.58] | 0.58 [0.55,0.61] | 0.62 [0.60,0.65] | 0.66 [0.63,0.68] | 0.68 [0.66,0.71] | 0.69 [0.67,0.72] | 0.71 [0.68,0.73] | 5 | >500 | >500 | 0.647 [0.620,0.673] |
| NEURAL_DECISION_s2 | 0.55 [0.51,0.58] | 0.60 [0.57,0.62] | 0.63 [0.60,0.66] | 0.67 [0.64,0.69] | 0.68 [0.66,0.71] | 0.69 [0.67,0.72] | 0.70 [0.68,0.73] | 5 | >500 | >500 | 0.651 [0.625,0.676] |
| NEURAL_RECON_s0 | 0.51 [0.47,0.54] | 0.58 [0.55,0.61] | 0.62 [0.59,0.65] | 0.66 [0.64,0.69] | 0.68 [0.65,0.70] | 0.69 [0.67,0.72] | 0.71 [0.69,0.73] | 5 | >500 | >500 | 0.643 [0.618,0.668] |
| NEURAL_RECON_s1 | 0.50 [0.47,0.54] | 0.57 [0.54,0.60] | 0.61 [0.58,0.64] | 0.66 [0.63,0.68] | 0.68 [0.66,0.70] | 0.70 [0.67,0.72] | 0.71 [0.69,0.73] | 5 | >500 | >500 | 0.640 [0.615,0.664] |
| NEURAL_RECON_s2 | 0.52 [0.49,0.56] | 0.58 [0.55,0.61] | 0.63 [0.60,0.65] | 0.67 [0.64,0.69] | 0.68 [0.66,0.71] | 0.70 [0.68,0.72] | 0.71 [0.69,0.73] | 5 | >500 | >500 | 0.649 [0.625,0.672] |
| NEURAL_SAFE_REGRET_s0 | 0.49 [0.44,0.53] | 0.51 [0.47,0.55] | 0.53 [0.49,0.57] | 0.55 [0.50,0.59] | 0.55 [0.51,0.60] | 0.56 [0.52,0.61] | 0.57 [0.53,0.61] | 10 | >500 | >500 | 0.540 [0.497,0.582] |
| NEURAL_SAFE_REGRET_s1 | 0.49 [0.45,0.53] | 0.51 [0.47,0.55] | 0.53 [0.49,0.57] | 0.54 [0.50,0.58] | 0.55 [0.51,0.59] | 0.55 [0.50,0.59] | 0.54 [0.50,0.59] | 10 | >500 | >500 | 0.534 [0.492,0.574] |
| NEURAL_SAFE_REGRET_s2 | 0.48 [0.44,0.52] | 0.51 [0.47,0.55] | 0.52 [0.48,0.56] | 0.54 [0.49,0.58] | 0.54 [0.50,0.59] | 0.55 [0.50,0.59] | 0.55 [0.50,0.59] | 10 | >500 | >500 | 0.531 [0.486,0.572] |

**ε = 0.2** (300 of 300 opponents with valid denominator)

| method | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 | N50 | N80 | N90 | AUC_F |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NEURAL_SAFE_REGRET | 0.43 [0.37,0.48] | 0.45 [0.40,0.50] | 0.48 [0.43,0.53] | 0.50 [0.45,0.55] | 0.51 [0.46,0.56] | 0.52 [0.47,0.57] | 0.52 [0.47,0.57] | 50 | >500 | >500 | 0.493 [0.442,0.542] |
| NEURAL_DECISION | 0.51 [0.48,0.55] | 0.58 [0.55,0.61] | 0.63 [0.60,0.65] | 0.67 [0.64,0.69] | 0.69 [0.66,0.71] | 0.70 [0.67,0.72] | 0.71 [0.69,0.74] | 5 | >500 | >500 | 0.649 [0.622,0.674] |
| NEURAL_RECON | 0.50 [0.46,0.53] | 0.57 [0.54,0.60] | 0.62 [0.59,0.65] | 0.66 [0.64,0.69] | 0.68 [0.66,0.70] | 0.70 [0.68,0.72] | 0.71 [0.69,0.74] | 10 | >500 | >500 | 0.644 [0.619,0.666] |
| BANK_POSTERIOR | 0.54 [0.50,0.58] | 0.61 [0.58,0.64] | 0.64 [0.62,0.67] | 0.68 [0.65,0.70] | 0.68 [0.66,0.70] | 0.70 [0.68,0.72] | 0.70 [0.68,0.72] | 5 | >500 | >500 | 0.658 [0.635,0.679] |
| TABULAR_EM_NASH | 0.11 [0.09,0.14] | 0.18 [0.15,0.20] | 0.25 [0.22,0.27] | 0.35 [0.32,0.38] | 0.44 [0.40,0.47] | 0.50 [0.47,0.53] | 0.57 [0.54,0.60] | 500 | >500 | >500 | 0.349 [0.321,0.376] |
| TABULAR_EM_UNIFORM | 0.33 [0.26,0.40] | 0.35 [0.27,0.41] | 0.38 [0.31,0.45] | 0.46 [0.40,0.52] | 0.52 [0.46,0.57] | 0.57 [0.52,0.62] | 0.62 [0.57,0.67] | 100 | >500 | >500 | 0.465 [0.404,0.523] |
| NEURAL_DECISION_s0 | 0.51 [0.47,0.54] | 0.58 [0.55,0.61] | 0.63 [0.60,0.65] | 0.67 [0.64,0.69] | 0.69 [0.66,0.71] | 0.70 [0.68,0.73] | 0.72 [0.69,0.74] | 5 | >500 | >500 | 0.650 [0.623,0.675] |
| NEURAL_DECISION_s1 | 0.51 [0.47,0.55] | 0.57 [0.54,0.60] | 0.62 [0.59,0.65] | 0.67 [0.64,0.69] | 0.69 [0.66,0.71] | 0.70 [0.68,0.73] | 0.72 [0.69,0.74] | 5 | >500 | >500 | 0.648 [0.622,0.673] |
| NEURAL_DECISION_s2 | 0.53 [0.49,0.56] | 0.59 [0.56,0.62] | 0.63 [0.60,0.65] | 0.67 [0.64,0.69] | 0.68 [0.66,0.71] | 0.69 [0.67,0.72] | 0.70 [0.68,0.73] | 5 | >500 | >500 | 0.648 [0.621,0.674] |
| NEURAL_RECON_s0 | 0.49 [0.45,0.53] | 0.57 [0.54,0.60] | 0.62 [0.59,0.65] | 0.66 [0.64,0.69] | 0.68 [0.66,0.70] | 0.70 [0.67,0.72] | 0.71 [0.69,0.73] | 10 | >500 | >500 | 0.642 [0.617,0.666] |
| NEURAL_RECON_s1 | 0.49 [0.45,0.53] | 0.57 [0.53,0.60] | 0.62 [0.59,0.65] | 0.66 [0.63,0.68] | 0.68 [0.66,0.70] | 0.70 [0.68,0.73] | 0.72 [0.69,0.74] | 10 | >500 | >500 | 0.642 [0.616,0.666] |
| NEURAL_RECON_s2 | 0.51 [0.47,0.55] | 0.57 [0.54,0.60] | 0.62 [0.59,0.65] | 0.67 [0.64,0.69] | 0.68 [0.66,0.71] | 0.70 [0.68,0.72] | 0.71 [0.69,0.74] | 5 | >500 | >500 | 0.646 [0.622,0.671] |
| NEURAL_SAFE_REGRET_s0 | 0.43 [0.38,0.48] | 0.45 [0.40,0.50] | 0.48 [0.43,0.53] | 0.50 [0.45,0.55] | 0.52 [0.47,0.57] | 0.52 [0.47,0.57] | 0.53 [0.48,0.58] | 50 | >500 | >500 | 0.495 [0.444,0.543] |
| NEURAL_SAFE_REGRET_s1 | 0.43 [0.38,0.48] | 0.46 [0.41,0.50] | 0.48 [0.43,0.53] | 0.50 [0.45,0.55] | 0.51 [0.46,0.56] | 0.52 [0.46,0.57] | 0.52 [0.46,0.57] | 50 | >500 | >500 | 0.492 [0.440,0.542] |
| NEURAL_SAFE_REGRET_s2 | 0.42 [0.37,0.47] | 0.45 [0.40,0.50] | 0.48 [0.43,0.52] | 0.50 [0.45,0.55] | 0.51 [0.46,0.56] | 0.52 [0.47,0.57] | 0.53 [0.48,0.58] | 50 | >500 | >500 | 0.491 [0.444,0.540] |

### Table R3. Bootstrap distribution of N-thresholds (median [2.5%, 97.5%] over 2000 opponent resamples; 1000 = not reached by 500)

| ε | method | N50 | N80 | N90 |
|---|---|---|---|---|
| 0.05 | NEURAL_SAFE_REGRET | 10 [5, 100] | 1000 [1000, 1000] | 1000 [1000, 1000] |
| 0.05 | NEURAL_DECISION | 5 [5, 5] | 1000 [1000, 1000] | 1000 [1000, 1000] |
| 0.05 | NEURAL_RECON | 5 [5, 10] | 1000 [1000, 1000] | 1000 [1000, 1000] |
| 0.05 | BANK_POSTERIOR | 5 [5, 5] | 1000 [1000, 1000] | 1000 [1000, 1000] |
| 0.05 | TABULAR_EM_NASH | 500 [200, 500] | 1000 [1000, 1000] | 1000 [1000, 1000] |
| 0.05 | TABULAR_EM_UNIFORM | 50 [20, 50] | 1000 [1000, 1000] | 1000 [1000, 1000] |
| 0.1 | NEURAL_SAFE_REGRET | 10 [5, 50] | 1000 [1000, 1000] | 1000 [1000, 1000] |
| 0.1 | NEURAL_DECISION | 5 [5, 5] | 1000 [1000, 1000] | 1000 [1000, 1000] |
| 0.1 | NEURAL_RECON | 5 [5, 10] | 1000 [1000, 1000] | 1000 [1000, 1000] |
| 0.1 | BANK_POSTERIOR | 5 [5, 5] | 1000 [1000, 1000] | 1000 [1000, 1000] |
| 0.1 | TABULAR_EM_NASH | 500 [200, 500] | 1000 [1000, 1000] | 1000 [1000, 1000] |
| 0.1 | TABULAR_EM_UNIFORM | 100 [50, 200] | 1000 [1000, 1000] | 1000 [1000, 1000] |
| 0.2 | NEURAL_SAFE_REGRET | 50 [10, 1000] | 1000 [1000, 1000] | 1000 [1000, 1000] |
| 0.2 | NEURAL_DECISION | 5 [5, 10] | 1000 [1000, 1000] | 1000 [1000, 1000] |
| 0.2 | NEURAL_RECON | 10 [5, 10] | 1000 [1000, 1000] | 1000 [1000, 1000] |
| 0.2 | BANK_POSTERIOR | 5 [5, 5] | 1000 [1000, 1000] | 1000 [1000, 1000] |
| 0.2 | TABULAR_EM_NASH | 500 [200, 500] | 1000 [1000, 1000] | 1000 [1000, 1000] |
| 0.2 | TABULAR_EM_UNIFORM | 100 [50, 200] | 1000 [1000, 1000] | 1000 [1000, 1000] |

### Table R4. Paired differences (decision − reconstruction), mean over opponents with 95% CI

**NEURAL_DECISION − NEURAL_RECON**

| ε | quantity | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 |
|---|---|---|---|---|---|---|---|---|
| 0.0 | regret diff | +0.000 [-0.000,+0.000] | +0.000 [+0.000,+0.001] | +0.001 [+0.001,+0.001] | +0.001 [+0.000,+0.001] | +0.001 [+0.000,+0.001] | +0.001 [+0.000,+0.001] | +0.001 [+0.001,+0.001] |
| 0.05 | regret diff | -0.006 [-0.009,-0.003] | -0.005 [-0.009,-0.003] | -0.005 [-0.008,-0.002] | -0.006 [-0.010,-0.002] | -0.008 [-0.012,-0.004] | -0.007 [-0.012,-0.003] | -0.008 [-0.013,-0.004] |
| 0.05 | F diff | +0.02 [+0.00,+0.03] | +0.01 [-0.00,+0.02] | +0.00 [-0.01,+0.01] | -0.00 [-0.01,+0.01] | +0.00 [-0.01,+0.01] | -0.00 [-0.01,+0.01] | +0.00 [-0.01,+0.01] |
| 0.1 | regret diff | -0.010 [-0.015,-0.005] | -0.009 [-0.013,-0.005] | -0.008 [-0.012,-0.004] | -0.008 [-0.014,-0.003] | -0.012 [-0.018,-0.007] | -0.012 [-0.019,-0.006] | -0.013 [-0.020,-0.006] |
| 0.1 | F diff | +0.03 [+0.02,+0.04] | +0.01 [+0.00,+0.02] | +0.01 [-0.00,+0.01] | +0.00 [-0.01,+0.01] | +0.00 [-0.01,+0.02] | -0.00 [-0.02,+0.01] | -0.00 [-0.02,+0.01] |
| 0.2 | regret diff | -0.013 [-0.019,-0.006] | -0.014 [-0.019,-0.009] | -0.012 [-0.018,-0.006] | -0.014 [-0.022,-0.007] | -0.019 [-0.028,-0.010] | -0.018 [-0.029,-0.009] | -0.020 [-0.031,-0.010] |
| 0.2 | F diff | +0.02 [+0.01,+0.03] | +0.01 [+0.00,+0.02] | +0.01 [-0.00,+0.01] | +0.01 [-0.00,+0.02] | +0.01 [-0.01,+0.02] | -0.00 [-0.01,+0.01] | -0.00 [-0.02,+0.02] |
**NEURAL_SAFE_REGRET − NEURAL_DECISION**

| ε | quantity | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 |
|---|---|---|---|---|---|---|---|---|
| 0.0 | regret diff | +0.005 [+0.005,+0.006] | +0.006 [+0.005,+0.006] | +0.006 [+0.005,+0.007] | +0.006 [+0.005,+0.007] | +0.006 [+0.005,+0.007] | +0.006 [+0.005,+0.007] | +0.006 [+0.005,+0.007] |
| 0.05 | regret diff | +0.008 [+0.006,+0.011] | +0.014 [+0.011,+0.018] | +0.017 [+0.013,+0.022] | +0.023 [+0.018,+0.028] | +0.025 [+0.020,+0.031] | +0.026 [+0.021,+0.032] | +0.028 [+0.022,+0.033] |
| 0.05 | F diff | -0.06 [-0.07,-0.05] | -0.09 [-0.10,-0.07] | -0.10 [-0.12,-0.08] | -0.12 [-0.14,-0.10] | -0.13 [-0.16,-0.11] | -0.14 [-0.17,-0.12] | -0.15 [-0.18,-0.12] |
| 0.1 | regret diff | +0.010 [+0.006,+0.013] | +0.016 [+0.012,+0.020] | +0.021 [+0.015,+0.027] | +0.028 [+0.021,+0.035] | +0.032 [+0.025,+0.040] | +0.034 [+0.026,+0.043] | +0.035 [+0.027,+0.044] |
| 0.1 | F diff | -0.05 [-0.07,-0.04] | -0.08 [-0.09,-0.06] | -0.10 [-0.12,-0.08] | -0.12 [-0.14,-0.10] | -0.13 [-0.16,-0.11] | -0.14 [-0.17,-0.12] | -0.15 [-0.18,-0.13] |
| 0.2 | regret diff | +0.022 [+0.015,+0.030] | +0.033 [+0.026,+0.041] | +0.037 [+0.028,+0.046] | +0.047 [+0.038,+0.059] | +0.053 [+0.042,+0.064] | +0.055 [+0.044,+0.067] | +0.057 [+0.045,+0.070] |
| 0.2 | F diff | -0.09 [-0.11,-0.07] | -0.13 [-0.15,-0.10] | -0.15 [-0.17,-0.12] | -0.17 [-0.20,-0.13] | -0.17 [-0.21,-0.14] | -0.18 [-0.21,-0.15] | -0.19 [-0.23,-0.15] |
**NEURAL_SAFE_REGRET − NEURAL_RECON**

| ε | quantity | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 |
|---|---|---|---|---|---|---|---|---|
| 0.0 | regret diff | +0.005 [+0.005,+0.006] | +0.006 [+0.005,+0.007] | +0.007 [+0.006,+0.008] | +0.007 [+0.006,+0.008] | +0.007 [+0.006,+0.008] | +0.007 [+0.006,+0.008] | +0.007 [+0.006,+0.008] |
| 0.05 | regret diff | +0.003 [-0.001,+0.007] | +0.009 [+0.005,+0.012] | +0.012 [+0.009,+0.016] | +0.017 [+0.012,+0.021] | +0.018 [+0.013,+0.022] | +0.019 [+0.014,+0.024] | +0.020 [+0.015,+0.025] |
| 0.05 | F diff | -0.04 [-0.06,-0.03] | -0.08 [-0.10,-0.06] | -0.10 [-0.12,-0.08] | -0.12 [-0.14,-0.10] | -0.13 [-0.16,-0.11] | -0.14 [-0.17,-0.11] | -0.15 [-0.18,-0.12] |
| 0.1 | regret diff | -0.000 [-0.007,+0.006] | +0.007 [+0.002,+0.013] | +0.013 [+0.007,+0.019] | +0.019 [+0.013,+0.027] | +0.020 [+0.013,+0.028] | +0.021 [+0.014,+0.030] | +0.022 [+0.014,+0.031] |
| 0.1 | F diff | -0.02 [-0.04,-0.01] | -0.06 [-0.09,-0.05] | -0.09 [-0.12,-0.07] | -0.12 [-0.15,-0.09] | -0.13 [-0.16,-0.10] | -0.15 [-0.18,-0.11] | -0.16 [-0.19,-0.12] |
| 0.2 | regret diff | +0.009 [-0.001,+0.019] | +0.019 [+0.010,+0.028] | +0.025 [+0.016,+0.034] | +0.033 [+0.023,+0.044] | +0.034 [+0.023,+0.045] | +0.037 [+0.025,+0.048] | +0.037 [+0.025,+0.049] |
| 0.2 | F diff | -0.07 [-0.09,-0.05] | -0.12 [-0.15,-0.09] | -0.14 [-0.17,-0.11] | -0.16 [-0.20,-0.13] | -0.17 [-0.21,-0.13] | -0.18 [-0.22,-0.14] | -0.19 [-0.23,-0.15] |
**NEURAL_SAFE_REGRET − BANK_POSTERIOR**

| ε | quantity | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 |
|---|---|---|---|---|---|---|---|---|
| 0.0 | regret diff | +0.005 [+0.005,+0.006] | +0.006 [+0.005,+0.007] | +0.006 [+0.005,+0.007] | +0.006 [+0.004,+0.007] | +0.005 [+0.004,+0.007] | +0.005 [+0.004,+0.006] | +0.005 [+0.004,+0.006] |
| 0.05 | regret diff | +0.015 [+0.012,+0.018] | +0.019 [+0.015,+0.023] | +0.018 [+0.013,+0.022] | +0.018 [+0.013,+0.023] | +0.015 [+0.009,+0.021] | +0.016 [+0.009,+0.022] | +0.015 [+0.008,+0.022] |
| 0.05 | F diff | -0.08 [-0.10,-0.07] | -0.11 [-0.13,-0.09] | -0.12 [-0.14,-0.10] | -0.13 [-0.16,-0.11] | -0.13 [-0.16,-0.10] | -0.14 [-0.17,-0.11] | -0.14 [-0.18,-0.11] |
| 0.1 | regret diff | +0.018 [+0.014,+0.023] | +0.021 [+0.015,+0.027] | +0.019 [+0.012,+0.027] | +0.020 [+0.012,+0.027] | +0.017 [+0.009,+0.025] | +0.018 [+0.010,+0.027] | +0.018 [+0.009,+0.028] |
| 0.1 | F diff | -0.07 [-0.09,-0.06] | -0.10 [-0.12,-0.08] | -0.12 [-0.14,-0.09] | -0.13 [-0.16,-0.10] | -0.13 [-0.17,-0.10] | -0.14 [-0.18,-0.11] | -0.15 [-0.19,-0.12] |
| 0.2 | regret diff | +0.036 [+0.028,+0.044] | +0.041 [+0.032,+0.050] | +0.035 [+0.025,+0.044] | +0.037 [+0.028,+0.048] | +0.032 [+0.022,+0.043] | +0.033 [+0.022,+0.043] | +0.028 [+0.016,+0.041] |
| 0.2 | F diff | -0.11 [-0.14,-0.09] | -0.15 [-0.19,-0.12] | -0.16 [-0.20,-0.13] | -0.17 [-0.21,-0.14] | -0.17 [-0.21,-0.13] | -0.18 [-0.22,-0.14] | -0.18 [-0.23,-0.14] |

### Table R5. Per-seed neural results (regret at ε=0.1)

| run | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 | best step |
|---|---|---|---|---|---|---|---|---|
| NEURAL_DECISION_s0 | 0.167 | 0.148 | 0.133 | 0.120 | 0.111 | 0.107 | 0.103 | |
| NEURAL_DECISION_s1 | 0.167 | 0.150 | 0.135 | 0.121 | 0.112 | 0.108 | 0.105 | |
| NEURAL_DECISION_s2 | 0.167 | 0.149 | 0.135 | 0.119 | 0.112 | 0.109 | 0.106 | |
| NEURAL_RECON_s0 | 0.177 | 0.156 | 0.142 | 0.128 | 0.123 | 0.120 | 0.118 | |
| NEURAL_RECON_s1 | 0.179 | 0.159 | 0.144 | 0.130 | 0.125 | 0.121 | 0.118 | |
| NEURAL_RECON_s2 | 0.176 | 0.157 | 0.142 | 0.127 | 0.123 | 0.120 | 0.118 | |
| NEURAL_SAFE_REGRET_s0 | 0.177 | 0.164 | 0.155 | 0.148 | 0.145 | 0.142 | 0.141 | |
| NEURAL_SAFE_REGRET_s1 | 0.175 | 0.163 | 0.153 | 0.146 | 0.141 | 0.139 | 0.138 | |
| NEURAL_SAFE_REGRET_s2 | 0.178 | 0.167 | 0.157 | 0.150 | 0.146 | 0.144 | 0.142 | |

### Table R6. Per-family regret at ε=0.1 (mean) and N80

**NASH_LOGIT_PERTURB** (oracle gain at ε=0.1: 0.194)

| method | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 | N50 | N80 | N90 |
|---|---|---|---|---|---|---|---|---|---|---|
| NEURAL_SAFE_REGRET | 0.125 | 0.125 | 0.123 | 0.120 | 0.119 | 0.119 | 0.118 | >500 | >500 | >500 |
| NEURAL_DECISION | 0.113 | 0.102 | 0.094 | 0.084 | 0.078 | 0.075 | 0.072 | 500 | >500 | >500 |
| NEURAL_RECON | 0.125 | 0.109 | 0.099 | 0.087 | 0.084 | 0.078 | 0.074 | 200 | >500 | >500 |
| BANK_POSTERIOR | 0.108 | 0.095 | 0.089 | 0.079 | 0.083 | 0.076 | 0.075 | 50 | >500 | >500 |
| TABULAR_EM_NASH | 0.141 | 0.133 | 0.120 | 0.108 | 0.096 | 0.086 | 0.071 | >500 | >500 | >500 |
| TABULAR_EM_UNIFORM | 0.142 | 0.136 | 0.132 | 0.119 | 0.111 | 0.102 | 0.090 | >500 | >500 | >500 |

**NASH_RANDOM_MIX** (oracle gain at ε=0.1: 0.286)

| method | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 | N50 | N80 | N90 |
|---|---|---|---|---|---|---|---|---|---|---|
| NEURAL_SAFE_REGRET | 0.099 | 0.092 | 0.089 | 0.084 | 0.082 | 0.080 | 0.078 | 5 | >500 | >500 |
| NEURAL_DECISION | 0.092 | 0.089 | 0.086 | 0.079 | 0.074 | 0.071 | 0.069 | 5 | >500 | >500 |
| NEURAL_RECON | 0.087 | 0.084 | 0.081 | 0.076 | 0.074 | 0.071 | 0.070 | 5 | >500 | >500 |
| BANK_POSTERIOR | 0.091 | 0.086 | 0.083 | 0.076 | 0.080 | 0.081 | 0.080 | 5 | >500 | >500 |
| TABULAR_EM_NASH | 0.252 | 0.232 | 0.215 | 0.184 | 0.163 | 0.141 | 0.114 | 500 | >500 | >500 |
| TABULAR_EM_UNIFORM | 0.098 | 0.096 | 0.094 | 0.087 | 0.081 | 0.074 | 0.064 | 5 | >500 | >500 |

**STRUCTURED_CORRELATED** (oracle gain at ε=0.1: 0.799)

| method | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 | N50 | N80 | N90 |
|---|---|---|---|---|---|---|---|---|---|---|
| NEURAL_SAFE_REGRET | 0.227 | 0.190 | 0.159 | 0.142 | 0.138 | 0.136 | 0.135 | 5 | >500 | >500 |
| NEURAL_DECISION | 0.202 | 0.158 | 0.120 | 0.094 | 0.082 | 0.077 | 0.073 | 5 | 20 | >500 |
| NEURAL_RECON | 0.224 | 0.178 | 0.139 | 0.108 | 0.098 | 0.094 | 0.091 | 5 | 50 | >500 |
| BANK_POSTERIOR | 0.184 | 0.148 | 0.127 | 0.117 | 0.113 | 0.111 | 0.107 | 5 | 20 | >500 |
| TABULAR_EM_NASH | 0.646 | 0.567 | 0.482 | 0.354 | 0.271 | 0.214 | 0.177 | 100 | >500 | >500 |
| TABULAR_EM_UNIFORM | 0.300 | 0.263 | 0.225 | 0.158 | 0.119 | 0.083 | 0.062 | 5 | 100 | 500 |

**UNSTRUCTURED_DIRICHLET** (oracle gain at ε=0.1: 0.684)

| method | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 | N50 | N80 | N90 |
|---|---|---|---|---|---|---|---|---|---|---|
| NEURAL_SAFE_REGRET | 0.256 | 0.251 | 0.251 | 0.244 | 0.237 | 0.233 | 0.229 | 5 | >500 | >500 |
| NEURAL_DECISION | 0.261 | 0.247 | 0.239 | 0.224 | 0.213 | 0.209 | 0.207 | 5 | >500 | >500 |
| NEURAL_RECON | 0.272 | 0.258 | 0.251 | 0.243 | 0.240 | 0.239 | 0.237 | 5 | >500 | >500 |
| BANK_POSTERIOR | 0.249 | 0.245 | 0.246 | 0.242 | 0.231 | 0.225 | 0.226 | 5 | >500 | >500 |
| TABULAR_EM_NASH | 0.594 | 0.529 | 0.467 | 0.383 | 0.317 | 0.275 | 0.225 | 100 | >500 | >500 |
| TABULAR_EM_UNIFORM | 0.244 | 0.225 | 0.210 | 0.176 | 0.144 | 0.115 | 0.083 | 5 | 200 | >500 |

### Table R7. Safety audit (OpenSpiel C++ best response on every deployed strategy)

| method | strategies | LP failures | max e−ε | # e−ε > 1e−7 | 99.9% quantile of e−ε | max |fast − OpenSpiel| |
|---|---|---|---|---|---|---|
| TABULAR_EM_UNIFORM | 67200 | 0 | 1.25e-10 | 0 | 6.09e-12 | 4.44e-16 |
| TABULAR_EM_NASH | 67200 | 0 | 7.35e-10 | 0 | 4.82e-12 | 5.00e-16 |
| BANK_POSTERIOR | 67200 | 0 | 6.45e-10 | 0 | 3.80e-12 | 4.44e-16 |
| NEURAL_DECISION_s0 | 67200 | 0 | 4.52e-11 | 0 | 2.10e-12 | 4.58e-16 |
| NEURAL_DECISION_s1 | 67200 | 0 | 3.37e-10 | 0 | 2.55e-12 | 4.44e-16 |
| NEURAL_DECISION_s2 | 67200 | 0 | 1.16e-10 | 0 | 1.79e-12 | 5.55e-16 |
| NEURAL_RECON_s0 | 67200 | 0 | 2.96e-10 | 0 | 4.96e-12 | 5.00e-16 |
| NEURAL_RECON_s1 | 67200 | 0 | 5.54e-10 | 0 | 4.29e-12 | 5.00e-16 |
| NEURAL_RECON_s2 | 67200 | 0 | 3.65e-10 | 0 | 3.56e-12 | 5.00e-16 |
| NEURAL_SAFE_REGRET_s0 | 67200 | 0 | 9.42e-11 | 0 | 1.65e-12 | 5.00e-16 |
| NEURAL_SAFE_REGRET_s1 | 67200 | 0 | 9.18e-11 | 0 | 1.73e-12 | 4.44e-16 |
| NEURAL_SAFE_REGRET_s2 | 67200 | 0 | 8.26e-11 | 0 | 1.61e-12 | 5.00e-16 |

### Table R8. Prediction errors vs N (mean over held-out opponents)

| method | quantity | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 |
|---|---|---|---|---|---|---|---|---|
| TABULAR_EM_UNIFORM | q_err_mean | 0.502 | 0.499 | 0.494 | 0.483 | 0.471 | 0.455 | 0.428 |
| TABULAR_EM_UNIFORM | g_nmse_mean | 1.090 | 1.021 | 0.923 | 0.761 | 0.633 | 0.511 | 0.378 |
| TABULAR_EM_UNIFORM | g_raw_mean | 1.094 | 1.061 | 1.010 | 0.922 | 0.852 | 0.781 | 0.695 |
| TABULAR_EM_NASH | q_err_mean | 0.514 | 0.513 | 0.509 | 0.501 | 0.491 | 0.477 | 0.454 |
| TABULAR_EM_NASH | g_nmse_mean | 1.404 | 1.329 | 1.217 | 1.006 | 0.820 | 0.671 | 0.510 |
| TABULAR_EM_NASH | g_raw_mean | 1.213 | 1.181 | 1.124 | 1.019 | 0.929 | 0.855 | 0.761 |
| BANK_POSTERIOR | g_nmse_mean | 0.749 | 0.637 | 0.565 | 0.517 | 0.506 | 0.498 | 0.499 |
| BANK_POSTERIOR | g_raw_mean | 0.840 | 0.755 | 0.691 | 0.649 | 0.636 | 0.627 | 0.625 |
| NEURAL_DECISION_s0 | g_nmse_mean | 0.785 | 0.656 | 0.555 | 0.457 | 0.415 | 0.389 | 0.372 |
| NEURAL_DECISION_s0 | g_raw_mean | 0.867 | 0.776 | 0.700 | 0.619 | 0.579 | 0.556 | 0.537 |
| NEURAL_DECISION_s1 | g_nmse_mean | 0.791 | 0.666 | 0.559 | 0.467 | 0.424 | 0.398 | 0.381 |
| NEURAL_DECISION_s1 | g_raw_mean | 0.870 | 0.786 | 0.703 | 0.626 | 0.587 | 0.562 | 0.545 |
| NEURAL_DECISION_s2 | g_nmse_mean | 0.782 | 0.665 | 0.562 | 0.464 | 0.422 | 0.396 | 0.380 |
| NEURAL_DECISION_s2 | g_raw_mean | 0.859 | 0.778 | 0.700 | 0.623 | 0.586 | 0.563 | 0.546 |
| NEURAL_RECON_s0 | q_err_mean | 0.374 | 0.342 | 0.317 | 0.291 | 0.280 | 0.273 | 0.268 |
| NEURAL_RECON_s0 | g_nmse_mean | 0.819 | 0.714 | 0.634 | 0.563 | 0.532 | 0.516 | 0.503 |
| NEURAL_RECON_s0 | g_raw_mean | 0.884 | 0.802 | 0.734 | 0.669 | 0.640 | 0.622 | 0.608 |
| NEURAL_RECON_s1 | q_err_mean | 0.374 | 0.344 | 0.318 | 0.293 | 0.283 | 0.276 | 0.271 |
| NEURAL_RECON_s1 | g_nmse_mean | 0.817 | 0.720 | 0.637 | 0.561 | 0.530 | 0.511 | 0.497 |
| NEURAL_RECON_s1 | g_raw_mean | 0.880 | 0.806 | 0.738 | 0.670 | 0.639 | 0.618 | 0.603 |
| NEURAL_RECON_s2 | q_err_mean | 0.371 | 0.342 | 0.317 | 0.292 | 0.282 | 0.274 | 0.269 |
| NEURAL_RECON_s2 | g_nmse_mean | 0.810 | 0.709 | 0.629 | 0.555 | 0.525 | 0.507 | 0.495 |
| NEURAL_RECON_s2 | g_raw_mean | 0.872 | 0.796 | 0.731 | 0.665 | 0.636 | 0.616 | 0.602 |
| NEURAL_SAFE_REGRET_s0 | g_nmse_mean | 1.924 | 1.963 | 1.978 | 2.012 | 2.010 | 2.006 | 2.009 |
| NEURAL_SAFE_REGRET_s0 | g_raw_mean | 2.039 | 2.030 | 2.016 | 2.020 | 2.010 | 2.004 | 2.003 |
| NEURAL_SAFE_REGRET_s1 | g_nmse_mean | 2.942 | 2.951 | 3.039 | 3.141 | 3.170 | 3.197 | 3.207 |
| NEURAL_SAFE_REGRET_s1 | g_raw_mean | 2.949 | 2.914 | 2.896 | 2.905 | 2.896 | 2.895 | 2.891 |
| NEURAL_SAFE_REGRET_s2 | g_nmse_mean | 2.808 | 3.079 | 3.454 | 3.498 | 3.555 | 3.637 | 3.637 |
| NEURAL_SAFE_REGRET_s2 | g_raw_mean | 2.631 | 2.724 | 2.797 | 2.779 | 2.792 | 2.841 | 2.845 |

### Table R9. Effective latent dimension (participation ratio of z across test opponents) vs N

| run | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 |
|---|---|---|---|---|---|---|---|
| NEURAL_DECISION_s0 | 4.1 | 4.3 | 4.3 | 4.0 | 3.9 | 3.8 | 3.7 |
| NEURAL_DECISION_s1 | 3.8 | 4.0 | 4.0 | 3.8 | 3.7 | 3.6 | 3.5 |
| NEURAL_DECISION_s2 | 3.9 | 3.9 | 3.9 | 3.6 | 3.5 | 3.4 | 3.3 |
| NEURAL_RECON_s0 | 3.2 | 3.1 | 3.0 | 2.9 | 2.9 | 2.8 | 2.7 |
| NEURAL_RECON_s1 | 2.8 | 2.7 | 2.7 | 2.7 | 2.7 | 2.7 | 2.7 |
| NEURAL_RECON_s2 | 2.8 | 2.7 | 2.7 | 2.7 | 2.8 | 2.7 | 2.7 |
| NEURAL_SAFE_REGRET_s0 | 2.1 | 2.2 | 2.3 | 2.4 | 2.4 | 2.4 | 2.4 |
| NEURAL_SAFE_REGRET_s1 | 1.9 | 2.1 | 2.2 | 2.2 | 2.3 | 2.3 | 2.3 |
| NEURAL_SAFE_REGRET_s2 | 2.1 | 2.3 | 2.5 | 2.5 | 2.6 | 2.6 | 2.5 |

### Table R10. Strategic geometry (test opponents, ε=0.1, 20 000 pairs)

ρ(d_beh, d_resp) = 0.686; ρ(d_beh reach-weighted, d_resp) = 0.728; ρ(‖g−g'‖, d_resp) = 0.689; ρ(‖g−g'‖, d_beh) = 0.768

| representation | N | ρ(d_z, d_beh) | ρ(d_z, d_resp) | behavior-matched separation far/near [95% CI] | d_beh far / near | d_resp far / near |
|---|---|---|---|---|---|---|
| NEURAL_DECISION_s0 | 20 | 0.789 | 0.585 | 1.041 [1.026, 1.056] | 0.619 / 0.611 | 0.493 / 0.193 |
| NEURAL_DECISION_s0 | 100 | 0.797 | 0.563 | 1.005 [0.991, 1.019] | 0.619 / 0.611 | 0.493 / 0.193 |
| NEURAL_DECISION_s0 | 500 | 0.788 | 0.552 | 0.999 [0.986, 1.013] | 0.619 / 0.611 | 0.493 / 0.193 |
| NEURAL_DECISION_s1 | 20 | 0.781 | 0.579 | 1.043 [1.028, 1.058] | 0.619 / 0.611 | 0.493 / 0.193 |
| NEURAL_DECISION_s1 | 100 | 0.785 | 0.553 | 1.007 [0.993, 1.021] | 0.619 / 0.611 | 0.493 / 0.193 |
| NEURAL_DECISION_s1 | 500 | 0.779 | 0.541 | 0.996 [0.982, 1.009] | 0.619 / 0.611 | 0.493 / 0.193 |
| NEURAL_DECISION_s2 | 20 | 0.786 | 0.569 | 1.022 [1.007, 1.036] | 0.619 / 0.611 | 0.493 / 0.193 |
| NEURAL_DECISION_s2 | 100 | 0.787 | 0.541 | 0.986 [0.972, 1.000] | 0.619 / 0.611 | 0.493 / 0.193 |
| NEURAL_DECISION_s2 | 500 | 0.778 | 0.526 | 0.976 [0.962, 0.989] | 0.619 / 0.611 | 0.493 / 0.193 |
| NEURAL_RECON_s0 | 20 | 0.791 | 0.489 | 0.909 [0.894, 0.923] | 0.619 / 0.611 | 0.493 / 0.193 |
| NEURAL_RECON_s0 | 100 | 0.793 | 0.479 | 0.894 [0.879, 0.908] | 0.619 / 0.611 | 0.493 / 0.193 |
| NEURAL_RECON_s0 | 500 | 0.769 | 0.465 | 0.893 [0.878, 0.908] | 0.619 / 0.611 | 0.493 / 0.193 |
| NEURAL_RECON_s1 | 20 | 0.773 | 0.452 | 0.874 [0.860, 0.888] | 0.619 / 0.611 | 0.493 / 0.193 |
| NEURAL_RECON_s1 | 100 | 0.782 | 0.451 | 0.875 [0.861, 0.889] | 0.619 / 0.611 | 0.493 / 0.193 |
| NEURAL_RECON_s1 | 500 | 0.769 | 0.447 | 0.881 [0.867, 0.896] | 0.619 / 0.611 | 0.493 / 0.193 |
| NEURAL_RECON_s2 | 20 | 0.780 | 0.459 | 0.878 [0.864, 0.893] | 0.619 / 0.611 | 0.493 / 0.193 |
| NEURAL_RECON_s2 | 100 | 0.785 | 0.455 | 0.879 [0.865, 0.893] | 0.619 / 0.611 | 0.493 / 0.193 |
| NEURAL_RECON_s2 | 500 | 0.770 | 0.449 | 0.884 [0.869, 0.898] | 0.619 / 0.611 | 0.493 / 0.193 |
| NEURAL_SAFE_REGRET_s0 | 20 | 0.579 | 0.501 | 1.259 [1.225, 1.287] | 0.619 / 0.611 | 0.493 / 0.193 |
| NEURAL_SAFE_REGRET_s0 | 100 | 0.622 | 0.548 | 1.279 [1.247, 1.310] | 0.619 / 0.611 | 0.493 / 0.193 |
| NEURAL_SAFE_REGRET_s0 | 500 | 0.628 | 0.553 | 1.281 [1.248, 1.311] | 0.619 / 0.611 | 0.493 / 0.193 |
| NEURAL_SAFE_REGRET_s1 | 20 | 0.618 | 0.552 | 1.309 [1.278, 1.340] | 0.619 / 0.611 | 0.493 / 0.193 |
| NEURAL_SAFE_REGRET_s1 | 100 | 0.661 | 0.596 | 1.317 [1.285, 1.347] | 0.619 / 0.611 | 0.493 / 0.193 |
| NEURAL_SAFE_REGRET_s1 | 500 | 0.665 | 0.604 | 1.320 [1.289, 1.349] | 0.619 / 0.611 | 0.493 / 0.193 |
| NEURAL_SAFE_REGRET_s2 | 20 | 0.593 | 0.515 | 1.256 [1.225, 1.285] | 0.619 / 0.611 | 0.493 / 0.193 |
| NEURAL_SAFE_REGRET_s2 | 100 | 0.627 | 0.557 | 1.277 [1.246, 1.305] | 0.619 / 0.611 | 0.493 / 0.193 |
| NEURAL_SAFE_REGRET_s2 | 500 | 0.627 | 0.563 | 1.286 [1.256, 1.313] | 0.619 / 0.611 | 0.493 / 0.193 |
| true g(q) | – | – | – | 1.270 [1.250, 1.289] | 0.619 / 0.611 | 0.493 / 0.193 |
| d_beh itself (control) | – | – | – | 1.013 [1.002, 1.024] | 0.619 / 0.611 | 0.493 / 0.193 |

### Table R11. Family identifiability from short histories (train-bank posterior MAP family accuracy)

| | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 |
|---|---|---|---|---|---|---|---|
| all families | 0.76 | 0.83 | 0.86 | 0.90 | 0.91 | 0.90 | 0.90 |
| NASH_LOGIT_PERTURB | 0.96 | 0.96 | 0.96 | 0.96 | 0.97 | 0.95 | 0.92 |
| NASH_RANDOM_MIX | 0.27 | 0.47 | 0.65 | 0.77 | 0.83 | 0.85 | 0.85 |
| STRUCTURED_CORRELATED | 0.89 | 0.93 | 0.93 | 0.92 | 0.92 | 0.92 | 0.92 |
| UNSTRUCTURED_DIRICHLET | 0.92 | 0.97 | 0.91 | 0.93 | 0.93 | 0.89 | 0.89 |
| posterior mass on true family | 0.44 | 0.55 | 0.65 | 0.75 | 0.80 | 0.84 | 0.88 |

