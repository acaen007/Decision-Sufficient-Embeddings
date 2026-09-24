# REPORT_LEDUC_FT — Decision-aware fine-tuning, done cleanly (Leduc)

*Question: does instance-specific decision-awareness (SPO+) add anything beyond decision-relevance
weighting (Jacobian-weighted reconstruction)?  Reuses the V3 pipeline (encoder, heads, exact ε-safe LP,
cached oracle responses x*(g) of the 1 200 training opponents, OpenSpiel audit, 150 validation and 300 test
opponents).  ε_train = 0.10.*

## 0. Pre-registration (written 2026-09-24 before any check, timing run or fine-tuning; not edited afterwards)

**Losses.**  SPO+ in maximization form, ℓ(ĝ, g) = max_{x∈S_ε}(2ĝ − g)ᵀx − 2ĝᵀx*(g) + gᵀx*(g), gradient
2x*(2ĝ − g) − 2x*(g), with ε = 0.10.  The target x*(g) is the **cached** oracle response
(`outputs/weights_v3/xstar_train_eps0.1.npy`).  Oracle optima are non-unique faces (the geometry diagnostic
found equal-value optimal vertices up to 0.57 apart), and the cache fixes one consistent choice per opponent.
ĝ enters the LP in raw chips: for RECON-JAC, ĝ = A·y(q̂) through the differentiable realization-plan map
(products of probabilities along sequences), so the SPO+ gradient reaches q̂ by autograd; for DEC, the
head's un-standardized output.  Anchors are always kept: Jacobian-weighted cross-entropy for RECON-JAC,
standardized MSE for DEC.  Loss = anchor + λ·mean_batch(SPO+); control = anchor only (λ = 0, no LPs).

**Fixes to V3's SPO+ protocol.**  (1) Full-batch SPO+: one exact LP for every one of the 32 examples in every
step (the LPs of one arm are solved in that arm's process; 3–4 arms run concurrently on the 4 cores).
(2) Checkpoint selection by **exact validation regret on a fixed set**: 150 validation opponents ×
N ∈ {20, 100, 500} × stream 0 = 450 deployed strategies, regret V_ε(g) − gᵀx_deployed(ĝ), evaluated at
fine-tuning steps 250, 500, …, 1500; the checkpoint with the lowest mean is kept.  The same rule applies to
every arm including controls.  Step 0 (the base checkpoint) is evaluated too, for reference only; it is not
a candidate, so every arm deploys a fine-tuned model.  (3) Anchor always on.  (4) Three seeds.

**Fine-tuning protocol (identical for all arms).**  Start from the converged base checkpoint (`best.pt`):
RECON-JAC-889k seeds 0–2 (`runs_v3/recjac889k_s*`) and DEC-889k seeds 0–2 (`runs/decision_s*`).  1 500
steps, batch 32, N drawn per batch from {5, 10, 20, 50, 100, 200, 500} as in base training; fresh AdamW
(weight decay 0.01); learning rate linear warm-up 0 → 1e−4 over 100 steps, cosine decay to 1e−5 at step 1 500
(the base runs ended at 3e−5); gradient clipping 1.0; dropout as in base training.  Batches are drawn from a
fine-tuning RNG seeded by the seed alone, so the control and SPO+ arm of a seed see identical batches.

**Phases.**
* Phase 1 (seed 0, RECON-JAC base): λ ∈ {0.1, 0.3, 1.0}.  λ* = the λ whose selected checkpoint has the
  lowest mean validation regret over the 450 points (ties → smaller λ).
* Phase 2 (RECON-JAC base, seeds 0–2): control (JAC-CE only) vs JAC-CE + λ*·SPO+ (seed 0 reuses the
  Phase-1 run for λ*).
* Phase 3 (DEC-889k base, seeds 0–2, if time): control (MSE only) vs MSE + λ*·SPO+, same λ*.
* Phase 4 (optional): perturbed Fenchel–Young, RECON-JAC base seed 0, anchor kept, weight λ*: gradient
  mean_k x*(ĝ + σZ_k) − x*(g), K = 4, Z ~ N(0, I); σ ∈ {0.5, 2.0} × σ₀ with σ₀ = RMS over training opponents and
  coordinates of (g − ḡ); σ picked by the same validation rule.

**Test evaluation.**  300 test opponents × N ∈ {5, 10, 20, 50, 100, 200, 500} × 8 streams, at
ε ∈ {0.05, 0.10, 0.20} (primary 0.10); seeds averaged and per seed; paired bootstrap 95 % CIs over opponents
(streams averaged within opponent, 2 000 resamples).  Per arm: safe regret, g-NMSE (standardized per
coordinate, as in V1–V3), decision hit rate = fraction of (history, N) with regret < 1e−6, per family.
EM-prior check: learned-prior EM (κ_N from validation, as in V3 T5c) with the 3-seed fine-tuned RECON-JAC+SPO+
q̂ as prior, versus the existing learned-prior EM and versus the same recipe with the control's q̂.

**Predictions** (Phase 2 unless stated; "difference" = SPO+ arm − control, paired over opponents).
* P1: at ε = 0.10 and each N ∈ {50, 100, 200, 500}, the seed-averaged regret difference is ≤ −0.008 chips with
  its 95 % CI excluding 0; and in each of the 3 seeds separately, the mean difference over those four N is
  negative.
* P2: at each N ∈ {50, 100, 200, 500}, the SPO+ arm's test g-NMSE is not significantly lower than the
  control's (the paired CI of the g-NMSE difference is not entirely below 0) — equal or worse.
* P3: the decision hit rate at ε = 0.10, pooled over the 7 N and averaged over seeds, is higher for the SPO+
  arm, with the paired CI of the difference excluding 0.
* P4 (Phase 3): the seed-averaged DEC-889k + λ*·SPO+ minus control difference, averaged over N ∈ {50, …, 500},
  is below −0.008, with CIs excluding 0 at each of those N.

**Decision rule.**  GO for the decision-aware line if P1 holds.  Otherwise the headline is that
instance-specific decision-awareness adds little beyond decision-relevance weighting.

**Guardrails.**  Every deployed strategy (validation and test) is audited with OpenSpiel's best response,
Expl ≤ ε + 1e−7; violations (must be 0) and LP failures are reported.  Nothing is tuned on test opponents.
If an SPO+ run diverges or stops improving, that is reported with the step; the loss is not changed
mid-run.  Wall-clock per phase and LP time per step are reported.
