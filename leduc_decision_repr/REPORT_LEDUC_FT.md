# REPORT_LEDUC_FT — Decision-aware fine-tuning, done cleanly (Leduc)

*Question: does instance-specific decision-awareness (SPO+) add anything beyond decision-relevance
weighting (Jacobian-weighted reconstruction)?  Reuses the V3 pipeline (encoder, heads, exact ε-safe LP,
cached oracle responses x*(g) of the 1 200 training opponents, OpenSpiel audit, 150 validation and 300 test
opponents).  ε_train = 0.10.*

## Verdict

**NO-GO: instance-specific decision-awareness adds little beyond decision-relevance weighting.**

We fine-tuned the converged RECON-JAC-889k model with full-batch SPO+ at λ* = 0.1 (chosen on validation).
Against the control, seed-averaged test safe regret at ε = 0.10 changed by:

| N | 50 | 100 | 200 | 500 |
|---|---|---|---|---|
| SPO+ − control (chips) | −0.0004 | −0.0006 | −0.0008 | −0.0019 [−0.0032, −0.0006] |

Only the N = 500 CI excludes 0.

* **P1 fails:** it required ≤ −0.008 at every one of those N.  All three seeds do point the same, small,
  favourable way.
* **P2 holds:** g-NMSE is unchanged.
* **P3 fails, and is uninformative:** the exact-decision hit rate is about 0.1% in both arms.
* **Validation vs test.**  Phase 3 was cancelled on the validation signal: SPO+ scored +0.0011
  [−0.0007, +0.0031] worse on the 450-point validation set.  That gap was within noise, and on test the sign
  reverses.  Both put the effect at ≤ 0.002 chips.  That is a quarter of the P1 bar and a tenth of the 0.019
  chips by which Jacobian weighting beat plain reconstruction in V3, so the decision does not depend on which
  one is believed.
* **SPO+ never learned much:** its training loss stayed flat (≈ 0.39) for all 1 500 steps in every seed.
* **Cancelled by user decision after Phase 2 validation:** Phase 3 (DEC-889k), Phase 4 (perturbed
  Fenchel–Young) and the EM-prior check (§8).
* **Safety:** all 302 400 deployed test strategies passed the exact audit (max Expl − ε = 1.6e−9, 0
  violations).  There was 1 LP failure, and the strategy it returned also passed.

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

## 1. Pre-training checks

`ft_checks.py` → `outputs/ft/checks.json`, run before any fine-tuning:
* **SPO+ properties on ĝ directly** (20 training opponents, 3 noise scales).  The cached x*(g) reproduces
  V_oracle to 1.1e−15.  ℓ(g, g) = 1.0e−14.  ℓ ≥ 0 (minimum 4.3e−4 over the perturbed ĝ).  Midpoint convexity
  held (minimum gap 5.4e−4).  The gradient 2x*(2ĝ − g) − 2x*(g) matched central finite differences with
  median relative error 1.1e−9 (maximum 4.6e−8).  ℓ ≥ regret held at every pair (minimum ℓ − R = 4.2e−4).
* **End-to-end chain checks** (double precision, dropout off, directional finite differences over all
  parameters, 3 random directions × 2 step sizes).  Through the RECON-JAC-889k realization-plan map
  q̂ → A·y(q̂): median relative error 3.3e−9.  Through the DEC-889k head: 3.9e−9.  ℓ ≥ regret also held at
  the models' own predictions.
* **LP cost**: 75 ms per exact safe LP, so full-batch SPO+ costs 2.4 s of LP time per step on one core.

## 2. Phase 1 — λ selection (RECON-JAC-889k seed 0; 85 min wall-clock, 4 arms in parallel)

Exact validation regret on the fixed 450-point set at each 250-step checkpoint (step 0 = the base
checkpoint, identical for all arms):

| arm | 0 | 250 | 500 | 750 | 1000 | 1250 | 1500 | selected |
|---|---|---|---|---|---|---|---|---|
| control (JAC-CE only) | 0.1136 | 0.1134 | **0.1124** | 0.1149 | 0.1128 | 0.1131 | 0.1125 | 500 |
| + 0.1·SPO+ | 0.1136 | 0.1159 | **0.1121** | 0.1158 | 0.1147 | 0.1144 | 0.1153 | 500 |
| + 0.3·SPO+ | 0.1136 | 0.1178 | **0.1123** | 0.1165 | 0.1156 | 0.1148 | 0.1157 | 500 |
| + 1.0·SPO+ | 0.1136 | 0.1191 | **0.1131** | 0.1161 | 0.1151 | 0.1147 | 0.1151 | 500 |

**λ\* = 0.1** (validation regret 0.11214 vs 0.11230 and 0.11310).  The margin over the seed-0 control
(0.11236) is 0.0002.  All four arms move together from checkpoint to checkpoint because they see
identical batches.  The SPO+ arms sit 0.002–0.006 above the control at every checkpoint except 500,
ordered by λ.

**The SPO+ arms stopped improving after step 500** (reported per the guardrail; the loss was not
changed).  Their training SPO+ loss, averaged over 250-step blocks, is essentially flat: 0.396 → 0.385
(λ = 0.1), 0.394 → 0.380 (λ = 0.3), 0.393 → 0.377 (λ = 1.0).  The single-batch values printed at validation steps range from 0.33 to 0.57
depending on the batch's N; they do not trend down.  Validation regret after step 500 is 0.114–0.117 in every SPO+ arm, never below its
step-500 value.  Validation g-NMSE barely moved (0.476–0.494, against 0.481 at the base).  No run
diverged.

Cost per SPO+ step: 2.28–2.30 s of LP time (32 exact LPs) + 0.90–0.93 s of model time; 48 000 training LPs
per arm; 84–85 min per arm.  Controls: 0.91 s per step, 27 min.  Validation strategies: 7 checkpoints × 450
per arm, all audited, max Expl − ε = 8.6e−11, 0 violations, 0 LP failures.

## 3. Phase 2 — RECON-JAC-889k: control vs + 0.1·SPO+ (3 seeds)

### 3.1 Training and validation (83 min for SPO+ seeds 1–2, run concurrently; seed 0 = the Phase 1 λ = 0.1 run)

Exact validation regret on the fixed 450-point set (step 0 = base checkpoint, not a selection candidate):

| run | 0 | 250 | 500 | 750 | 1000 | 1250 | 1500 | selected (step) |
|---|---|---|---|---|---|---|---|---|
| control s0 | 0.1136 | 0.1134 | 0.1124 | 0.1149 | 0.1128 | 0.1131 | 0.1125 | 0.1124 (500) |
| SPO+ s0 | 0.1136 | 0.1159 | 0.1121 | 0.1158 | 0.1147 | 0.1144 | 0.1153 | 0.1121 (500) |
| control s1 | 0.1148 | 0.1184 | 0.1175 | 0.1161 | 0.1143 | 0.1137 | 0.1140 | 0.1137 (1250) |
| SPO+ s1 | 0.1148 | 0.1211 | 0.1182 | 0.1171 | 0.1165 | 0.1160 | 0.1160 | 0.1160 (1500) |
| control s2 | 0.1139 | 0.1150 | 0.1143 | 0.1122 | 0.1118 | 0.1121 | 0.1136 | 0.1118 (1000) |
| SPO+ s2 | 0.1139 | 0.1162 | 0.1161 | 0.1133 | 0.1149 | 0.1132 | 0.1142 | 0.1132 (1250) |

Seed means of the selected checkpoints: control 0.1126, SPO+ 0.1138.

* **SPO+ s1 never got back to its starting point.**  Its selected checkpoint (0.1160) is worse than its
  own step 0 (0.1148).  The rule excludes step 0, so it is deployed as pre-registered.
* **SPO+ stopped improving, reported per the guardrail with the loss unchanged.**  The training SPO+ loss,
  averaged over 250-step blocks, was essentially flat in every seed:
  - s0: 0.396 → 0.385
  - s1: 0.395 → 0.389
  - s2: 0.397 → 0.385
* **The anchor was untouched.**  The JAC cross-entropy block averages (0.657–0.666) match the paired
  controls' within 0.001 in every block.  At λ = 0.1 the SPO+ term does not move the anchor.
* No run diverged.

### 3.2 Test safe regret at ε = 0.10 (300 opponents × 8 streams, seeds averaged)

| N | 5 | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|
| base RECON-JAC-889k (V3) | 0.1755 | 0.1511 | 0.1309 | 0.1115 | 0.1025 | 0.0971 | 0.0926 |
| control (JAC-CE, 1 500 more steps) | 0.1747 | 0.1509 | 0.1306 | 0.1113 | 0.1023 | 0.0963 | 0.0924 |
| + 0.1·SPO+ | 0.1729 | 0.1507 | 0.1312 | 0.1109 | 0.1017 | 0.0955 | 0.0905 |
| **SPO+ − control** | −0.0019 | −0.0003 | +0.0006 | −0.0004 | −0.0006 | −0.0008 | **−0.0019** |
| 95% CI | [−0.0041, +0.0003] | [−0.0017, +0.0012] | [−0.0006, +0.0018] | [−0.0014, +0.0007] | [−0.0016, +0.0005] | [−0.0021, +0.0004] | [−0.0032, −0.0006] |

Per seed, SPO+ − control (\* = CI below 0):

| seed | 5 | 10 | 20 | 50 | 100 | 200 | 500 | mean over N ≥ 50 |
|---|---|---|---|---|---|---|---|---|
| s0 | +0.0008 | +0.0008 | +0.0010 | +0.0000 | +0.0001 | −0.0000 | −0.0013 | −0.0003 |
| s1 | −0.0033\* | −0.0010 | +0.0003 | −0.0007 | −0.0007 | −0.0008 | −0.0017\* | −0.0010 |
| s2 | −0.0031\* | −0.0006 | +0.0004 | −0.0004 | −0.0011 | −0.0017 | −0.0027\* | −0.0014 |

**P1 fails.**  The per-seed sign condition holds (all three means over N ≥ 50 are negative).  The magnitude
condition does not: the largest seed-averaged difference is −0.0019, against a required ≤ −0.008 at every N
from 50 to 500, and three of those four CIs include 0.

Per family, SPO+ − control (\* = CI below 0, † = CI above 0):

| family | 5 | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|
| NASH-LOGIT | −0.0106\* | −0.0054\* | −0.0011 | −0.0003 | +0.0001 | −0.0003 | −0.0006 |
| NASH-MIX | +0.0035† | +0.0045† | +0.0031† | +0.0016† | +0.0012† | +0.0009 | +0.0002 |
| STRUCTURED | −0.0016 | −0.0010 | −0.0004 | −0.0027\* | −0.0029\* | −0.0038\* | −0.0046\* |
| DIRICHLET | +0.0012 | +0.0008 | +0.0007 | −0.0001 | −0.0007 | −0.0002 | −0.0026 |

The only consistent large-N gain is on STRUCTURED-CORRELATED opponents (−0.003 to −0.005 at N ≥ 50).
NASH-RANDOM-MIX opponents lose slightly at N ≤ 100.

### 3.3 g-NMSE (P2)

| N | 5 | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|
| control | 0.793 | 0.665 | 0.555 | 0.451 | 0.405 | 0.377 | 0.359 |
| + SPO+ | 0.792 | 0.664 | 0.554 | 0.450 | 0.405 | 0.377 | 0.358 |
| diff (upper CI) at N ≥ 50 | | | | −0.0005 (+0.0009) | +0.0000 (+0.0014) | −0.0002 (+0.0012) | −0.0010 (+0.0006) |

**P2 holds:** no CI lies entirely below 0.  SPO+ neither improved nor degraded the implied g.

## 4. Decision hit rate (P3) and near-hits (ε = 0.10, seeds averaged)

Fraction of (history, N) whose deployed strategy has regret < 1e−6 (pre-registered), and descriptively
< 1e−3 and < 1e−2:

| N | 5 | 10 | 20 | 50 | 100 | 200 | 500 | pooled |
|---|---|---|---|---|---|---|---|---|
| hit < 1e−6, control | 0 | 0 | 0.0003 | 0.0006 | 0.0008 | 0.0022 | 0.0024 | 0.089% |
| hit < 1e−6, SPO+ | 0 | 0.0001 | 0.0003 | 0.0010 | 0.0017 | 0.0022 | 0.0025 | 0.111% |
| hit < 1e−6, base | 0 | 0 | 0 | 0.0007 | 0.0011 | 0.0028 | 0.0026 | 0.103% |
| < 1e−3, control | 0 | 0.002 | 0.003 | 0.004 | 0.007 | 0.011 | 0.014 | |
| < 1e−3, SPO+ | 0 | 0.001 | 0.003 | 0.005 | 0.009 | 0.010 | 0.013 | |
| < 1e−2, control | 0.012 | 0.029 | 0.050 | 0.073 | 0.088 | 0.111 | 0.126 | |
| < 1e−2, SPO+ | 0.014 | 0.031 | 0.050 | 0.071 | 0.090 | 0.113 | 0.136 | |

Pooled differences, SPO+ − control:
* < 1e−6: +0.02 pp [−0.06, +0.13] — **P3 fails**.
* < 1e−3: −0.01 pp [−0.14, +0.15].
* < 1e−2: +0.22 pp [−0.08, +0.55].

Exact decision recovery is essentially absent in both arms, as it was at the base.  A 1 093-dimensional
safe response to an estimated g almost never lands on the oracle's face.

## 5. ε transfer (trained at ε = 0.10), SPO+ − control

| ε | 5 | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|
| 0.05 | −0.0014\* | −0.0005 | +0.0006 | +0.0001 | +0.0000 | −0.0004 | −0.0010 |
| 0.10 | −0.0019 | −0.0003 | +0.0006 | −0.0004 | −0.0006 | −0.0008 | −0.0019\* |
| 0.20 | −0.0016 | +0.0003 | +0.0001 | −0.0002 | −0.0006 | −0.0017\* | −0.0025\* |

\* = CI below 0.  Regret levels at N = 500: control 0.0711 (ε = 0.05) and 0.1270 (ε = 0.20); SPO+ 0.0702 and
0.1245.  The small large-N edge is slightly larger at ε = 0.20 and vanishes at ε = 0.05.

## 6. Fine-tuned arms vs their base checkpoints (ε = 0.10)

| N | 5 | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|
| control − base | −0.0007 | −0.0001 | −0.0003 | −0.0002 | −0.0003 | −0.0007 | −0.0002 |
| SPO+ − base | −0.0026\* | −0.0004 | +0.0003 | −0.0006 | −0.0008 | −0.0016\* | −0.0021\* |
| g-NMSE, control − base | +0.0009 | −0.0016 | −0.0021 | −0.0034\* | −0.0042\* | −0.0040\* | −0.0041\* |

* **Continued training alone:** 1 500 more steps at a low learning rate lowered g-NMSE by about 0.004 at
  N ≥ 50 but did not change regret (all CIs include 0).
* **With SPO+:** regret improved by 0.002 at N ≥ 200, on top of an unchanged g-NMSE.

## 7. How noisy was the validation signal?  (post hoc, validation only; `ft_val_noise.py`)

The selected checkpoints were re-scored per point on the 450-point validation set; the stored means are
reproduced to 1e−14.  Differences SPO+ − control are paired over the 150 validation opponents (averaged over
the three N) with bootstrap CIs:

| seed | SPO+ − control | 95% CI |
|---|---|---|
| s0 | −0.0002 | [−0.0019, +0.0014] |
| s1 | +0.0022 | [−0.0005, +0.0050] |
| s2 | +0.0014 | [−0.0024, +0.0050] |
| seed mean | +0.0011 | [−0.0007, +0.0031] |

"SPO+ trails its control on every seed" was true of the point estimates but not significant for any seed.
The test comparison has 16× more data points per arm (300 × 8 vs 150 × 1 per N) and reverses the sign.
Both signals agree that the effect is ≤ 0.002 chips in magnitude.  These selected-checkpoint values are also
optimistic (a minimum over 6 checkpoints), equally for both arms.

## 8. Phase 3, Phase 4 and the EM-prior check — cancelled

After Phase 2 validation, the user cancelled Phase 3 (DEC-889k control vs MSE + λ*·SPO+), Phase 4
(perturbed Fenchel–Young) and the EM-prior check.  The user's reason: "validation shows SPO+ trailing its
control on every seed, and the remaining phases would not change the direction of the project."

* P4 is therefore untested.
* The two Phase 3 control runs that had started (DEC seeds 0–1) were stopped at about step 200 and
  discarded; no result from them is used.
* §7 shows that the validation gap was within noise and that test reverses its sign.  That does not change
  the P1 outcome or the verdict, since both put the effect far below the pre-registered bar.

## 9. Audit table

| deployed strategies | n | max Expl − ε | violations (> 1e−7) | LP failures |
|---|---|---|---|---|
| test, control s0 | 50 400 | 2.5e−10 | 0 | 0 |
| test, control s1 | 50 400 | 9.7e−10 | 0 | 1 |
| test, control s2 | 50 400 | 1.6e−9 | 0 | 0 |
| test, SPO+ s0 | 50 400 | 1.1e−9 | 0 | 0 |
| test, SPO+ s1 | 50 400 | 6.5e−10 | 0 | 0 |
| test, SPO+ s2 | 50 400 | 1.6e−9 | 0 | 0 |
| validation, all 9 FT runs × 7 checkpoints | 28 350 | 8.6e−11 | 0 | 0 |

* **The one LP failure** (control s1, ε = 0.05, N = 10) is a non-optimal solver status.  The strategy it
  returned was still audited: Expl − ε = 2e−15.  It is counted in the regret averages and excluded only from
  the "max" column above.
* **Not audited:** the 48 000 training LPs per SPO+ arm are gradient targets, not deployed strategies.

## 10. Wall-clock and LP cost

| phase | wall-clock (UTC) | notes |
|---|---|---|
| Phase 1 (4 arms) + controls s0–s2 | 15:01:49 – 16:27:10 (85 min) | 4 cores; controls ran one after another in the 4th slot |
| Phase 2 SPO+ training, seeds 1–2 | 16:27:10 – 17:50:21 (83 min) | alongside control evaluations |
| Phase 2 test evaluations (6 arms) | 16:27:10 – 18:19:04 | ~21–23 min each on 2 workers (50 400 LPs + audits) |
| whole study | 3 h 17 min | |

* **Per SPO+ step:** 2.23–2.30 s of LP time (32 exact LPs, ≈ 70 ms each) plus 0.90–0.93 s of model time.
  An SPO+ arm takes 83–85 min, against 27 min for a control (3.1×).
* **Post hoc:** the validation re-scoring (§7) took 2 min 45 s.

## 11. Declared deviations

1. **Cancelled phases.**  Phase 3, Phase 4 and the EM-prior check were cancelled by user decision (§8).
   Two partially trained Phase 3 control runs were killed and discarded.
2. **SPO+ stall.**  The SPO+ training loss stayed essentially flat for all 1 500 steps in every seed, and
   validation regret never improved after step 500 in Phase 1 (§2, §3.1).  This is reported, not acted on.
3. **SPO+ s1 below its base.**  Its selected checkpoint is worse on validation than the base it started
   from.  It was deployed anyway, because step 0 is not a candidate under the pre-registered rule.
4. **Evaluation retries.**  Two control evaluations failed on first launch, within 20 s and before any
   prediction, because fine-tuning run directories lacked `config.json` (architecture only).  `ft_eval.py`
   now copies it from the base run; the weights are the fine-tuned `best.pt`.  Both were re-run from scratch.
5. **P3 uninformative.**  Hit rates are about 0.1%, so the near-hit rates (1e−3, 1e−2) are added as
   descriptive statistics, not tests.
6. **Post-hoc analysis.**  The validation-noise analysis (§7) was added after the fact and uses validation
   data only.
