# Latent-manifold coverage diagnostic (Leduc, no training)

## Verdict

**Decision rule: POOR.  This latent space is a space of *beliefs*, not of *opponents*, and it cannot carry
latent-space drift tracking as it stands.**

* **In-distribution ceiling below the bar.**  Inside the training latent cloud, the best latent point for
  each opponent earns 0.84 of the attainable safe gain on in-distribution test opponents (bar: 0.90).  That
  is below what tabular EM (0.85) and the learned-prior EM (0.90) actually achieve from 500 observed hands.
* **Training opponents too.**  It earns 0.86 even on training opponents (g-space R² 0.82).  The trivial
  nearest-training-opponent ceiling there is 1.0.
* **Structured families are the exception:** 0.95 for FAR-ARCH and 0.91 for FAR-CFR.
* **Far families are worst:** 0.69 for NEAR and 0.81 for FAR-EXPL.

**The space is low-dimensional, as hoped.**  About 8 principal directions hold 97.5% of the training cloud,
and the ceiling saturates by k ≈ 8 (0.82 → 0.83 in-distribution from k = 8 to full).  The problem is not
dimension.  Even with those dimensions, the decoder cannot reproduce individual opponents precisely.

**Two observations locate the problem.**
* **Inference is already near the ceiling.**  The JAC-opp encoder at N = 500 is only 0.03–0.08 below the
  in-cloud ceiling on every family.  So the neural models' large-N floor found in the earlier studies is a
  *representation* ceiling, not an inference gap.
* **Every opponent's best-fitting point lies outside the cloud.**  Unconstrained fits leave the training
  cloud for 100% of opponents, *including training opponents*, and reach higher ceilings: ≥ 0.89
  in-distribution and ≥ 0.91 on training opponents, and still rising when stopped.
  - The cloud is where the encoder puts its (shrunk, posterior-mean-like) beliefs after ≤ 500 censored
    hands.
  - The opponents themselves sit outside it, in a region the decoder only reaches by extrapolation.

**Drift paths.**
* **They stay near the manifold:** ceilings of 0.91–0.93 all along the paths.  The midpoint is somewhat
  further off (relative g error 0.39 vs ≤ 0.35 at the endpoints).
* **But the latent geometry is not linear.**  Fitted latent paths are curved (length 2.3× the endpoint
  distance).  Decoding the straight latent line between the endpoints is much worse mid-path (error 0.59,
  ceiling 0.89).  "Drift = straight-line motion in this latent space" does not hold.

**What this means for the plan.**
* **Filtering with this decoder would inherit the ceiling.**  Latent-space filtering of drift with the
  *current* decoder would track a belief manifold whose best point earns 0.84 in-distribution.  That is
  worse than the windowed learned-prior EM already achieves (0.85 before a switch, 0.82 under drift).
* **Build an opponent manifold first.**  The representation to build is a *generative* opponent space:
  a decoder that reproduces individual opponents (ceiling ≥ 0.95 on training and in-distribution opponents)
  with inference kept separate.  Section 4 has the concrete proposal.

## 0. Pre-registration

Written and committed before any fit was run.  Not edited afterwards; deviations go in the deviations
section.

### Question
If inference were perfect, how much of each opponent family can the learned latent space represent, as
measured by what you can *earn* against the opponent?  And do straight-line drift paths between two opponents
stay on the latent manifold?

### Objects
* **Decoder** (primary): the reconstruction head of JAC-opp seed 0 (`runs_jacopp/a3_s0`), z ∈ R^128 →
  q(z) = masked softmax (144 × 3) → g(z) = A·y(q(z)).  Secondary decoder: RECON-889k seed 0
  (`runs_v3/rec889k_s0`, plain cross-entropy), full-dimension fits only.
* **Training latent cloud:** the matching encoder's z on the training opponents at N = 500, 4 streams each
  (4 800 points).  Its PCA (mean μ, eigenpairs λ_i, V_i) defines whitened coordinates w.
* **"On-manifold" region of dimension k:** z = μ + V_k diag(√λ_k) w with ‖w‖ ≤ r_k, where r_k is the 99th
  percentile of ‖w‖ over the training cloud projected on the top k components.  k ∈ {2, 4, 8, 16, 32,
  full}, where "full" = every component with λ_i > 1e−6 · λ_max.
* **"Free" fit:** z ∈ R^128 unconstrained.  This measures the decoder's capacity, not the training region.
* **Fit:** minimize ‖g(z) − g*‖² (raw chips) with Adam (lr 0.05, 600 steps).  After each step, project onto
  the ball ‖w‖ ≤ r_k for on-manifold fits.  Two initializations: μ, and the encoder z of the training
  opponent nearest in g-space.  The lower final loss is kept.
* **Ceiling:** deploy x = exact ε-safe LP(g(z*)) at ε = 0.10 against the true opponent (OpenSpiel audit of
  every strategy).  Report the fraction of attainable safe gain (pooled ratio of means) and regret.
* **Comparisons** (from the generalization study, same opponents, N = 500, streams averaged):
  - the JAC-opp encoder (amortized inference), TAB-EM and the PRIOR-EM hybrid;
  - the 1-NN bank ceiling: deploy the LP on the g of the nearest training opponent, a "discrete manifold"
    ceiling.
* **Also reported:**
  - the g-space R² = 1 − ‖g(z*) − g*‖² / ‖g* − ḡ_train‖²;
  - the whitened distance of z* from the training cloud (nearest training point, relative to the 99th
    percentile of within-cloud nearest-neighbour distances);
  - where the encoder's own z for each opponent lies relative to the cloud.
* **Opponents:**
  - the five non-NE families of the generalization study (100 each);
  - 200 training opponents (a sanity check: the decoder should fit what it was trained on);
  - the NE family (projection error only; fraction undefined).
* **Drift paths:** the 40 DRIFT pairs of the non-stationary study, λ ∈ {0, 0.1, …, 1}, target
  q_λ = (1 − λ) q_A + λ q_B.
  - Fitted on-manifold (full k) and free.
  - Also the *latent straight line* z_lin(λ) = (1 − λ) z*(0) + λ z*(1), decoded.
  - Path metrics: projection error vs λ, ceiling vs λ, and tortuosity (fitted path length / endpoint distance
    in w).

### Expectations
* **M1:** ID-REF on-manifold (full k) ceiling fraction ≥ 0.95.
* **M2:** the NEAR and FAR-EXPL on-manifold ceilings are each ≥ 0.10 below ID-REF's.
* **M3:** on NEAR and FAR-EXPL, the free-fit ceiling exceeds the on-manifold ceiling by ≥ 0.05, and z*_free
  lies outside the training 99% ball for the majority of their opponents.
* **M4:** the encoder's own z (N = 500) lies inside the training 99% ball for ≥ 90% of opponents in every
  family.  The encoder snaps far opponents into the training region.
* **M5:** on drift paths, the median relative projection error at λ = 0.5 exceeds its value at the
  endpoints, and the decoded latent straight line is worse than the fitted path at λ = 0.5.

### Decision rule for the next step
Using the on-manifold, full-k ceiling of the JAC-opp decoder:
* **GOOD** (go straight to latent filtering on this latent space): ≥ 0.90 on every non-NE family, and ≥ TAB-EM
  at N = 500 on every family.
* **PARTIAL:** ≥ 0.90 on ID-REF, FAR-ARCH and FAR-CFR but not on NEAR and/or FAR-EXPL.  Latent filtering is
  viable in-support; broaden the training population before out-of-support drift work.
* **POOR:** < 0.90 on ID-REF.  The representation must be fixed first.

The ceiling-vs-k curve is reported as the evidence on "is the opponent space low-dimensional".

### Guardrails
* Exact audit of every deployed strategy (Expl ≤ ε + 1e−7; 0 violations; LP failures reported).
* Nothing is trained or tuned; the fit settings above are fixed in advance.
* Convergence of the fits is reported (final loss and its change over the last 100 steps).

## 1. Results (`manifold_diag.py`, `manifold_analysis.py`; figures in `outputs/manifold/`)

### 1.1 Ceilings vs what methods achieve (fraction of attainable safe gain, ε = 0.10; ratio of means, 95% CI)

| | ID-REF | NEAR | FAR-ARCH | FAR-CFR | FAR-EXPL | TRAIN (100) |
|---|---|---|---|---|---|---|
| 1-NN training opponent (discrete bank ceiling) | 0.77 | 0.64 | 0.91 | 0.78 | 0.70 | 1.00 |
| JAC-opp encoder, N = 500 (achieved) | 0.82 | 0.66 | 0.92 | 0.83 | 0.76 | — |
| **latent ceiling, inside the training cloud (full k)** | **0.84** [0.79, 0.89] | **0.69** [0.64, 0.73] | **0.95** [0.94, 0.97] | **0.91** [0.88, 0.93] | **0.81** [0.79, 0.84] | **0.86** |
| latent ceiling, unconstrained (lower bound; not converged) | 0.89 | 0.81 | 0.99 | 0.92 | 0.88 | 0.91 |
| TAB-EM, N = 500 (achieved) | 0.85 | 0.90 | 0.84 | 0.80 | 0.85 | — |
| PRIOR-EM (JAC-opp), N = 500 (achieved) | 0.90 | 0.90 | 0.93 | 0.87 | 0.91 | — |

g-space R² of the in-cloud fits:

| ID-REF | NEAR | FAR-ARCH | FAR-CFR | FAR-EXPL | TRAIN | NE (projection only) |
|---|---|---|---|---|---|---|
| 0.81 | 0.57 | 0.96 | 0.81 | 0.72 | 0.82 | 0.89 |

`fig1_ceilings.png`.  A continuous latent space beats the discrete training bank (0.84 vs 0.77
in-distribution), but not the tabular likelihood.

### 1.2 Dimension (`fig2_dimension.png`; subset of 30 per family + 30 training opponents)

| k (principal directions allowed) | 2 | 4 | 8 | 16 | 32 | full (108) | unconstrained |
|---|---|---|---|---|---|---|---|
| ID-REF ceiling | 0.67 | 0.77 | 0.82 | 0.83 | 0.83 | 0.83 | 0.87 |
| TRAIN ceiling | 0.68 | 0.78 | 0.87 | 0.89 | 0.89 | 0.89 | 0.93 |
| FAR-ARCH ceiling | 0.71 | 0.91 | 0.97 | 0.97 | 0.96 | 0.96 | 0.98 |
| NEAR ceiling | 0.53 | 0.63 | 0.74 | 0.76 | 0.75 | 0.74 | 0.86 |
| training cloud variance explained (JAC-opp) | 0.49 | 0.71 | 0.975 | 0.997 | 0.999 | 1 | — |

The encoder uses about 8 of its 128 dimensions.  Ceilings saturate there.  The plain-cross-entropy decoder
(RECON-889k) has a *lower* ceiling on the same subset (ID-REF 0.81 vs 0.83; TRAIN 0.80 vs 0.89).  The
Jacobian decision-relevance weighting improved the decoder's fidelity where it matters.

### 1.3 Where the fitted points and the encoder sit (top-16 principal coordinates, whitened)

* **Unconstrained optima leave the cloud** (outside its 99% ball) for 94–100% of opponents in every family
  (FAR-ARCH 69%), *including training opponents (94%)*.  Their median nearest-neighbour distance to the
  cloud is 6–16, against 2.7 for the cloud's own 99th percentile.
* **In-cloud optima** lie inside the ball but in sparse regions: 77% of training opponents' optima are
  further from any cloud point than the 99th-percentile within-cloud spacing.
* **Encoder z at N = 500:** inside the full-dimension 99% ball for 98% (ID-REF), 100% (FAR-ARCH, FAR-CFR),
  75% (NEAR) and 43.5% (FAR-EXPL) of histories.  The encoder does *not* snap far opponents into the cloud;
  it extrapolates, and its extrapolations are no better than the in-cloud ceiling.

### 1.4 Drift paths (`fig3_drift_paths.png`; 40 DRIFT pairs × λ ∈ {0, 0.25, 0.5, 0.75, 1})

| λ | 0 | 0.25 | 0.5 | 0.75 | 1 |
|---|---|---|---|---|---|
| median relative g error, fitted in-cloud | 0.35 | 0.34 | 0.39 | 0.36 | 0.25 |
| median relative g error, decoded latent straight line | 0.35 | 0.46 | 0.59 | 0.50 | 0.25 |
| ceiling, fitted in-cloud | 0.91 | 0.93 | 0.92 | 0.93 | 0.91 |
| ceiling, latent straight line | 0.91 | 0.92 | 0.89 | 0.91 | 0.91 |

* **Path shape.**  Fitted path tortuosity (length / endpoint distance) has median 2.27 (90th percentile
  2.66).  A straight behavioural path is a curved latent path.
* **Where paths go.**  Mixtures of two in-support opponents stay near the manifold and are about as
  representable as the endpoints.  The pool mixes test opponents and archetypes, and archetypes are very
  representable.

## 2. Pre-registered expectations

* **M1 — fails:** ID-REF in-cloud ceiling 0.843 < 0.95.
* **M2 — fails.**  The NEAR ceiling is 0.15 below ID-REF, but FAR-EXPL is only 0.03 below: the
  in-distribution ceiling is itself low.
* **M3 — holds.**  The unconstrained ceiling exceeds the in-cloud ceiling by 0.12 (NEAR) and 0.07
  (FAR-EXPL), and 100% of those unconstrained optima lie outside the ball.  The same happens on *every*
  family, training opponents included, so this is not specific to far opponents.
* **M4 — fails.**  The encoder keeps 98–100% of ID-REF, FAR-ARCH and FAR-CFR histories inside the ball,
  but only 75% (NEAR) and 43.5% (FAR-EXPL).  It extrapolates rather than snapping.
* **M5 — holds.**  Mid-path error is 0.39 against at most 0.35 at the endpoints, and the latent straight
  line reaches 0.59 at λ = 0.5.
* **Decision rule — POOR.**  The ID-REF in-cloud ceiling is < 0.90.  The ceiling is ≥ 0.90 on FAR-ARCH
  (0.95) and FAR-CFR (0.91) only.

## 3. Audits and convergence

* **Audits:** 4 055 deployed strategies from fits and ceilings, all passed (max Expl − ε = 1.1e−11,
  0 violations, 0 LP failures), plus 1 610 oracle-value LPs.
* **Convergence at 300 steps.**
  - The in-cloud fits have converged in every family but one: the median relative loss change over the
    last 100 steps is within ±0.5%, with the 90th percentile ≤ 1%.
  - The exception is FAR-ARCH, where the median loss *rose* 5% over the last 100 steps.  That looks like
    projected-Adam oscillation at the ball boundary, so its 0.95 ceiling may be slightly understated.
  - The unconstrained fits have *not*: the median change is 12–31%.  Their ceilings are lower bounds.
* **Initialization.**  Starting from the cloud mean instead of the nearest training opponent's encoding
  changes subset ceilings by −0.03 to +0.02 (FAR-CFR 0.85 vs 0.89), so local optima matter only modestly.
* **Wall-clock:** 39 min in total (decoder fits 32 min for JAC-opp and 3.4 min for RECON-889k; LPs 2 min).

## 4. What to build next

1. **A generative opponent manifold, not a belief encoder.**  Train a decoder z → q on the *true* opponent
   policies, as an autoencoder or latent-variable model over q, with the JAC-opp decision-weighted loss and
   a broadened opponent population (archetypes, exploiters, learners).  Require this diagnostic's in-cloud
   ceiling ≥ 0.95 on training and in-distribution opponents before anything else.  The current decoder only
   ever learned to decode what an encoder could infer from ≤ 500 censored hands, which is shrunk toward the
   population.
2. **Inference in that space, kept separate.**  Either an amortized encoder trained to output the *true*
   opponent's z, or Bayesian filtering in z with the exact hidden-card hand likelihood.  Filtering supports
   drift directly through a random-walk or learned dynamics prior.
3. **Keep the tabular likelihood in the loop.**  Every study so far shows it covers what any manifold misses
   (PRIOR-EM ≥ 0.87 on every family).
4. **Linearity is a design choice.**  If "drift = straight-line motion" is wanted, it has to be trained in,
   for example with interpolation-consistency losses.  The current latent geometry is curved (tortuosity
   2.3).

## 5. Deviations (all compute-driven and fixed before any fit ran)

1. **Adam steps.**  300 instead of 600: the decoder costs ≈ 4 ms per target-step, and the pre-registered
   design would have taken ≈ 10 h.  In-cloud fits converged; unconstrained fits did not, so their ceilings
   are lower bounds.
2. **Initialization.**  Only the nearest-training-opponent initialization for the bulk fits.  Both
   initializations were compared on the subset (§3).
3. **Subset fits.**  The k-grid, the secondary decoder (RECON-889k) and the initialization check used a
   stratified subset: 30 per non-NE family + 30 training opponents.  Primary ceilings (in-cloud full k and
   unconstrained, JAC-opp) use all 100 opponents per family.
4. **Training sanity set:** 100 opponents instead of 200.
5. **Drift paths:** λ ∈ {0, 0.25, 0.5, 0.75, 1} instead of 11 points.
6. **"Distance from the training cloud" metric.**  As pre-specified, in full-dimension whitened
   coordinates, it was uninformative: it flags 100% of points, training opponents included, because
   whitening amplifies ~100 near-zero-variance directions.  It was recomputed in the top-16 principal
   coordinates (§1.3; `outputs/manifold/location_top16.json`).
