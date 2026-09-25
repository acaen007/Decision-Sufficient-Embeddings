# Latent-manifold coverage diagnostic (Leduc, no training)

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
