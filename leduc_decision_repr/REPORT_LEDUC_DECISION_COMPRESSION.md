# Decision-sufficient compression of opponents (Leduc; no histories, no inference)

*(Results are appended below the pre-registration after the run.)*

## 0. Pre-registration

Written and committed before any partition was built, any autoencoder was trained, or any quantity below was
computed.  It is not edited afterwards; deviations go in the deviations section.

### Question
**Can a much smaller code preserve ε-safe decisions than is needed to preserve behaviour?**
* Everything here maps the opponent's *true* policy q to a code and back, so history and inference play no
  role.
* V3's oracle rank analysis found that variance-ranked directions of g are poor for decisions: at ε = 0.10, the
  29 directions holding 90% of g's variance keep only about 63% of the safe value.
* That result says nothing about the *smallest* decision-sufficient code.  This study measures it.

### Common definitions
* **Decision value.**  Deploy x = exact ε-safe LP on the code's decision vector (Part A: the cell's
  response; Part B: ĝ = g(q̂)), and score it against the opponent's true g.
  - Fraction = (u − V_0) / (V_ε − V_0), pooled as a ratio of means over opponents.
  - V_0 and V_ε are the population oracle values (families file for OOD).
  - Primary ε = 0.10; transfer ε ∈ {0.05, 0.20} on test.
* **Behavioural information (primary: observable).**  p_q is the exact distribution over the 415 reachable
  observation types of a hand, played by the learner's blueprint against q.
  - Behavioural fraction = 1 − Σ_i KL(p_i ‖ p̂_i) / Σ_i KL(p_i ‖ p̄), where p̂_i is the code's reconstruction and
    p̄ is the mean hand distribution of the 1 200 training opponents.
  - 1 = behaviour fully preserved; 0 = no better than the population mixture.
  - Secondary "policy fraction": the same with the uniform per-infoset KL(q_I ‖ q̂_I) in place of the
    hand-distribution KL.
* **Opponent sets.**
  - train: 1 200 population training opponents;
  - val: 150;
  - test: 300 (in-distribution);
  - OOD: the generalization study's NEAR, FAR-ARCH, FAR-CFR and FAR-EXPL families (100 each; ε = 0.10 values
    from `outputs/gen/families.npz`).
* **Safety.**  Every deployed test / OOD strategy is an exact LP solution, audited with OpenSpiel's exact best
  response (Expl ≤ ε + 1e−7).  In Part A each distinct cell response is audited once per ε.

### Part A — oracle compressibility (no learning; `dc_oracle.py`)
**K-cell partitions of opponent space, built on the 1 200 training opponents,**
for K ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256, 512} and "all" (1 200 singletons, i.e. 1-nearest-neighbour).
Each partition has an *assignment rule* that is a function of q.
* **BEH (behaviour-optimal):** k-means with the KL divergence on p_q (Bregman k-means).
  - Centroid = mean hand distribution; assignment = argmin_k KL(p_q ‖ c_k).
  - k-means++ initialisation, 3 restarts, ≤ 50 iterations.
* **GVAR (g-variance):** Euclidean k-means on g, with the same initialisation and restarts; assignment =
  nearest centroid.
* **DEC (decision-optimal): a regret Lloyd algorithm at ε = 0.10.**
  - Cell response x_k = ε-safe LP on the cell's mean g.
  - Assignment = argmax_k x_kᵀ g(q).
  - Iterate assignment and update ≤ 20 times, from two initialisations (the GVAR and BEH partitions); keep
    the one with higher training value.
  - An empty cell is reseeded with the oracle response of the worst-served training opponent.
* **Deployment at ε.**  For every partition and ε, the deployed response of cell k is the ε-safe LP on the
  mean g of its training members.  This is the value-optimal single response for the cell, because value is
  linear in g.
* **Behavioural reconstruction** of a cell is the mean p of its training members.

**Reported per partition, K and opponent set:** decision fraction, behavioural fraction, bits = log2 K.

### Part B — policy autoencoder (`dc_ae.py`)
* **Training data.**
  - The 1 200 training opponents plus 20 000 fresh draws from the same four training families: the
    population generator, seed 2024, indices k = 10 000–14 999 per family.  The population itself used
    k < 413.
  - val = the 150 validation opponents.
* **Model (identical across arms).**
  - Encoder: q (432 entries, illegal zeros) → MLP 432→256→256→d (GELU).
  - Decoder: d → 256 → 256 → 432 logits, masked softmax per infoset → q̂.
  - d ∈ {2, 4, 8, 16, 32, 64}.
* **Arms (the loss is the only difference):**
  - **RECON:** mean over the 144 infosets of KL(q_I ‖ q̂_I), uniform weights ("ordinary reconstruction").
  - **JAC-OPP:** the same, weighted per infoset by the opponent's projected Jacobian norm w_I(q) = ‖J_I P_I‖²
    (the JAC-opp weights, computed at the true q, normalised to mean 1 per opponent).
  - **G-MSE:** ‖g(q̂) − g(q)‖², with the exact differentiable g.  Opp-JAC is its local quadratic approximation.
  - **REGRET (response-bank regret):**
    - B = the exact ε = 0.10 safe responses of the 1 200 training opponents; every candidate is exactly ε-safe.
    - Loss L = max_m B_mᵀg − Σ_m softmax(β B ĝ)_m B_mᵀg, plus λ · G-MSE, with ĝ = g(q̂).
    - β ∈ {100, 1000} and λ ∈ {0, 0.1} are selected on *val* decision fraction (ε = 0.10) at d = 8, seed 0.
      The selected (β, λ) is used at every d and seed.
* **Training.**  AdamW (lr 1e−3, wd 1e−4), batch 256, 6 000 steps, 200 warm-up steps then cosine decay.  The
  final checkpoint is used (no test-based or early stopping).  Seeds {0, 1, 2}.
* **Evaluation per model:**
  - test at ε ∈ {0.05, 0.10, 0.20};
  - val and OOD at ε = 0.10;
  - metrics: decision fraction, behavioural fraction, policy fraction, g R².
  - Headline numbers are seed means, with 95% bootstrap CIs over opponents (2 000 resamples) of the seed-averaged
    per-opponent values.

### Part C — decision-equivalence structure (folded into B; test opponents, ε = 0.10)
* **Distances for all 44 850 test pairs (i, j):**
  - D_beh = Jensen–Shannon divergence of p_i and p_j;
  - D_dec = ½[(V_ε(i) − x*_jᵀg_i) + (V_ε(j) − x*_iᵀg_j)], the symmetric cross-regret of swapping the exact safe
    responses;
  - latent distance ‖z_i − z_j‖, divided by the model's median pairwise distance.
* **Pair classes:**
  - DEQ-BF (decision-equivalent, behaviourally far): D_dec ≤ 10th percentile and D_beh ≥ median;
  - BC-DD (behaviourally close, decision-different): D_beh ≤ 10th percentile and D_dec ≥ median.
  - If either class has fewer than 50 pairs, the 10th percentile becomes the 20th.
* **Per model:**
  - Spearman ρ(latent, D_dec) and ρ(latent, D_beh), and the partial Spearman of latent vs D_dec given D_beh;
  - separation ratio SR = mean latent distance of DEQ-BF pairs / mean latent distance of BC-DD pairs.
  - SR < 1 means the code groups opponents by decision; SR > 1 means it groups them by behaviour.

### Expectations
**Part A:**
* **A1:** DEC's test decision fraction exceeds BEH's and GVAR's at every K ≥ 4 (point estimates, ε = 0.10).
* **A2:** K₉₀(DEC) ≤ 32 (the smallest K on the grid with test fraction ≥ 0.90), and K₉₀(BEH) ≥ 4 × K₉₀(DEC),
  or BEH never reaches 0.90 by K = 512.
* **A3:** at K₉₀(DEC), DEC's test behavioural fraction is at least 0.10 below BEH's at the same K.
* **A4:** DEC's decision advantage over BEH at K = 16 is smaller on OOD than on test.

**Part B:**
* **H1 (decision-sufficient compression, primary).**  All three must hold at some d ≤ 8:
  - the best decision-aware arm (G-MSE, REGRET or JAC-OPP) keeps ≥ 0.90 of the test decision value (ε = 0.10,
    seed mean);
  - RECON at the same d keeps ≤ 0.80;
  - RECON needs ≥ 4× that d to reach the decision-aware arm's value.
* **H2 (it discards behaviour):** at that d, the decision-aware arm's behavioural fraction is at least 0.10
  below RECON's.
* **F (falsification of "the objective matters"):** all four arms' decision fractions lie within 0.03 of each
  other at every d.
* **H3 (JAC as a surrogate):** JAC-OPP is within 0.02 of G-MSE's decision fraction at every d ≥ 4.
* **H4 (the cost side).**  At d = 8, the decision-aware advantage (best decision-aware arm − RECON) is:
  - (a) smaller on OOD than on test;
  - (b) for REGRET (trained at ε = 0.10), smaller at ε = 0.05 and at ε = 0.20 than at ε = 0.10.

**Part C:**
* **C1:** at d = 8, REGRET and G-MSE have a larger ρ(latent, D_dec) − ρ(latent, D_beh) than RECON.
* **C2:** at d = 8, SR < 1 for REGRET and SR > 1 for RECON.

**Prior guess (not a test):**
* A1, C1 and the G-MSE part of H3 are likely.
* H1 as stated (≥ 0.90 at d ≤ 8 with RECON ≤ 0.80) is roughly a coin flip.
* The REGRET arm is the least predictable, because earlier regret-trained models collapsed.

### Guardrails
* Nothing is selected on test: the partitions are built on train, and REGRET's (β, λ) is chosen on val.
* The oracle quantities use exact LPs.  Every deployed test / OOD strategy is audited, and 0 violations are
  required.
* Wall-clock per stage is reported.
* If running long, drop seeds 1–2 at d ∈ {2, 64} first, never whole arms.
