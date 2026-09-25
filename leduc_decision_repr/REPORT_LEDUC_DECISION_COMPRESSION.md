# Decision-sufficient compression of opponents (Leduc; no histories, no inference)

## Summary and verdict

**Yes: ε-safe decisions compress far beyond behaviour.**  A code needs roughly 8–16× fewer latent dimensions
to keep the same safe value when it is trained on decisions instead of behaviour.  The pre-registered primary
test H1 still misses, by 0.011, because of its absolute bar (≥ 0.90 at d ≤ 8).

**The oracle partitions (no learning) show the effect most cleanly.**
* **Four decision cells beat any behavioural partition.**
  - Four cells (2 bits), each deploying one exact safe response, keep **0.754** of the attainable safe value on
    300 test opponents (ε = 0.10).
  - The best behaviour-optimal partition keeps **0.749–0.752 at any size up to 512 cells** (9 bits).
  - The behaviourally nearest training opponent's response keeps 0.745; the nearest in g, 0.755.
* **The decision cells keep little behaviour.**  Their behavioural fraction is 0.12 at K = 4 and 0.30 at K = 256,
  against 0.34 and 0.61 for behaviour cells.
* **Behaviour does not determine the safe response.**  Behaviourally similar opponents often need different
  responses.
* **The best 1,200-response bank keeps 0.922.**  Decision partitions reach 0.90 at K = 256; behavioural
  partitions plateau at 0.75.

**Learned codes: the policy autoencoder q → z → q̂.**  One network, with only the loss varied, for 3 seeds at each
d.  Test decision fraction, ε = 0.10:

| d | ordinary RECON | behaviour-optimal OBS-RECON (post-hoc) | JAC-OPP | exact G-MSE | **REGRET** (response bank) |
|---|---|---|---|---|---|
| 2 | 0.741 | 0.696 | 0.772 | 0.754 | **0.800** |
| 4 | 0.765 | 0.731 | 0.805 | 0.793 | **0.845** |
| 8 | 0.791 | 0.772 | 0.853 | 0.850 | **0.889** |
| 16 | 0.832 | 0.796 | 0.883 | 0.872 | **0.909** |
| 64 | 0.864 | 0.814 | 0.896 | 0.891 | **0.913** |

* **REGRET keeps the most decision value at every d.**  Seed ranges are ≤ 0.015.
* **Compression factors.**
  - Ordinary reconstruction needs d = 16 to match REGRET at d = 2, and d = 32 to match it at d = 4.  Even at
    d = 64 it does not reach REGRET at d = 8.
  - The behaviour-optimal autoencoder needs d = 32 to match REGRET at d = 2.  It never reaches REGRET at d = 4.
* **The trade-off at d = 8:**
  - REGRET keeps **89% of safe value with 66% of observable behaviour**;
  - the behaviour-optimal code keeps 82% of behaviour but only 77% of safe value;
  - at d = 64, the behaviour-optimal code keeps 92% of behaviour and still only 81% of safe value.
* **REGRET discards the policy itself.**  Its per-infoset policy fraction at d ≤ 4 is *negative*: its decoded
  policy is further from the truth than the population-mean policy, yet its safe decisions are the best.

**The cost side is smaller than expected.**
* **Unseen families.**  The advantage holds on the four unseen opponent families.  At d = 8, REGRET − RECON is
  +0.103 OOD against +0.098 in distribution.
* **Other safety budgets.**  It also holds at ε = 0.05 (+0.090) and ε = 0.20 (+0.086), although REGRET's bank was
  built at ε = 0.10.

**The Jacobian weighting is a good surrogate, not the source of the gain.**
* JAC-OPP tracks the exact g-loss within 0.018 at every d, and is slightly better.
* Both sit well below REGRET.  The decision objective, not the Jacobian weighting, produces the compression.

**Decision-equivalence structure (Part C): real in the oracle, only partly learned.**
* **In the oracle.**  Decision cells put 19% of "decision-equivalent but behaviourally far" test pairs in the same
  cell, against 2% of "behaviourally close but decision-different" pairs.  Behaviour cells do the reverse: 1% vs
  26%.
* **In the learned codes.**  Every learned latent is still organised mainly by behaviour: separation ratio 1.8–7.7
  > 1.  REGRET's geometry moves most towards decisions: the lowest separation ratio, and a positive partial
  correlation with cross-regret given behaviour (+0.09 at d = 8 vs −0.14 for RECON).

**Pure regret training collapses (λ = 0: 0.46–0.50 on val),** as every earlier regret-trained model did.  A small
g-MSE anchor (λ = 0.1) fixes it.  A post-hoc check shows the collapse is mostly a training problem, not only a
train/deploy mismatch.

**Pre-registered expectations:**
* **Hold:** A1, A3, H3, H4(b), C1.
* **Fail:**
  - A2: decision cells need K = 256 for 0.90, not ≤ 32;
  - A4 and H4(a): the advantage does *not* shrink out of distribution;
  - H1: 0.889 < 0.90;
  - H2 against uniform RECON (the decision code keeps slightly *more* observable behaviour, 0.66 vs 0.62);
  - F: the objective clearly matters;
  - C2.

**Safety:** 140 427 deployed strategies audited, max Expl − ε = 2.5e−10, **0 violations**, 0 LP failures.

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

## 1. Part A — oracle partitions (`dc_oracle.py`, `fig1_oracle_partitions.png`)

**Decision fraction** on 300 test opponents (ε = 0.10); 95% CIs of about ±0.02–0.03 are in `oracle.json`:

| K (bits) | 1 (0) | 2 (1) | 4 (2) | 8 (3) | 16 (4) | 32 (5) | 64 (6) | 128 (7) | 256 (8) | 512 (9) | 1-NN of 1 200 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **DEC** | 0.599 | 0.705 | **0.754** | 0.810 | 0.841 | 0.866 | 0.888 | 0.896 | 0.904 | 0.916 | 0.922 |
| BEH | 0.599 | 0.611 | 0.650 | 0.703 | 0.720 | 0.752 | 0.749 | 0.745 | 0.747 | 0.749 | 0.745 |
| GVAR | 0.599 | 0.614 | 0.651 | 0.681 | 0.723 | 0.718 | 0.736 | 0.748 | 0.756 | 0.762 | 0.755 |

**Behavioural fraction** of the same cells (test):

| K | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1-NN |
|---|---|---|---|---|---|---|---|---|---|---|
| DEC | 0.03 | 0.12 | 0.24 | 0.32 | 0.36 | 0.36 | 0.35 | 0.30 | 0.18 | 0.04 |
| BEH | 0.18 | 0.34 | 0.46 | 0.53 | 0.56 | 0.58 | 0.60 | 0.61 | 0.61 | 0.60 |
| GVAR | 0.08 | 0.33 | 0.41 | 0.45 | 0.50 | 0.54 | 0.53 | 0.54 | 0.52 | 0.49 |

**Findings:**
* **K = 1 is the common starting point.**  All three partitions give the single safe response to the population's
  mean g, which keeps 0.599.  Everything above that is what a code adds.
* **Decision cells climb steeply, behavioural cells plateau.**
  - Decision cells add 0.155 with 2 bits.
  - Behavioural and g-variance cells plateau at about 0.75, and do so by K ≈ 32.
  - More behavioural resolution does not buy decisions.  The 1-NN rows make the point at the extreme: the exact
    nearest training opponent in behaviour, or in g, gives a response worth 0.745 / 0.755.
* **Behavioural partitions overfit, decision partitions do not.**
  - BEH K = 256 scores 0.856 on train and 0.747 on test.
  - DEC K = 256 scores 0.947 on train and 0.904 on test.
  - Behavioural similarity within training opponents does not carry decision similarity to new ones.
* **Decision cells are behaviourally poor, and get poorer at large K.**  Their behavioural fraction peaks at 0.36
  (K = 32–64) and falls to 0.04 for 1-NN.
  - Past the peak, the cells group behaviourally unlike opponents that share a response.
  - The cell mixture is then a poor behavioural description.
* **Out of distribution.**  Mean over NEAR, FAR-ARCH, FAR-CFR and FAR-EXPL at ε = 0.10:
  - DEC: K = 4 0.721, K = 16 0.805, 1-NN 0.900;
  - BEH never exceeds 0.70;
  - at K = 16 the advantage is +0.126 OOD against +0.121 test.
* **ε transfer (K = 16, partition fixed, responses re-solved at each ε):**
  - DEC 0.793 / 0.841 / 0.810 at ε = 0.05 / 0.10 / 0.20;
  - BEH 0.700 / 0.720 / 0.730.

## 2. Part B — policy autoencoder (`dc_ae.py`, `fig2_autoencoder.png`, `fig3_tradeoff.png`, `fig4_cost_side.png`)

**Selection of REGRET's (β, λ) on val (d = 8, seed 0), decision fraction:**

| (β, λ) | val decision fraction |
|---|---|
| **(100, 0.1)** | **0.894 (selected)** |
| (1000, 0.1) | 0.881 |
| (100, 0) | 0.495 |
| (1000, 0) | 0.460 |

Pure regret collapses; see §2.4.

### 2.1 Test results (seed means over 3 seeds; ε = 0.10)

| arm | d = 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| **Decision fraction** | | | | | | |
| RECON | 0.741 | 0.765 | 0.791 | 0.832 | 0.849 | 0.864 |
| JAC-OPP | 0.772 | 0.805 | 0.853 | 0.883 | 0.894 | 0.896 |
| G-MSE | 0.754 | 0.793 | 0.850 | 0.872 | 0.888 | 0.891 |
| **REGRET** | **0.800** | **0.845** | **0.889** | **0.909** | **0.909** | **0.913** |
| *OBS-RECON (post-hoc)* | 0.696 | 0.731 | 0.772 | 0.796 | 0.814 | 0.814 |
| **Behavioural fraction (observable)** | | | | | | |
| RECON | 0.516 | 0.563 | 0.622 | 0.738 | 0.775 | 0.796 |
| JAC-OPP | 0.595 | 0.695 | 0.811 | 0.851 | 0.864 | 0.867 |
| G-MSE | 0.559 | 0.664 | 0.771 | 0.821 | 0.837 | 0.843 |
| REGRET | 0.394 | 0.544 | 0.659 | 0.712 | 0.718 | 0.719 |
| *OBS-RECON* | 0.612 | 0.728 | 0.824 | 0.876 | 0.906 | 0.915 |
| **Policy fraction (uniform per-infoset KL)** | | | | | | |
| RECON | 0.595 | 0.642 | 0.686 | 0.732 | 0.759 | 0.783 |
| JAC-OPP | 0.517 | 0.579 | 0.636 | 0.684 | 0.711 | 0.720 |
| G-MSE | 0.364 | 0.494 | 0.605 | 0.652 | 0.673 | 0.681 |
| REGRET | −0.678 | −0.101 | 0.214 | 0.302 | 0.325 | 0.332 |
| *OBS-RECON* | 0.341 | 0.390 | 0.450 | 0.503 | 0.514 | 0.519 |
| **g R²** | | | | | | |
| RECON | 0.574 | 0.631 | 0.688 | 0.801 | 0.827 | 0.843 |
| JAC-OPP | 0.638 | 0.727 | 0.816 | 0.863 | 0.882 | 0.887 |
| G-MSE | 0.661 | 0.765 | 0.839 | 0.880 | 0.900 | 0.906 |
| REGRET | 0.624 | 0.720 | 0.794 | 0.827 | 0.831 | 0.832 |
| *OBS-RECON* | 0.507 | 0.593 | 0.663 | 0.713 | 0.722 | 0.728 |

* **Uncertainty.**
  - 95% CIs at d = 8 (bootstrap over opponents): RECON [0.764, 0.819], JAC-OPP [0.831, 0.873], G-MSE [0.827, 0.869],
    REGRET [0.871, 0.905], OBS-RECON [0.746, 0.796].
  - Seed ranges are ≤ 0.015 everywhere.
* **Each objective keeps what it optimises.**
  - RECON keeps the most per-infoset policy.
  - OBS-RECON keeps the most observable behaviour.
  - G-MSE reconstructs g best.
  - REGRET keeps the most decision value, while reconstructing g *less* well than G-MSE.  At d = 64 its g R² is
    below even plain RECON's.
* **Matching dimensions** (the smallest d on the grid at which an arm reaches REGRET's value):

| REGRET at | RECON needs | OBS-RECON needs | G-MSE needs | JAC-OPP needs |
|---|---|---|---|---|
| d = 2 (0.800) | 16 | 32 | 8 | 4 |
| d = 4 (0.845) | 32 | never (≤ 64) | 8 | 8 |
| d = 8 (0.889) | never | never | 64 | 32 |

### 2.2 The cost side at d = 8 (`fig4_cost_side.png`)

| arm | ε = 0.05 | ε = 0.10 | ε = 0.20 | OOD mean | NEAR | FAR-ARCH | FAR-CFR | FAR-EXPL |
|---|---|---|---|---|---|---|---|---|
| RECON | 0.778 | 0.791 | 0.792 | 0.748 | 0.591 | 0.984 | 0.839 | 0.729 |
| JAC-OPP | 0.834 | 0.853 | 0.855 | 0.823 | 0.719 | 0.988 | 0.859 | 0.813 |
| G-MSE | 0.832 | 0.850 | 0.856 | 0.814 | 0.715 | 0.974 | 0.884 | 0.794 |
| **REGRET** | **0.868** | **0.889** | **0.878** | **0.851** | **0.760** | 0.981 | **0.912** | **0.841** |
| *OBS-RECON* | 0.762 | 0.772 | 0.759 | 0.730 | 0.695 | 0.892 | 0.821 | 0.657 |

* **Out of distribution.**  The decision advantage survives: REGRET − RECON is +0.103 OOD against +0.098 in
  distribution.  It is largest on NEAR (+0.169) and FAR-EXPL (+0.112), the two families where every learned model
  broke in the generalization study.
* **Across ε.**  It also survives changes of safety budget, shrinking only slightly away from the bank's ε
  (+0.090 at 0.05, +0.086 at 0.20).
* **Archetypes are easy for every code** (FAR-ARCH ≥ 0.97).

### 2.3 Why REGRET plateaus at 0.91
REGRET only learns to rank the 1 200 training responses.  The best of those for each test opponent (Part A's
1-NN under the decision rule) is worth 0.922, and REGRET reaches 0.913 at d = 64.  Its ceiling is therefore the
bank's coverage, not the latent size.  The other arms plateau lower (0.81–0.90); their g R² stays at or below
0.91, which suggests the fixed architecture and budget also limit them.

### 2.4 Pure regret and the bank post-hoc check (`dc_posthoc.py`)
The collapse of λ = 0 raised a question.  Was it a train/deploy mismatch?  The loss only constrains ĝ along the
1 200 bank directions, while deployment runs the full LP.
* **Deploying the bank argmax instead** gives 0.684 (β = 100) and 0.579 (β = 1000) on test.  That is better than
  its LP deployment, but far below the anchored run (0.857 bank, 0.889 LP).  So the collapse is mainly an
  optimisation failure of the soft bank regret, with the mismatch a secondary factor.
* **For every anchored model the full LP on ĝ beats the bank argmax,** for example REGRET d = 8: 0.889 vs 0.859.
  The learned codes carry more than "which training response to copy".
* **Safety:** the 840 distinct bank responses deployed in this check were audited, with 0 violations.

## 3. Part C — decision-equivalence structure (`fig5_equivalence.png`)

* **The two distances are related.**  Over the 44 850 test pairs, cross-regret D_dec and behavioural distance D_beh
  are strongly correlated (Spearman 0.72).
* **Pair classes.**  The 10th-percentile classes had fewer than 50 pairs, so the pre-registered 20th-percentile
  fallback applies:
  - 607 DEQ-BF pairs (decision-equivalent, behaviourally far);
  - 177 BC-DD pairs (behaviourally close, decision-different).

| model (d = 8) | ρ(latent, D_dec) | ρ(latent, D_beh) | partial ρ(latent, D_dec \| D_beh) | SR |
|---|---|---|---|---|
| RECON | 0.441 | 0.702 | −0.136 | 3.06 |
| JAC-OPP | 0.624 | 0.856 | +0.016 | 2.87 |
| G-MSE | 0.613 | 0.826 | +0.040 | 2.91 |
| **REGRET** | 0.611 | 0.796 | **+0.087** | **2.26** |
| *OBS-RECON* | 0.598 | 0.877 | −0.108 | 3.03 |

* **Every latent is organised mainly by behaviour** (SR > 1 at every d, for every arm).  Decision-aware objectives
  move the geometry towards decisions:
  - the partial correlation rises to +0.19 for REGRET at d = 64, against +0.05 for RECON;
  - REGRET has the lowest SR at every d ≥ 4.
* **The oracle shows the structure that exists.**

  | K = 16 partition | DEQ-BF pairs sharing a cell | BC-DD pairs sharing a cell |
  |---|---|---|
  | decision | 19% | 2.3% |
  | behaviour | 0.8% | 26% |
  | g-variance | 4.3% | 29% |

  - At K = 64 the decision partition never puts a BC-DD pair in the same cell.
* **The learned continuous codes capture only a small part of that grouping.**  They are decision-*sufficient*
  (they deploy well) without being decision-*organised* (their distances still mostly reflect behaviour).

## 4. Expectations

| test | result | holds? |
|---|---|---|
| **A1:** DEC > BEH, GVAR at every K ≥ 4 | margins +0.10 … +0.15 | **yes** |
| **A2:** K₉₀(DEC) ≤ 32 and K₉₀(BEH) ≥ 4× | K₉₀(DEC) = 256; BEH never | **no** |
| **A3:** at K₉₀(DEC), BEH − DEC behavioural fraction ≥ 0.10 | 0.61 − 0.30 = 0.31 | **yes** |
| **A4:** DEC − BEH smaller OOD than test (K = 16) | +0.126 vs +0.121 | **no** |
| **H1:** at some d ≤ 8, best decision arm ≥ 0.90, RECON ≤ 0.80, RECON needs ≥ 4×d | d = 8: REGRET 0.889 (< 0.90); RECON 0.791; RECON never matches | **no** (by 0.011) |
| **H2:** decision arm's behavioural fraction ≥ 0.10 below RECON's | 0.659 vs 0.622 (vs OBS-RECON, post-hoc: 0.659 vs 0.824) | **no** |
| **F:** all arms within 0.03 at every d | spread 0.05–0.10 | **no** (the objective matters) |
| **H3:** \|JAC-OPP − G-MSE\| ≤ 0.02 at every d ≥ 4 | +0.003 … +0.012 | **yes** |
| **H4(a):** advantage smaller OOD | +0.103 vs +0.098 | **no** |
| **H4(b):** REGRET advantage smaller at ε = 0.05, 0.20 | +0.090, +0.086 vs +0.098 | **yes** |
| **C1:** REGRET and G-MSE have larger ρ_dec − ρ_beh than RECON (d = 8) | −0.18, −0.21 vs −0.26 | **yes** |
| **C2:** SR < 1 for REGRET and > 1 for RECON (d = 8) | 2.26 and 3.06 | **no** |

The prior guess held on A1, C1 and the G-MSE part of H3.  H1 was the coin flip and landed just short.

## 5. What this means for the project's novelty

1. **Decision-sufficient compression is real and large for certified safe exploitation.**
   - In the oracle: 2 bits of decision information beat 9 bits of behavioural information.
   - With learning: a regret-trained code at d = 2–4 matches reconstruction codes 8× larger and
     behaviour-optimal codes 16× larger, while giving up most of the policy.
   - It survives unseen opponent families and other safety budgets.
   - **This is the claim to build on.**  In the earlier history-based studies, inference effects hid it.
2. **The Jacobian weighting is a good, cheap surrogate for the exact g-loss (within 0.02), but it is not the
   headline.**  The decision objective is: regret over a bank of certified safe responses, with a g anchor.  A
   natural framing is the one discussed earlier: value-equivalent opponent models for certified ε-safe
   exploitation.
3. **What is still missing is decision-*organised* geometry.**  The learned latents keep behavioural
   neighbourhoods even when they are decision-sufficient.  Matters for future work that relies on latent distances
   (drift tracking, curricula, clustering) may need an explicit metric-learning term (for example, a contrastive
   term on cross-regret).
4. **The next step, a regret-driven curriculum, is now concretely motivated.**
   - REGRET's plateau is its bank's coverage (0.913 vs the bank's 0.922).
   - Its weakest families are NEAR and FAR-EXPL.
   - A teacher that adds opponents, and their certified responses, where the code loses value targets exactly that
     ceiling.

## 6. Audits and wall-clock

**Audits.**  Every deployed strategy is an exact LP solution, audited with OpenSpiel's exact best response:
* Part A: 8 487 distinct cell responses;
* Part B: 131 100 deployments (90 models × 1 450, plus the val-only selection runs);
* bank check: 840.

Total **140 427**:
* max Expl − ε = 2.5e−10;
* **0 violations**;
* 0 LP failures.

**Wall-clock (4 CPUs):**

| stage | wall-clock |
|---|---|
| Part A partitions + evaluation + audit | 6.8 min |
| Part B data prep (20 000 generated opponents, Jacobian weights) | 1.7 min |
| REGRET (β, λ) selection | 20.6 min |
| Training, 72 runs | 71.5 min |
| Evaluation, 72 models | 30.6 min |
| OBS-RECON post-hoc (18 runs, train + eval) | 20.1 min |
| Bank check | ~2 min |
| Analysis + figures | ~2 min |
| **Total** | **≈ 2.6 h** |

## 7. Deviations and notes

1. **OBS-RECON is post-hoc.**
   - Uniform-weight RECON (the pre-registered "ordinary reconstruction") turned out to be a poor *observable*
     behaviour model.  It spends capacity on rarely reached infosets.
   - That made H2's behavioural comparison uninformative, so a behaviour-optimal autoencoder was added, trained on
     the exact KL between observable hand distributions.  It uses the same network, budget and seeds.
   - It is reported separately and does not enter any pre-registered verdict.
2. **The bank-argmax deployment check is post-hoc** (§2.4).
3. **Part C used the pre-registered 20th-percentile fallback** (the 10th-percentile classes had fewer than 50
   pairs).
4. **REGRET d = 8, seed 0 in the main grid is the selected selection run.**  It has an identical configuration and
   was copied rather than retrained.
5. **A scheduling bug delayed the post-hoc stage and did not affect any result.**  My wait loop's process pattern
   matched its own command line.
6. **Absolute fractions are limited by the fixed autoencoder (256-unit MLPs, 6 000 steps) and, for REGRET, by the
   bank.**  The comparisons are at matched architecture, data and budget, so the absolute levels (for example, the
   0.90 bar in H1) should not be read as fundamental limits.
7. **Scope.**
   - The true opponent policy is given; there is no inference from hands.
   - One game (Leduc).
   - REGRET's bank is built at ε = 0.10 from the 1 200 training opponents.
   - This measures what a code *can* keep, not how well it can be inferred from play.
