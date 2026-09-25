# Generalization and practical safety on unseen opponent families (Leduc, evaluation only)

## 0. Pre-registration

Written and committed before any opponent family was generated or any method was evaluated on it.  Not
edited afterwards; deviations go in the deviations section.

### Methods (frozen; nothing is tuned on the new families)
Neural models are **seed 0**.  Every checkpoint and setting is the one selected on the *original*
validation set.

| group | method | source |
|---|---|---|
| classical | FIXED-NE | our suit-symmetric equilibrium blueprint (x_nash), deployed without modelling |
| classical | BANK | train-bank posterior over the 1 200 training opponents (exact likelihood) |
| classical | TAB-EM | tabular EM, uniform Dirichlet prior (α = 1, 200 iterations), as in V1 |
| neural | DEC-889k | `runs/decision_s0` |
| neural | RECON-889k | `runs_v3/rec889k_s0` |
| neural | RECON-JAC-global | `runs_jacopp/a0_s0` (same selection rule as JAC-opp) |
| neural | JAC-opp | `runs_jacopp/a3_s0` |
| hybrid | PRIOR-EM (RECON-131k) | EM with per-history Dirichlet prior κ·q̂, q̂ from `runs/recon_s0`, κ = 3 |
| hybrid | PRIOR-EM (JAC-opp) | the same with q̂ from `runs_jacopp/a3_s0`, κ = 3 |

κ = 3 for both hybrids, as specified.  For RECON-131k seed 0 this is the original validation choice at every
N.  For the JAC-opp prior, the original validation chose κ = 10 at N = 10 and 20 and 3 elsewhere; κ = 3 is
used at every N, as specified.

Every modelling method maps its estimate ĝ through the same exact ε-safe LP (HiGHS) at ε = 0.10.  The
deployed behavioural strategy is audited with OpenSpiel's exact best response.  FIXED-NE deploys x_nash
directly and is audited once.

### Opponent families
All policies are opponent (player 1) behavioural policies.  Every generated policy is suit-symmetrized
(group average in realization space; this preserves Nash-ness) and stored at the rank level (144 × 3).
Seeds are fixed and recorded per opponent.

* **ID-REF (100):** 25 per training family, drawn from the original test set with a fixed RNG
  (`default_rng([2026, 1])`).  The histories are streams 0–1 of the existing test datasets, which the same
  simulator produced.
* **NEAR (100)** — the training generators with parameters outside every training range, 25 per family:
  - **NASH-LOGIT:** τ ~ logU[2.5, 6] (training [0.15, 2]); per-infoset noise scale ~ U[1.2, 2.0] (training
    [0.2, 1.0]).
  - **NASH-MIX:** η ~ U[0.9, 1.0] (training [0.05, 0.85]); random component per infoset from Dirichlet(0.25)
    (training Dirichlet(1)).
  - **STRUCTURED:** all 8 traits with |t| ~ U[2.5, 4] and random sign (training N(0, 1)).  The final logits
    are multiplied by β ~ U[1.5, 2.5] (training β = 1).
  - **DIRICHLET:** α ~ logU[0.04, 0.15] for 13 opponents and α ~ logU[8, 30] for 12 (training logU[0.3, 3]).
* **FAR-ARCH (100)** — rule-based archetypes, 25 each, with randomized parameters.
  - **Hand strength** s is the training feature: pair = 1; otherwise J −0.6, Q 0, K +0.6.
  - **Per infoset,** p_fold applies only when facing a raise and p_raise only where raising is legal; the
    rest of the probability goes to call/check.  σ is the logistic function, and k ~ U[6, 15] is its slope.
  - **Each archetype** is then mixed as 0.95·q + 0.05·uniform over legal actions.

  | archetype | p_fold (facing a raise) | p_raise |
  |---|---|---|
  | ROCK | σ(k (τ_f − s)), τ_f ~ U[0.2, 0.8] | σ(k (s − τ_r)), τ_r ~ U[0.7, 0.95] |
  | CALLING STATION | f ~ U[0, 0.08] | r ~ U[0, 0.08], plus U[0, 0.3] when paired |
  | MANIAC | f ~ U[0, 0.08] | r ~ U[0.6, 0.95], plus 0.05·(s + 0.6) |
  | TAG (tight-aggressive) | σ(k (τ_f − s)), τ_f ~ U[−0.3, 0.3] | σ(k (s − τ_r)) with τ_r ~ U[0.2, 0.7]; when not facing a raise, an extra bluff raise b ~ U[0.05, 0.25] |

* **FAR-CFR (100)** — opponents mid-learning: the player-1 average strategy of OpenSpiel solvers on
  leduc_poker after few iterations.
  - 20 × CFR and 20 × CFR+, with iteration counts 1–50 (distinct per variant).
  - 30 × external-sampling MCCFR and 30 × outcome-sampling MCCFR, with iterations ~ U{1..50}·10 for outcome
    sampling (whose single iterations touch one trajectory) and U{1..50} for external sampling, and seeds
    1–30.
  - Infosets an average policy never reached are uniform.
* **FAR-EXPL (100)** — exploiters: a logit-softened (quantal) best response of player 1 to a random fixed
  rank-symmetric player-0 strategy.
  - The player-0 strategy is Dirichlet(α) per rank infoset, α ∈ {0.5, 1, 2}.
  - The response is computed bottom-up over player 1's sequences with q(a|I) ∝ exp(Q̄(I,a)/T).  Here Q̄ is the
    *conditional* expected value (chips) of action a under the softened continuation, and T ~ logU[0.1, 2]
    chips.
* **NE (5 opponents × 20 streams = 100 histories)** — opponent equilibria, each verified to be exploitable
  by at most 1e−6 (1e−3 for the CFR+ solution):
  - **1 × CFR+** after 2 000 iterations.
  - **4 × extreme points of the opponent's Nash set:** ε = 0 opponent-side LPs maximizing c·y over
    player 1's Nash set.  One direction is the value against the uniform learner (our blueprint1
    selection rule) and three are random Gaussian directions (seeds 1–3), suit-symmetrized.

New histories use the project's simulator, with the learner playing its blueprint while data is collected
(as in all datasets): stream seed 7, split ids 20–24 (disjoint from train/val/test), 500 hands.

### Protocol
* **Grid:** N ∈ {5, 10, 20, 50, 100, 200, 500}; 2 streams per opponent (20 for NE); ε = 0.10 only.
* **Recorded per point:** u(x, q), regret V_ε(g) − u, and Expl(x) (OpenSpiel audit).
* **Fraction of attainable safe gain,** per opponent (streams averaged):
  (u − V_0) / (V_ε − V_0), with V_0 = the ε = 0 LP value (the best of our equilibria against this
  opponent).
  - **Pooled** as a ratio of means over opponents, with a 95% bootstrap CI (2 000 resamples); the median
    per-opponent fraction is also reported.
  - **Excluded:** opponents with V_ε − V_0 < 0.01 chips (counts reported); they stay in the raw-regret
    plots.
* **NE family:** loss = v* − u(x, y*), which must lie in [0, ε] (up to the CFR+ opponent's exploitability).
* **Harm rate:** fraction of (opponent, N) points (streams averaged) where u(method) < u(FIXED-NE) − 1e−9.

### Expectations (operational tests)
"Fraction" is the pooled ratio-of-means fraction; FAR = {FAR-ARCH, FAR-CFR, FAR-EXPL}.

* **E1 — bank posterior degrades most on FAR, especially at large N.**
  - Drop = fraction(ID-REF) − mean fraction over FAR families.
  - E1 holds if BANK's drop at N = 500 is the largest among the 8 modelling methods, and its drop at N = 500
    exceeds its drop at N = 20.
* **E2 — neural models' fraction drops on NEAR and more on FAR; their large-N floors rise.**
  - E2 holds for a neural model if fraction(ID-REF) > fraction(NEAR) > mean FAR fraction at both N = 20 and
    N = 500, and its raw regret at N = 500 is higher on every FAR family than on ID-REF.
  - Reported per model; E2 holds overall if it holds for at least 3 of the 4 neural models.
* **E3 — the learned-prior EM degrades gracefully and approaches tabular EM at large N on every family.**
  - E3 holds for a hybrid if, on every family including NE, |regret(hybrid) − regret(TAB-EM)| at N = 500 is
    < 0.005 chips or its CI includes 0, and |difference| at N = 500 ≤ |difference| at N = 20.
* **E4 — on FAR families the learned prior may HURT at small N.**
  - E4 holds if, for at least one hybrid and one FAR family, hybrid − TAB-EM regret is > 0 with the CI
    excluding 0 at some N ≤ 20.
  - Reported per family: the smallest N from which the difference stays ≤ 0 (the crossing point), or
    "never".
* **E5 — against NE opponents every method loses at most ε, and on average far less.**
  - E5 holds if every (method, history, N) loss lies in [−δ, ε + 1e−7], where δ is the CFR+ opponent's
    exploitability (0 for the LP opponents).
  - It also requires the mean loss per method and N to be ≤ ε/2.

### Guardrails
* Exact audit of every deployed strategy (Expl ≤ ε + 1e−7; 0 violations required; LP failures reported).
* No tuning on the new families.
* Opponent counts, exclusions and wall-clock per stage are reported.
* If time runs short: fewer opponents per family or fewer streams, never whole families or methods.
