# Generalization and practical safety on unseen opponent families (Leduc, evaluation only)

## Summary

**The hybrid holds up.**  The learned-prior EMs are the best or tied-best method on every family at N = 500:
fraction of attainable safe gain 0.87–0.94.
* They are never significantly worse than tabular EM at any N on any family.
* On the "far" families they help most at small N (FAR-ARCH −0.12 chips of regret at N = 20).
* Against Nash opponents they lose at most 0.079 chips, and ≤ 0.004 on average from N = 20 on.

**Tabular EM generalizes too, but slowly.**  It is insensitive to family (0.80–0.90 at N = 500) but slow at
small N (FAR-ARCH: 0.61 at N = 20).  It also plays worst against Nash opponents, losing 0.025–0.040 chips on
average at every N while all other modelling methods lose ≤ 0.006 from N = 20 on.

**The bank posterior and all four neural models break — but not where the families' names say they
would.**
* **Where they break:** on NEAR (the training generators pushed past their parameter ranges) and FAR-EXPL
  (softened exploiters).  At N = 500 they plateau at 0.29–0.41 chips of regret (fraction 0.60–0.77),
  against 0.09 and 0.19 for tabular EM and 0.09–0.12 for the hybrids.
* **Where they don't:** on rule-based archetypes (FAR-ARCH) and mid-learning CFR/MCCFR opponents (FAR-CFR)
  they do as well as or better than in-distribution (fraction up to 0.93).
* **What predicts it:** distance from the training bank in decision space, not the generator.  NEAR and
  FAR-EXPL sit 2–3× further from their nearest training opponent (median ‖g − g_train‖ 1.16 and 1.72) than
  ID-REF, FAR-ARCH or FAR-CFR (0.47–0.53).  Opponent by opponent, that distance predicts how far a learned
  model falls behind tabular EM at N = 500 (Spearman 0.51–0.66).  The hybrids do not depend on it.

**Practical safety holds everywhere.**
* **Audit:** all 61 601 deployed strategies passed (max Expl − ε = 1.5e−10, 0 violations, 0 LP failures).
  Every modelling strategy uses exactly the whole ε budget.
* **Against Nash opponents,** every method loses less than ε.
* **Harm vs FIXED-NE** (earning less than our own equilibrium): 4–15% of points overall.  It is almost
  entirely confined to near-equilibrium full-width CFR/CFR+ opponents (headroom 0.03–0.09 chips), with
  mean shortfalls of 0.01–0.06 chips.

**Pre-registered expectations:** E5 holds; E1–E4 fail.
* E1 and E2 fail because the FAR-ARCH and FAR-CFR families turned out not to be far.
* E3 fails in the favourable direction: the hybrid stays *better* than tabular EM at N = 500 instead of
  converging to it.
* E4 fails because the learned prior never hurts significantly.

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

## 1. What was built (`gen_families.py`, 2.4 min)

| family | opponents | histories | median headroom V_ε − V_0 | excluded (< 0.01) | median distance to training bank ‖g − g_nn‖ | median opponent exploitability |
|---|---|---|---|---|---|---|
| ID-REF | 100 (25 per training family) | 200 | 0.46 | 0 | 0.52 | 1.89 |
| NEAR | 100 (25 per generator) | 200 | 0.85 | 0 | 1.16 | 3.42 |
| FAR-ARCH | 100 (25 ROCK, 25 CALLING STATION, 25 MANIAC, 25 TAG) | 200 | 0.42 | 0 | 0.47 | 1.26 |
| FAR-CFR | 100 (20 CFR, 20 CFR+, 30 ES-MCCFR, 30 OS-MCCFR) | 200 | 0.48 | 0 | 0.53 | 2.17 |
| FAR-EXPL | 100 (quantal responses) | 200 | 1.26 | 0 | 1.72 | 2.88 |
| NE | 5 (CFR+ and 4 LP extreme points) | 100 (20 streams each) | 0 (by definition) | 5 (fraction undefined) | 0.32 | ≤ 4.7e−5 |

Checks:
* Every policy is a valid rank policy (legal support, rows sum to 1).
* Archetypes give every legal action ≥ 0.05/k probability (k = number of legal actions).
* ID-REF's recomputed V_0 and V_ε match the stored oracle values to 1e−8.
* The quantal-response code reproduces the exact best-response payoff at T → 0 (3.420411 vs 3.420411).
* **NE opponents:**
  - The four LP extreme points are exploitable by at most 1.3e−14 and the CFR+ solution by 4.7e−5.
  - They are distinct: pairwise L1 distances between realization plans are in `outputs/gen/families_meta.json`.
  - For every NE opponent, V_ε = V_0 = v*, so there is no headroom.

## 2. Figures (`outputs/gen/`)

**Encoding, used in every figure:**
* Classical methods are cool (FIXED-NE navy dashed, TAB-EM blue, BANK violet).
* Neural models are warm (DEC-889k magenta, RECON-889k yellow, RECON-JAC-global red, JAC-opp orange).
* Hybrids are green/aqua with thicker lines.
* Every method also has its own marker.

**Axes and variants.**  The N axis is logarithmic.  Bands are 95% bootstrap CIs over opponents (over
histories for NE).  Figures 1–2 show the highlighted subset; the `_full` versions show all nine methods.

* **Fig 1 — `fig1_fraction.png` / `fig1_fraction_full.png`.**  Fraction of attainable safe gain vs N, one
  panel per family (ratio of means; lines at 0 = best equilibrium and 1 = oracle).
* **Fig 2 — `fig2_regret.png` / `fig2_regret_full.png`.**  Raw safe regret (chips), with the NE panel as
  loss v* − u.  FIXED-NE's regret (0.39–1.27 chips) is marked off-scale in each panel.
* **Fig 3 — `fig3_heatmaps.png`.**  Fraction captured, method × family, at N = 20 and N = 500 (annotated,
  diverging scale centred at 0).
* **Fig 4 — `fig4_hybrid_vs_tabem.png`.**  Learned-prior EM regret − TAB-EM regret vs N, one line per
  family, for each hybrid.
* **Fig 5 — `fig5_safety.png`.**
  - (a) Loss against Nash opponents vs N, with the ε line.
  - (b) Expl(x) − ε of every deployed strategy, symlog axis, with the 1e−7 audit tolerance.  Plotting Expl
    itself was uninformative: 100% of modelling strategies sit at Expl = ε to within 1e−6, so it is a single
    line at 0.100.
  - (c) Harm rate vs N, pooled over the five non-NE families.

## 3. Summary table (ε = 0.10; seed 0; fraction = ratio of means over opponents)

Fraction of attainable safe gain, **N = 20 / N = 500**:

| method | ID-REF | NEAR | FAR-ARCH | FAR-CFR | FAR-EXPL | NE mean loss (chips) N = 20 / 500 | harm rate (pooled) N = 20 / 500 |
|---|---|---|---|---|---|---|---|
| FIXED-NE | −0.01 / −0.01 | −0.02 / −0.02 | −0.01 / −0.01 | −0.00 / −0.00 | −0.01 / −0.01 | 0.000 / 0.000 | 0 / 0 (reference) |
| BANK | 0.74 / 0.76 | 0.61 / 0.62 | 0.82 / 0.84 | 0.77 / 0.76 | 0.67 / 0.67 | 0.002 / 0.002 | 0.078 / 0.072 |
| TAB-EM | 0.67 / 0.85 | 0.68 / 0.90 | 0.61 / 0.84 | 0.74 / 0.80 | 0.69 / 0.85 | 0.037 / 0.025 | 0.150 / 0.090 |
| DEC-889k | 0.74 / 0.80 | 0.58 / 0.63 | 0.82 / 0.88 | 0.81 / 0.84 | 0.71 / 0.75 | 0.004 / 0.001 | 0.076 / 0.064 |
| RECON-889k | 0.72 / 0.77 | 0.56 / 0.60 | 0.81 / 0.92 | 0.78 / 0.82 | 0.66 / 0.72 | 0.004 / 0.000 | 0.074 / 0.056 |
| RECON-JAC-global | 0.75 / 0.82 | 0.61 / 0.66 | 0.81 / 0.93 | 0.80 / 0.84 | 0.71 / 0.77 | 0.006 / 0.001 | 0.080 / 0.062 |
| JAC-opp | 0.75 / 0.82 | 0.60 / 0.66 | 0.83 / 0.92 | 0.82 / 0.83 | 0.71 / 0.76 | 0.003 / 0.001 | 0.078 / 0.072 |
| PRIOR-EM (RECON-131k) | 0.76 / **0.90** | 0.68 / **0.90** | 0.83 / **0.94** | 0.79 / **0.89** | 0.75 / **0.92** | 0.003 / 0.003 | 0.078 / **0.038** |
| PRIOR-EM (JAC-opp) | 0.75 / **0.90** | 0.68 / **0.90** | 0.83 / 0.93 | 0.80 / 0.87 | 0.75 / 0.91 | 0.004 / 0.003 | 0.078 / 0.058 |

Median per-opponent fractions at N = 500 are close to the ratios of means (e.g. JAC-opp 0.80 / 0.65 / 0.92 /
0.89 / 0.80; PRIOR-EM (RECON-131k) 0.87 / 0.92 / 0.95 / 0.92 / 0.96).  All values are in `outputs/gen/analysis.json`.

Raw regret at N = 500 (chips): each FIXED-NE value is its regret at every N.

| method | ID-REF | NEAR | FAR-ARCH | FAR-CFR | FAR-EXPL |
|---|---|---|---|---|---|
| FIXED-NE | 0.493 | 0.914 | 0.559 | 0.385 | 1.273 |
| BANK | 0.117 | 0.344 | 0.090 | 0.092 | 0.412 |
| TAB-EM | 0.073 | 0.090 | 0.086 | 0.077 | 0.193 |
| DEC-889k | 0.099 | 0.330 | 0.066 | 0.060 | 0.320 |
| RECON-889k | 0.112 | 0.363 | 0.044 | 0.070 | 0.352 |
| RECON-JAC-global | 0.090 | 0.309 | 0.037 | 0.060 | 0.288 |
| JAC-opp | 0.089 | 0.303 | 0.043 | 0.064 | 0.303 |
| PRIOR-EM (RECON-131k) | 0.048 | 0.088 | 0.035 | 0.044 | 0.096 |
| PRIOR-EM (JAC-opp) | 0.048 | 0.086 | 0.039 | 0.051 | 0.120 |

**Harm by family.**  Harm rate is the fraction of (opponent, N) points with u < u(FIXED-NE).
* **Outside FAR-CFR** it is ≤ 0.03 for every method except TAB-EM, which reaches 0.20 on ID-REF and
  FAR-ARCH at N = 20.
* **On FAR-CFR** it is 0.28–0.36 for every non-hybrid method (hybrids 0.17–0.27 at N = 500).  All of it
  comes from the 40 full-width CFR/CFR+ opponents, which are nearly at equilibrium: median exploitability
  0.11–0.29 and median headroom 0.03–0.09 chips.
  - **How often:** 75–95% of their points fall below FIXED-NE for TAB-EM and JAC-opp, and 30–55% for the
    RECON-131k hybrid at N = 500.
  - **How much:** the mean shortfall is 0.055 (TAB-EM), 0.038 (JAC-opp) and 0.011 (hybrid), with a
    maximum of 0.076, below ε.
  - The 60 MCCFR opponents (exploitability ≈ 2.2) are never harmed.
* File: `outputs/gen/posthoc_farcfr_harm.json`.

## 4. Pre-registered expectations

* **E1 — fails.**  The bank's drop from ID-REF to the FAR mean at N = 500 is only +0.003.  That is the
  second-largest of the eight modelling methods, after TAB-EM's +0.020.  The JAC-opp hybrid's is +0.001, and
  the four neural models and the RECON-131k hybrid *improve* on FAR (drops −0.015 to −0.050).
  - Its N = 500 drop does exceed its N = 20 drop (−0.008), but it is not the largest.
  - The bank does degrade most, together with the neural models, on the two families far from the bank in
    g-space.  On NEAR its fraction is 0.62 vs 0.76 on ID-REF; on FAR-EXPL it is 0.67.
  - Its regret is flat in N there (0.34–0.41 at N = 500), because the posterior concentrates on the
    nearest wrong bank members.  So the spirit of E1 (the bank breaks on novel opponents, especially at
    large N) holds for NEAR and FAR-EXPL.  But FAR-ARCH and FAR-CFR are not novel in decision space, so
    the operational test fails.
* **E2 — fails for all four neural models.**
  - Fraction(ID-REF) > fraction(NEAR) holds at N = 20 and 500 (0.72–0.75 → 0.56–0.61; 0.77–0.82 →
    0.60–0.66).
  - But the FAR mean is *higher* than NEAR, and on FAR-ARCH and FAR-CFR even higher than ID-REF.
  - Large-N floors rise sharply on NEAR (0.30–0.36 vs 0.09–0.11 chips) and FAR-EXPL (0.29–0.35), but fall
    on FAR-ARCH (0.04–0.07) and FAR-CFR (0.06–0.07).
* **E3 — fails, in the favourable direction.**  "Degrades gracefully" is borne out: both hybrids are the
  best or tied-best method on every family at N = 500, and are never significantly worse than TAB-EM.
  They do not *approach* TAB-EM at N = 500, though — they stay better:

  | hybrid | ID-REF | FAR-ARCH | FAR-CFR | FAR-EXPL | NE | NEAR |
  |---|---|---|---|---|---|---|
  | PRIOR-EM (RECON-131k) − TAB-EM at N = 500 | −0.024 | −0.051 | −0.033 | −0.097 | −0.022 | −0.002 (n.s.) |
  | PRIOR-EM (JAC-opp) − TAB-EM at N = 500 | −0.025 | −0.047 | −0.026 | −0.073 | −0.022 | −0.004 (n.s.) |

  All but NEAR have CIs excluding 0.  With a uniform Dirichlet(1) prior, 500 hands still leave tabular EM
  far from consistent on rarely visited infosets.  The learned prior fills exactly those.
* **E4 — fails: the learned prior never hurts significantly at any N on any family.**
  - The only region where it is not better is NEAR at N = 10–200 (+0.002 to +0.017, all CIs include 0).
    Those are the generators pushed past their training ranges, where a "confidently wrong" prior would be
    expected.
  - On the FAR families it helps most exactly at small N: FAR-ARCH −0.09 to −0.12 at N = 5–20, FAR-EXPL
    −0.03 to −0.08.
  - **Where Fig 4 crosses zero** (the smallest N from which the difference stays ≤ 0):

    | hybrid | ID-REF | FAR-ARCH | FAR-EXPL | NE | FAR-CFR | NEAR |
    |---|---|---|---|---|---|---|
    | PRIOR-EM (RECON-131k) | N = 5 | N = 5 | N = 5 | N = 5 | N = 10 (+0.003 n.s. at N = 5) | N = 500 |
    | PRIOR-EM (JAC-opp) | N = 5 | N = 5 | N = 5 | N = 5 | N = 5 | N = 500 |
* **E5 — holds.**  Every (method, history, N) loss against the NE opponents lies in [−δ, ε].  The largest
  single loss is 0.079 (PRIOR-EM (RECON-131k)).  FIXED-NE's minimum is −1e−5 against the CFR+ opponent,
  within its δ = 4.7e−5.  Mean loss per method is at most 0.040 (TAB-EM at N = 50); every other method is
  ≤ 0.022 at N = 5 and ≤ 0.006 from N = 20 on.

## 5. Exploratory (post hoc): distance from the training bank explains where learned models break

Per non-NE opponent, the N = 500 gap to tabular EM (regret(method) − regret(TAB-EM)) was compared with the
opponent's nearest-neighbour distance to the 1 200 training opponents in g-space
(`outputs/gen/posthoc_distance_vs_gap.json`):

| method | Spearman(distance, gap) | mean gap, nearest third | mean gap, farthest third |
|---|---|---|---|
| BANK | +0.66 | +0.001 | +0.251 |
| DEC-889k | +0.51 | −0.019 | +0.190 |
| JAC-opp | +0.51 | −0.027 | +0.169 |
| PRIOR-EM (RECON-131k) | −0.18 | −0.030 | −0.060 |
| PRIOR-EM (JAC-opp) | −0.11 | −0.029 | −0.045 |

* **Near the training support,** the learned models match or beat tabular EM.
* **Beyond it,** they fall behind by 0.17–0.25 chips, because their estimate stays anchored to the training
  support.
* **The hybrids' advantage over tabular EM does not shrink with distance.**  Its likelihood term keeps
  moving the estimate towards the data.

## 6. Audit table (every deployed strategy, ε = 0.10)

| method | deployed strategies | max Expl − ε | violations (> 1e−7) | LP failures |
|---|---|---|---|---|
| FIXED-NE | 1 (the same x_nash for every point) | −0.1000 (Expl = 2.3e−13) | 0 | — |
| BANK | 7 700 | 7.3e−12 | 0 | 0 |
| TAB-EM | 7 700 | 6.8e−11 | 0 | 0 |
| DEC-889k | 7 700 | 1.5e−10 | 0 | 0 |
| RECON-889k | 7 700 | 4.2e−11 | 0 | 0 |
| RECON-JAC-global | 7 700 | 9.2e−11 | 0 | 0 |
| JAC-opp | 7 700 | 5.6e−11 | 0 | 0 |
| PRIOR-EM (RECON-131k) | 7 700 | 7.0e−11 | 0 | 0 |
| PRIOR-EM (JAC-opp) | 7 700 | 2.0e−11 | 0 | 0 |
| **total** | **61 601** | **1.5e−10** | **0** | **0** |

Every modelling strategy has Expl(x) ∈ [ε − 1e−6, ε + 1.5e−10].  The exact safe LP always spends the full
budget, so "practical safety" here means exactly ε, never less.

## 7. Wall-clock

| stage | wall-clock |
|---|---|
| Families, values and histories (`gen_families.py`) | 2.4 min |
| Predictions, 8 modelling methods × 1 100 histories × 7 N (`gen_eval.py`) | 5.5 min.  Slowest: TAB-EM 73 s, PRIOR-EM 46–55 s; neural 13–20 s each, BANK 2 s |
| Exact LPs + OpenSpiel audits, 61 600 on 4 workers | 13.5 min (≈ 53 ms per LP + audit per worker) |
| Analysis, post-hoc checks, figures | < 2 min |
| Total compute | ≈ 23 min |

## 8. Deviations and notes

1. **κ for the JAC-opp prior.**  κ = 3 at every N, as the spec's method list states, although the original
   validation selected κ = 10 at N = 10 and 20 for that prior.  For the RECON-131k prior, κ = 3 is the
   original validation choice at every N.
2. **Seeds.**  Neural models *and* both hybrid priors use seed 0.  RECON-JAC-global is the regret-selected
   `runs_jacopp/a0_s0` rather than V3's loss-selected run (declared in §0).
3. **ID-REF histories.**  ID-REF reuses streams 0–1 of the existing test datasets, which the same simulator
   and learner blueprint produced.  All other families use fresh streams with disjoint split ids.
4. **Figure 5(b)** shows Expl − ε on a symlog axis instead of raw Expl, because the raw distribution is a
   single point at ε for every modelling method (see §6).
5. **The operational forms of E1–E5** are mine.  The spec stated them qualitatively; the operational versions
   were written and committed before any evaluation.
6. **Post-hoc analyses.**  The distance-vs-gap analysis (§5) and the FAR-CFR harm breakdown (§3) were not
   pre-registered.
7. **Nothing was tuned on the new families, and no family, method or opponent was cut.**  Opponent counts
   are as pre-registered (100 per family, 5 × 20 streams for NE); no opponent fell below the 0.01-chip
   headroom cut.
