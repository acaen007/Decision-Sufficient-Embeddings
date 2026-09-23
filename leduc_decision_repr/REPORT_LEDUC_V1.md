# REPORT_LEDUC_V1 — Decision-sufficient opponent representations in two-player Leduc poker

*Sample efficiency of decision-focused vs. full-behavioral opponent representations at identical, exactly certified safety.*

> Sections 1–9 describe the frozen design; sections 10–16 report the results of the single held-out test evaluation; §17 gives runtime, reproduction commands and artifact paths; Appendix A has the full tables.

## 1. Scientific motivation

We ask whether an agent can learn a compact opponent representation that preserves only what is needed to
choose a good *safe* response, rather than reconstructing the opponent's full behavior.  In sequence form the
learner's utility against an opponent realization plan `y_q` is `u(x, q) = xᵀ A y_q = xᵀ g(q)` with
`g(q) = A y_q`; `g(q)` is therefore a natural *decision-sufficient* target: it contains exactly the
opponent-dependent coefficients needed to evaluate any of the learner's plans.  We compare two systems that
share an identical history encoder, latent size, data, optimizer budget and certified safe solver, and differ
only in the representation objective:

* **NEURAL_RECON**: `H_N → z → q̂ (all opponent infosets) → ŷ → ĝ = A ŷ → exact ε-safe LP → x̂`
* **NEURAL_DECISION**: `H_N → z → ĝ → the same ε-safe LP → x̂`

The central question (task §5): at identical, exactly verified exploitability bounds, does learning the
opponent's decision consequences instead of its complete strategy reduce the number of observed Leduc hands
needed to recover the same amount of safe exploitation?  The headline x-axis is the number of observed hands
N; the y-axis is safe response regret and the fraction of oracle-safe exploitation recovered.

### Prior RPS findings motivating this experiment
Stationary and memory-1/2 RPS experiments (earlier in this project) found that response-aware objectives
extract decision-relevant information with far less representational capacity (effective dimension ≈5 vs
≈12.7 at nominal d=32 in memory-2 RPS) while reconstruction eventually catches up or overtakes once given
enough capacity (crossover at nominal d≈8–12).  The present hypothesis is the *statistical* analogue: with few
observations decision-focused learning may outperform reconstruction, with abundant observations reconstruction
may catch up.  The experiment is designed so that a negative result is equally interpretable (§16).

## 2. Game

`pyspiel.load_game("leduc_poker")`: 2 players, 6 cards (3 ranks × 2 suits), ante 1, raise sizes 2 (round 1)
and 4 (round 2), at most 2 raises per round, actions FOLD / CALL(=check) / RAISE.  Rank = card id // 2
(verified against showdown payoffs).  Enumerated tree: 9 457 histories, 5 520 terminals, 468 OpenSpiel
information sets per player (these distinguish physical cards of the same rank), 1 093 sequences per player,
4 920 non-zero entries in the sequence-form payoff matrix `A`.  Game value for player 0:
`v* = −0.08560642408` (OpenSpiel's own sequence-form LP gives −0.08560642405; CFR+ agrees).

**Learner = player 0** (acts first in each round).  OpenSpiel Leduc has a fixed seat order, so every
observed hand is an independent game with the learner in seat 0; seats do not alternate.  This is the simplest
faithful setting and is listed as a limitation (§15).

**Suit symmetry.**  OpenSpiel's information-set strings include the physical card id, so the two copies of a
rank are distinct infosets even though suit is strategically meaningless (the game is invariant under the 8
suit permutations; verified numerically).  We therefore define opponent policies on the 144 *rank-level*
infosets (private rank, public rank or none, betting history) and tie them across their physical copies
(468 → 144).  The learner's and opponent's equilibrium blueprints are made suit-symmetric by averaging their
realization plans over the permutation group (the equilibrium set is convex and group-invariant, so the
average is again an exact equilibrium; audited exploitability 2.3e−13).  The learner's deployed strategies
from the safe LP are *not* symmetrized (they need not be).

**Equilibrium selection (blueprints).**  Nash equilibria are not unique.  The learner's observation blueprint is
the ε=0 safe-LP solution that maximizes value against a uniform-random opponent (a deterministic,
population-independent rule), then symmetrized; the opponent's Nash policy used by the Nash-based families is
selected symmetrically.  At infosets unreachable under a player's own plan the behavioral policy is uniform.
55% of the learner's rank infosets are deterministic under this blueprint; consequently only 401 of the 792
possible player-0 hand observation types ever occur in the data.

## 3. Sequence form and exact safety (derivations)

Sequences: id 0 = empty sequence, one id per (infoset, legal action).  Perfect recall was asserted during
enumeration (every node of an infoset has the same own-sequence prefix).  Realization constraints
`E x = e` (row 0: `x_∅ = 1`; row `1+I`: `x_{parent(I)} − Σ_a x_{(I,a)} = 0`), likewise `F y = f`.
Payoff: `A[s₀, s₁] = Σ_z chance(z)·u₀(z)` over terminals whose last sequences are `(s₀, s₁)`, so that
`u₀(x, y) = xᵀ A y` and `g(q) = A y_q`.

**Security.**  `sec(x) = min_y { xᵀ A y : F y = f, y ≥ 0 }`.  By LP duality
`sec(x) = max_v { fᵀ v : Fᵀ v ≤ Aᵀ x }`, and since `f = (1, 0, …, 0)`, `fᵀ v = v₀`.  The dual variables
`v_J` are (minus) the opponent's infoset values; we verified numerically that the dual optimum equals the exact
tree best-response value for random `x`.  The learner's security loss is `e(x) = v* − sec(x)`.

**ε-safe LP** (identical for the oracle and all methods):
```
max_{x, v}  ĝᵀ x
s.t.        E x = e,  x ≥ 0
            Fᵀ v − Aᵀ x ≤ 0
            v₀ ≥ v* − ε              (v otherwise free)
```
Player 1's game-value LP is the mirror image with `P = −Aᵀ`.  Solved with HiGHS (highspy, dual simplex,
primal/dual feasibility tolerances 1e−9, model kept in memory and warm-started across consecutive solves).
The returned plan is converted to a behavioral policy (uniform at unreachable infosets); that *policy* is the
deployed object, and its value and exploitability are computed from its own re-derived realization plan.

**Independent audit.**  Every deployed strategy is audited with OpenSpiel's C++
`TabularBestResponse(game, 1, policy).value("")`, which gives `max_y u₁`, so `e(x) = v* + BR₁(x)`; this uses
OpenSpiel's own game tree and best-response code, not our `A`.  Requirement: `e(x) ≤ ε + 1e−7`.  A fast exact
tree best response (our own, derived from the enumerated tree rather than from `A`) is also computed for every
strategy and compared with the OpenSpiel value.

## 4. Validation tests (all passing; `python -m pytest leduc_decision_repr/tests`, 31 tests)

Game-theoretic suite (task §21): (1) random behavioral policies → realization plans satisfy `Ex=e`, `Fy=f`
to 1e−12; (2) behavioral↔realization round trip exact on reachable infosets, uniform on unreachable ones;
(3) `xᵀAy` equals an independent OpenSpiel tree walk to 1e−10 (random and deterministic policies) and reach
probabilities sum to one; (4) `xᵀg(q) = xᵀAy_q`; (5) game value matches OpenSpiel's sequence-form LP to 1e−6
and CFR+ within its NashConv; (6) fast tree best response matches OpenSpiel's C++ and Python best responses
to 1e−9; the dual LP optimum matches the tree security value; (7) safe LP: exploitability bound respected
(both auditors, tolerance 1e−7), value ≥ Nash value, ≤ unrestricted best response, monotone in ε, ε=0 gives
a minimax strategy, and a large ε reaches the unrestricted best-response value; (8) suit symmetry: payoff
invariance under all 8 permutations, symmetrized blueprints remain exact equilibria and are suit-tied.
Additional: tree public state (pot, money) equals OpenSpiel's strings at all 9 457 nodes; action deltas equal
OpenSpiel money differences.

Tokenizer/simulator suite (task §14): items 1–8 in `tests/test_tokenizer.py`, items 9–10 (padding
invariance, chronological sensitivity) in `tests/test_models.py`, plus the vectorized simulator equals a scalar
reference implementation, produces the exact prefix property, and its empirical hand-type frequencies match
the exact reach probabilities (χ² check on 20 000 hands).

Baseline suite: rank-policy→g map equals the sequence-form computation; hand likelihood ratios equal exact
reach-probability ratios; showdown hands have exactly one candidate opponent rank and folded hands several;
EM log-likelihood is monotone and the reach-weighted policy error falls to <0.05 by 20 000 hands; the
train-bank posterior identifies the generating opponent from 200 hands with mass >0.99 and is the uniform
prior at N=0.

## 5. Opponent population and split

1 650 opponent policies (player 1, rank-level), four families of 412–413, stratified split
train 1 200 / validation 150 / test 300 = 300/38/75, 300/38/75, 300/37/75, 300/37/75 per family.  Held-out
means held-out *policy*: no test opponent is used for training or checkpoint selection.  Seeds: opponent k of
family f is generated from `default_rng([2024, f, k])`; split permutations from `[2024, 999, f]`.

* **NASH_LOGIT_PERTURB** — `logits = log(floor₀.₀₂(nash)) + τ (Wᵀφ(I)/‖φ‖ + ε_I)`, `W ~ N(0,1)` over 23 shared
  infoset features, `ε_I ~ N(0, s²)`, `s ~ U[0.2, 1]`, `τ ~ LogU[0.15, 2.0]`.
* **NASH_RANDOM_MIX** — `q = (1−η)·nash + η·Dirichlet(1)` per infoset, `η ~ U[0.05, 0.85]`.
* **STRUCTURED_CORRELATED** — trait-driven logits (8 traits: aggression, passivity, strength sensitivity,
  bluffing, over-folding, over-calling, round-2 shift, tightness) plus random coefficients (N(0, 0.6²)) on the
  shared features (private rank, public rank, pairing, round, facing aggression, amount to call, raise counts,
  pot/contributions, legal actions, hand strength) plus per-infoset noise N(0, 0.3²).
* **UNSTRUCTURED_DIRICHLET** — independent `Dirichlet(α)` per infoset, `α ~ LogU[0.3, 3]` (control family with
  no shared cross-infoset structure).

Saved: policy tables (rank and physical), family labels, generator parameters, seeds, realization plans,
`g(q)`, unrestricted best-response values, and the safe oracle at ε ∈ {0, 0.05, 0.10, 0.20}
(6 600 LPs, 0 failures, max audited violation 1.4e−10).

**Precondition analysis (population level, training opponents unless stated).**

| quantity | value |
|---|---|
| full exploitability `BR − v*`, mean (all / A / B / C / D) | 1.99 / 0.78 / 1.35 / 2.87 / 2.98 chips per hand |
| oracle-safe gain `V_ε − u(x_Nash)`, mean, ε = 0 / 0.05 / 0.10 / 0.20 | 0.004 / 0.351 / 0.493 / 0.682 |
| gain at ε=0.10, deciles 10/50/90 % | 0.19 / 0.44 / 0.96 |
| policy table: legal dims / participation ratio / dims for 90 % var | 336 / 8.1 / 83 |
| `g(q)`: non-constant dims / PR (raw, standardized) / dims for 90 % var (raw, std.) | 1 074 / 6.2, 10.7 / 29, 42 |
| per family PR (policy → g): A, B, C, D | 11.1→7.9, 9.2→6.1, 4.5→3.6, 103.9→25.5 |
| Spearman ρ(d_beh, d_resp) over 50 000 random population pairs, ε=0.10 | 0.69 |

Two facts matter for interpretation.  At **ε = 0 there is essentially no exploitable value** (mean 0.004
chips; the learner must be exactly minimax and only equilibrium selection remains), so ε=0 is a pure safety
check, not a learning benchmark.  And **g(q) is lower-dimensional than the policy** in a linear sense (90 %
of variance in 29 vs 83 dimensions; PR 25 vs 104 for the unstructured family) but is far from
one-dimensional, so decision prediction offers compression but not a trivial target.

## 6. Observation process and datasets

During the N observation hands the learner plays the fixed Nash blueprint and the opponent plays q (no
probing, no adaptation).  One stream = 500 completed hands; N ∈ {5, 10, 20, 50, 100, 200, 500} are prefixes
of the same stream, so N=10 contains the first 5 hands of N=5, etc.  Streams: train 4, validation 4, test 8
per opponent (7 800 streams, 3.9 M hands).  The uniforms of stream s of opponent k in split σ come from
`default_rng([7, σ, k, s])` only, so each stream is a deterministic function of its seed.  A hand is stored as
its terminal-history index; tokenization is a lookup.  Test streams are averaged *within* opponent before
opponents are treated as statistical units.

## 7. Tokenization (exactly what player 0 observed)

One token per game event: `HAND_START`, `ACTION`, `PUBLIC_CARD`, `SHOWDOWN`, `HAND_END` (≤ 12 events per
hand).  Categorical fields (embedded): event type, actor (self/opponent/chance), betting round, action type
(fold / check-call / raise), our private rank, public rank (from the `PUBLIC_CARD` event onward), revealed
opponent rank (**only in a `SHOWDOWN` token**), terminal type (fold-self / fold-opp / showdown, `HAND_END`
only), the actor's legal-action set ({C,R}, {F,C,R}, {F,C}).  Numeric fields (normalized, linearly
projected): pot before the action, amount to call, our and the opponent's contributions before the action,
raises so far this round, the actual contribution delta of the action (derived from the game transition,
e.g. 8 chips for a round-2 raise facing a raise), and the payoff (`HAND_END`).  Physical card identity is
collapsed to rank everywhere.  The only function that reads the tree's card array is `observe_terminal`, and
it reads the opponent's card only when the terminal is a showdown.  There are 792 distinct player-0
observation types over the 5 520 terminals; the hand encoder is evaluated once per distinct type per batch
and gathered (identical function; dropout masks shared between identical hands).

Example tokenized hands (from `outputs/eval/test/tokenized_examples.txt`; also Figure 10):

```
Hand A: we fold in round 1 (opponent card never revealed)  [observation type 18; internal terminal 66; tree knows opponent card 1 (rank 0)]
 # event       actor  rnd action     our pub opp term      legal   |  pot call c_us c_op rais delta   pay
 0 HAND_START  -      1   -          J   -   -   -         -       |    2    0    1    1    0     0   0.0
 1 ACTION      SELF   1   CHECK/CALL J   -   -   -         {C,R}   |    2    0    1    1    0     0   0.0
 2 ACTION      OPP    1   RAISE      J   -   -   -         {C,R}   |    2    0    1    1    0     2   0.0
 3 ACTION      SELF   1   FOLD       J   -   -   -         {F,C,R} |    4    2    1    3    1     0   0.0
 4 HAND_END    -      1   -          J   -   -   FOLD_SELF -       |    4    0    1    3    0     0  -1.0

Hand B: public card dealt, opponent folds in round 2 (card never revealed)  [observation type 3; internal terminal 12; tree knows opponent card 1 (rank 0)]
 # event       actor  rnd action     our pub opp term      legal   |  pot call c_us c_op rais delta   pay
 0 HAND_START  -      1   -          J   -   -   -         -       |    2    0    1    1    0     0   0.0
 1 ACTION      SELF   1   CHECK/CALL J   -   -   -         {C,R}   |    2    0    1    1    0     0   0.0
 2 ACTION      OPP    1   CHECK/CALL J   -   -   -         {C,R}   |    2    0    1    1    0     0   0.0
 3 PUBLIC_CARD CHANCE 2   -          J   Q   -   -         -       |    2    0    1    1    0     0   0.0
 4 ACTION      SELF   2   CHECK/CALL J   Q   -   -         {C,R}   |    2    0    1    1    0     0   0.0
 5 ACTION      OPP    2   RAISE      J   Q   -   -         {C,R}   |    2    0    1    1    0     4   0.0
 6 ACTION      SELF   2   RAISE      J   Q   -   -         {F,C,R} |    6    4    1    5    1     8   0.0
 7 ACTION      OPP    2   FOLD       J   Q   -   -         {F,C}   |   14    4    9    5    2     0   0.0
 8 HAND_END    -      2   -          J   Q   -   FOLD_OPP  -       |   14    0    9    5    0     0   5.0

Hand C: showdown after a public card (opponent rank revealed only in the SHOWDOWN token)  [observation type 4; internal terminal 13; tree knows opponent card 1 (rank 0)]
 # event       actor  rnd action     our pub opp term      legal   |  pot call c_us c_op rais delta   pay
 0 HAND_START  -      1   -          J   -   -   -         -       |    2    0    1    1    0     0   0.0
 1 ACTION      SELF   1   CHECK/CALL J   -   -   -         {C,R}   |    2    0    1    1    0     0   0.0
 2 ACTION      OPP    1   CHECK/CALL J   -   -   -         {C,R}   |    2    0    1    1    0     0   0.0
 3 PUBLIC_CARD CHANCE 2   -          J   Q   -   -         -       |    2    0    1    1    0     0   0.0
 4 ACTION      SELF   2   CHECK/CALL J   Q   -   -         {C,R}   |    2    0    1    1    0     0   0.0
 5 ACTION      OPP    2   RAISE      J   Q   -   -         {C,R}   |    2    0    1    1    0     4   0.0
 6 ACTION      SELF   2   RAISE      J   Q   -   -         {F,C,R} |    6    4    1    5    1     8   0.0
 7 ACTION      OPP    2   CHECK/CALL J   Q   -   -         {F,C}   |   14    4    9    5    2     4   0.0
 8 SHOWDOWN    CHANCE 2   -          J   Q   J   -         -       |   18    0    9    9    0     0   0.0
 9 HAND_END    -      2   -          J   Q   -   SHOWDOWN  -       |   18    0    9    9    0     0   0.0

```

## 8. Models, losses and training

**Encoder (shared).**  Event vector = Σ categorical embeddings + linear(numeric) + event-position
embedding; `[HAND_CLS]` + 2 pre-LayerNorm Transformer layers (d_model 128, 4 heads, FF 256, dropout 0.1 on
residual/FF branches, no attention-probability dropout) → hand embedding (CLS, LayerNorm).  Hand embeddings +
learned hand-position embedding + `[HISTORY_CLS]` → 2 identical layers (bidirectional over the N completed
hands, key-padding mask) → CLS → linear → `z ∈ R¹²⁸`.  618 752 encoder parameters.

**Reconstruction head** (131 331 params): `z` and a representation of each of the 144 opponent rank infosets
(learned 64-d embedding ⊕ 23 structured features) → MLP → masked softmax over legal actions, evaluated for all
infosets at once.  Loss: soft-target cross-entropy `−Σ_a q(I,a) log q̂(I,a)` averaged uniformly over infosets
and batch, against the *true* policy q (the encoder only ever sees observed hands).  At evaluation
`q̂ → ŷ → ĝ = A ŷ` (differentiable map, checked against the numpy sequence form).

**Decision head** (889 413 params): `z → MLP(512, 512) → ĝ` predicted in standardized units
`(g − μ)/σ` with μ, σ from the 1 200 training opponents; the 19 of 1 093 coordinates with
`σ ≤ 1e−3·max σ` are excluded from the loss and predicted at their training mean.  Loss: normalized MSE over
the 1 074 valid coordinates.  Raw `‖ĝ − g‖₂` is also reported.

**Training** (frozen, `configs/frozen_v1.json`): each step samples N uniformly from the 7 budgets and 32
(train opponent, stream) pairs, uses the first N hands, and regresses the target.  AdamW, lr 3e−4 with 200
warm-up steps and cosine decay to 3e−5, weight decay 0.01, gradient clipping 1.0, 6 000 steps
(192 000 samples ≈ 5.7 passes over the 33 600 (stream, N) combinations), validation every 250 steps on all
150 validation opponents × 4 streams × 7 budgets; the checkpoint with the lowest validation loss (each method's
own objective) is used.  Three seeds (0, 1, 2) per objective; the seed fixes initialization and the sampling
stream.  The configuration was frozen after a 400-step train/val smoke test and a validation-split dry run of the
evaluation pipeline; the test split was evaluated once, after training.

**Training curves (Figure 11).**  Validation loss vs step for all six runs: the three decision seeds converge
tightly (final validation NMSE 0.567 / 0.569 / 0.568; best checkpoints at steps 6000 / 4750 / 6000), the
three reconstruction seeds likewise (final CE 0.637 / 0.637 / 0.635; best at 6000 / 5750 / 5750) but were
still decreasing slowly at the end of the budget.  The validation g-NMSE implied by the reconstruction models
plateaus near 0.56–0.57 (N=500) from step ≈ 3000, while the decision heads reach 0.40.  Wall time per run:
8 500–9 200 s (round one, one thread each) and 3 300–3 500 s (round two, two threads each).

## 9. Baselines

* **NASH** — deploy the blueprint (regret = oracle gain by definition).
* **ORACLE_SAFE** — true `g(q)` into the same LP: `V_ε(q)`, the ceiling.
* **TABULAR_EM_UNIFORM / TABULAR_EM_NASH** — EM over the opponent's rank-level policy with the hidden private
  rank marginalized: for each observed hand and each candidate opponent rank consistent with card removal and
  any showdown reveal (1 116 (type, rank) pairs), the E-step posterior uses the current q̂, the M-step is the
  Dirichlet-MAP update `q̂(I,a) = (expected count + α·prior(I,a)) / (expected visits + α)` with α = 1 pseudo
  count per infoset, prior = uniform over legal actions or the opponent's Nash policy; 200 iterations or
  change < 1e−7.  Folded hands never reveal the card.
* **BANK_POSTERIOR** — exact marginalized likelihood of the history under each of the 1 200 training
  opponents (q-independent factors cancel), uniform prior, `ḡ = Σ_k p(q_k|H) g(q_k)`.

All methods' ĝ go through the identical LP at the identical ε and are audited identically.

## 10. Headline result: safe response regret vs hands observed (Figure 1, Figure 2, Tables R1–R4)

All numbers: 300 held-out test opponents, 8 independent observation streams each (averaged within opponent),
one evaluation of the test split after the configuration was frozen; every deployed strategy audited
(§11b).  Neural methods are averaged over 3 seeds (per-seed rows in Appendix Table R1 differ from the seed
mean by ≤ 0.003 chips).

**Safe response regret `R_safe = V_ε(q) − u(x̂, q)` (chips/hand, mean, 95 % paired-bootstrap CI):**

| ε | method | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 |
|---|---|---|---|---|---|---|---|---|
| 0.05 (Nash regret 0.347) | NEURAL_DECISION | 0.126 | 0.112 | 0.102 | 0.091 | 0.086 | 0.084 | 0.081 |
| | NEURAL_RECON | 0.132 | 0.117 | 0.107 | 0.097 | 0.093 | 0.091 | 0.089 |
| | BANK_POSTERIOR | 0.119 | 0.107 | 0.101 | 0.096 | 0.096 | 0.094 | 0.094 |
| | TABULAR_EM_UNIFORM | 0.147 | 0.135 | 0.122 | 0.098 | 0.079 | 0.064 | 0.050 |
| | TABULAR_EM_NASH | 0.318 | 0.285 | 0.251 | 0.200 | 0.163 | 0.136 | 0.110 |
| 0.10 (Nash regret 0.491) | NEURAL_DECISION | 0.167 | 0.149 | 0.135 | 0.120 | 0.112 | 0.108 | 0.105 |
| | NEURAL_RECON | 0.177 | 0.157 | 0.142 | 0.128 | 0.124 | 0.120 | 0.118 |
| | BANK_POSTERIOR | 0.158 | 0.144 | 0.136 | 0.128 | 0.127 | 0.123 | 0.122 |
| | TABULAR_EM_UNIFORM | 0.196 | 0.180 | 0.165 | 0.135 | 0.114 | 0.093 | 0.075 |
| | TABULAR_EM_NASH | 0.408 | 0.365 | 0.321 | 0.257 | 0.212 | 0.179 | 0.147 |
| 0.20 (Nash regret 0.684) | NEURAL_DECISION | 0.238 | 0.210 | 0.188 | 0.165 | 0.154 | 0.147 | 0.142 |
| | NEURAL_RECON | 0.251 | 0.224 | 0.200 | 0.180 | 0.172 | 0.165 | 0.162 |
| | BANK_POSTERIOR | 0.225 | 0.202 | 0.190 | 0.176 | 0.174 | 0.170 | 0.171 |
| | TABULAR_EM_UNIFORM | 0.276 | 0.257 | 0.232 | 0.191 | 0.166 | 0.139 | 0.113 |
| | TABULAR_EM_NASH | 0.556 | 0.498 | 0.436 | 0.352 | 0.291 | 0.247 | 0.208 |

(CIs are ±0.010–0.015 for the neural methods; full CIs in Appendix Table R1.)  At ε = 0 the oracle gain is
0.005 chips and every method except TABULAR_EM_NASH sits at 0.002–0.005; ε = 0 is a pure safety check.

**Paired difference decision − reconstruction (mean over opponents, 95 % CI; negative = decision better):**

| ε | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 |
|---|---|---|---|---|---|---|---|
| 0.05 | −0.006 [−0.009, −0.002] | −0.005 [−0.008, −0.003] | −0.005 [−0.008, −0.002] | −0.006 [−0.010, −0.002] | −0.008 [−0.012, −0.004] | −0.007 [−0.012, −0.003] | −0.008 [−0.013, −0.003] |
| 0.10 | −0.010 [−0.015, −0.005] | −0.009 [−0.012, −0.005] | −0.008 [−0.012, −0.004] | −0.008 [−0.013, −0.003] | −0.012 [−0.018, −0.007] | −0.012 [−0.019, −0.006] | −0.013 [−0.020, −0.006] |
| 0.20 | −0.013 [−0.020, −0.007] | −0.014 [−0.020, −0.009] | −0.012 [−0.018, −0.006] | −0.014 [−0.022, −0.007] | −0.019 [−0.027, −0.011] | −0.018 [−0.028, −0.010] | −0.020 [−0.030, −0.010] |

Regret ratio decision/reconstruction: 0.94–0.96 at N ≤ 20, 0.89–0.92 at N ≥ 100 (all three ε).
Area under regret-vs-log N: ε=0.10: 0.126 [0.114, 0.137] vs 0.136 [0.123, 0.149] (ratio 0.925);
ε=0.05: ratio 0.937; ε=0.20: ratio 0.918.

**Hands-equivalent reading of the same curves (ε = 0.10; identical at 0.05 and 0.20).**  The reconstruction
model first reaches the decision model's regret at N=5 with N=10, at N=10 with N=20, at N=20 with N=50, at
N=50 only with N=500, and never (within the 500-hand budget) matches the decision model's regret at
N ≥ 100.  So at small budgets the decision representation is worth roughly a 2–2.5× reduction in observed
hands, and at large budgets the reconstruction model does not catch up at all.

**Fraction of oracle-safe gain recovered `F` (opponents with gain ≥ 0.02; 291/299/300 of 300 at
ε = 0.05/0.10/0.20):**

| ε | method | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 | N50 | N80 | N90 | AUC_F |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.10 | NEURAL_DECISION | 0.54 | 0.59 | 0.63 | 0.66 | 0.68 | 0.70 | 0.71 | 5 | >500 | >500 | 0.650 [0.624, 0.675] |
| 0.10 | NEURAL_RECON | 0.51 | 0.58 | 0.62 | 0.66 | 0.68 | 0.70 | 0.71 | 5 | >500 | >500 | 0.644 [0.620, 0.669] |
| 0.10 | BANK_POSTERIOR | 0.56 | 0.61 | 0.64 | 0.67 | 0.68 | 0.70 | 0.71 | 5 | >500 | >500 | 0.659 [0.639, 0.679] |
| 0.10 | TABULAR_EM_UNIFORM | 0.40 | 0.42 | 0.44 | 0.50 | 0.55 | 0.59 | 0.65 | 100 | >500 | >500 | 0.508 [0.459, 0.556] |
| 0.10 | TABULAR_EM_NASH | 0.09 | 0.16 | 0.22 | 0.33 | 0.42 | 0.49 | 0.57 | 500 | >500 | >500 | 0.333 [0.304, 0.361] |

The picture at ε = 0.05 and 0.20 is the same (Appendix Table R2).  The paired F difference
(decision − reconstruction) is +0.02 to +0.03 at N=5 (CI excludes 0 at every ε), +0.01 at N=10 (CI excludes
0 at ε=0.10 only) and indistinguishable from 0 for N ≥ 20.  F is dominated by the highly exploitable
opponents (it is a ratio with the oracle gain in the denominator) and is therefore less sensitive than regret
to the differences that matter for the near-equilibrium opponents.

**N-thresholds (Figure 3, Table R3).**  N50 = 5 for the decision model, the bank posterior and (except at
ε=0.20, where it is 10) the reconstruction model; 100 for uniform-prior EM; 500 for Nash-prior EM.  No method
reaches 80 % or 90 % of the oracle-safe gain by 500 hands (bootstrap: not reached in 100 % of resamples).
The N80/N90 comparison the task asked for is therefore empty for every method: the amortized methods plateau
at F ≈ 0.70–0.71 and the tabular estimator is still at 0.65 at N=500.  The ratio
N80(reconstruction)/N80(decision) is undefined; the informative ratios are the hands-equivalents above.

**What the baselines add.**
* The **train-bank posterior** — exact Bayesian reuse of the 1 200 training opponents — is the best method at
  N ≤ 10 (regret 0.158 vs 0.167 for the decision model at N=5, ε=0.10; F 0.56 vs 0.54) and is overtaken by the
  decision model from N=20 on; it plateaus at 0.122 because a held-out opponent is never in the bank and the
  posterior collapses onto its nearest bank members.
* **Uniform-prior tabular EM** starts worst of the reasonable methods (it knows nothing about the population),
  crosses the reconstruction model at N ≈ 100, the decision model at N=100 (ε=0.05) / N=200 (ε=0.10, 0.20),
  and is the best method at N=500 (0.075 vs 0.105 at ε=0.10).  Per-opponent estimation wins once enough hands
  are seen, because it is not limited by amortized generalization.
* **Nash-prior EM** is worst everywhere: the selected opponent equilibrium is deterministic at 53 % of infosets
  and, as a prior, drives the safe LP toward responses to an opponent the population never resembles; at ε=0 it
  even under-performs the blueprint (regret 0.053 at N=5) by selecting equilibria that do worse than the
  blueprint against the actual opponents.
* All amortized methods (both neural objectives and the bank posterior) share a **regret floor of
  ≈0.10–0.12 chips at ε = 0.10 (≈ 20–25 % of the oracle gain)** that more hands do not remove.

## 11. Results by opponent family (Figure 6, Table R6)

Regret at ε = 0.10 (mean over the 75 test opponents of each family; oracle gain in parentheses) and the
paired decision − reconstruction difference with 95 % CI:

| family (oracle gain) | method | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 |
|---|---|---|---|---|---|---|---|---|
| NASH_LOGIT_PERTURB (0.194) | NEURAL_DECISION | 0.113 | 0.102 | 0.094 | 0.084 | 0.078 | 0.075 | 0.072 |
| | NEURAL_RECON | 0.125 | 0.109 | 0.099 | 0.087 | 0.084 | 0.078 | 0.074 |
| | BANK_POSTERIOR | 0.108 | 0.095 | 0.089 | 0.079 | 0.083 | 0.076 | 0.075 |
| | TABULAR_EM_UNIFORM | 0.142 | 0.136 | 0.132 | 0.119 | 0.111 | 0.102 | 0.090 |
| | decision − recon | −0.013 [−0.019, −0.007] | −0.007 [−0.013, −0.002] | −0.005 [−0.010, −0.001] | −0.003 [−0.009, +0.002] | −0.006 [−0.015, +0.001] | −0.003 [−0.013, +0.006] | −0.002 [−0.012, +0.007] |
| NASH_RANDOM_MIX (0.286) | NEURAL_DECISION | 0.092 | 0.089 | 0.086 | 0.079 | 0.074 | 0.071 | 0.069 |
| | NEURAL_RECON | 0.087 | 0.084 | 0.081 | 0.076 | 0.074 | 0.071 | 0.070 |
| | BANK_POSTERIOR | 0.091 | 0.086 | 0.083 | 0.076 | 0.080 | 0.081 | 0.080 |
| | TABULAR_EM_UNIFORM | 0.098 | 0.096 | 0.094 | 0.087 | 0.081 | 0.074 | 0.064 |
| | decision − recon | +0.005 [−0.002, +0.012] | +0.005 [−0.001, +0.011] | +0.005 [+0.001, +0.010] | +0.003 [−0.003, +0.008] | +0.000 [−0.005, +0.005] | −0.000 [−0.006, +0.005] | −0.001 [−0.008, +0.006] |
| STRUCTURED_CORRELATED (0.799) | NEURAL_DECISION | 0.202 | 0.158 | 0.120 | 0.094 | 0.082 | 0.077 | 0.073 |
| | NEURAL_RECON | 0.224 | 0.178 | 0.139 | 0.108 | 0.098 | 0.094 | 0.091 |
| | BANK_POSTERIOR | 0.184 | 0.148 | 0.127 | 0.117 | 0.113 | 0.111 | 0.107 |
| | TABULAR_EM_UNIFORM | 0.300 | 0.263 | 0.225 | 0.158 | 0.119 | 0.083 | 0.062 |
| | decision − recon | −0.023 [−0.034, −0.010] | −0.020 [−0.029, −0.012] | −0.019 [−0.028, −0.011] | −0.014 [−0.023, −0.005] | −0.016 [−0.027, −0.007] | −0.017 [−0.027, −0.008] | −0.018 [−0.028, −0.008] |
| UNSTRUCTURED_DIRICHLET (0.684) | NEURAL_DECISION | 0.261 | 0.247 | 0.239 | 0.224 | 0.213 | 0.209 | 0.207 |
| | NEURAL_RECON | 0.272 | 0.258 | 0.251 | 0.243 | 0.240 | 0.239 | 0.237 |
| | BANK_POSTERIOR | 0.249 | 0.245 | 0.246 | 0.242 | 0.231 | 0.225 | 0.226 |
| | TABULAR_EM_UNIFORM | 0.244 | 0.225 | 0.210 | 0.176 | 0.144 | 0.115 | 0.083 |
| | decision − recon | −0.010 [−0.020, −0.001] | −0.012 [−0.020, −0.003] | −0.012 [−0.024, −0.001] | −0.019 [−0.037, −0.004] | −0.027 [−0.048, −0.009] | −0.029 [−0.050, −0.008] | −0.030 [−0.053, −0.007] |

N80 at ε = 0.10 is reached only for STRUCTURED_CORRELATED opponents: decision 20 hands, reconstruction 50,
bank posterior 20, uniform EM 100 (N90: only uniform EM, at 500).

Reading.  (i) The decision model's advantage is **not** confined to the structured family: it is largest and
present at every N for STRUCTURED_CORRELATED (−0.02) and UNSTRUCTURED_DIRICHLET (growing from −0.01 to −0.03
with N), present at N ≤ 20 for NASH_LOGIT_PERTURB, and absent (slightly reversed, not significant) for
NASH_RANDOM_MIX.  (ii) For the **unstructured control family** every amortized method plateaus at regret
0.21–0.24 while per-opponent EM keeps improving to 0.083 — exactly the predicted failure mode of population
priors when infosets carry no shared structure.  Yet the amortized methods still recover ≈ 62–70 % of the
oracle gain from 5 hands for this family, because Dirichlet opponents are identified as such almost
immediately (92 % at N=5) and share a family-level exploit (they call and fold at random, so value-betting and
bluffing work against all of them); what cannot be inferred is the per-infoset residual.  (iii) The
near-equilibrium NASH_LOGIT_PERTURB family is where safe exploitation is hardest for everyone (F ≈ 0.2–0.6):
the available gain is small (0.19) and its location depends on fine perturbation details.
(iv) In NASH_RANDOM_MIX the exploitable value comes from the random component, which every method estimates
about equally well; here the two neural objectives are indistinguishable.

## 11b. Independent safety audit (Figure 4, Table R7)

Every one of the 604 800 deployed test strategies (9 methods × 2 400 histories × 7 budgets × 4 ε) and the
6 600 oracle strategies was audited with OpenSpiel's C++ `TabularBestResponse`: **0 LP failures; maximum
audited exploitability minus ε = 7.4e−10 (Nash-prior EM), 6.5e−10 (bank), 5.5e−10 (worst neural run);
99.9 % quantile ≤ 6e−12; 0 strategies exceed ε + 1e−7**.  The fast tree best response agrees with OpenSpiel
to 5.6e−16 on every strategy.  Mean audited exploitability by ε is 0.000 / 0.050 / 0.100 / 0.200 for every
method: the constraint binds at ε > 0 for essentially all strategies, i.e. every method spends its entire
safety budget, so the comparison is genuinely at identical safety.

## 12. Prediction quality: behavioral error and decision-vector error (Figure 5, Table R8)

Standardized g error (NMSE over the 1 074 non-constant coordinates; 1.0 = predicting the training mean) and
uniform-infoset RMS behavioral error on the 300 held-out opponents (mean over 8 streams, then over opponents):

| method | quantity | N=5 | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|---|
| NEURAL_DECISION (3 seeds) | g NMSE | 0.786 | 0.662 | 0.559 | 0.463 | 0.420 | 0.394 | 0.378 |
| NEURAL_RECON (3 seeds) | g NMSE of A ŷ | 0.815 | 0.714 | 0.633 | 0.560 | 0.529 | 0.511 | 0.498 |
| BANK_POSTERIOR | g NMSE | 0.749 | 0.637 | 0.565 | 0.517 | 0.506 | 0.498 | 0.499 |
| TABULAR_EM_UNIFORM | g NMSE | 1.090 | 1.021 | 0.923 | 0.761 | 0.633 | 0.511 | 0.378 |
| TABULAR_EM_NASH | g NMSE | 1.404 | 1.329 | 1.217 | 1.006 | 0.820 | 0.671 | 0.510 |
| NEURAL_RECON (3 seeds) | ‖q̂−q‖ (RMS/infoset) | 0.373 | 0.343 | 0.317 | 0.292 | 0.282 | 0.274 | 0.269 |
| TABULAR_EM_UNIFORM | ‖q̂−q‖ | 0.502 | 0.499 | 0.494 | 0.483 | 0.471 | 0.455 | 0.428 |
| TABULAR_EM_NASH | ‖q̂−q‖ | 0.514 | 0.513 | 0.509 | 0.501 | 0.491 | 0.477 | 0.454 |

Three things stand out.  (i) The decision head has the lowest g error at every N ≥ 10, and its error keeps
falling with N; the reconstruction model's *implied* ĝ = A ŷ is uniformly worse (by 0.05–0.12 NMSE) even
though its behavioral error is far below the tabular estimators'.  Reconstruction therefore does *not*
automatically preserve the decision vector: errors at many low-reach infosets are cheap in the uniform
cross-entropy but are amplified through the realization-plan products and A.  (ii) The train-bank posterior is
the best predictor at N=5 (0.749) — Bayesian reuse of the population is hard to beat with five hands — but it
plateaus at ≈0.50 because a held-out opponent is never in the bank; the neural decision head overtakes it from
N=10 onward.  (iii) The tabular EM estimators start far worse (their uniform / Nash prior at unvisited infosets
is a poor guess) and improve steadily; with a uniform prior EM reaches the decision head's g error only at
N=500.  The Nash-prior EM has a *larger* g error than the uniform-prior EM at every N, because the selected
opponent equilibrium is deterministic at 53 % of rank infosets and pulls the estimate toward extreme actions
that the population rarely plays.

Seed spread is small: g-NMSE at N=500 is 0.372 / 0.381 / 0.380 for the three decision seeds and
0.503 / 0.497 / 0.495 for the three reconstruction seeds.

## 13. Effective latent dimension (Figure 8, Table R9)

Participation ratio of the covariance of the stream-averaged z across the 300 test opponents (a linear,
comparative diagnostic, not an intrinsic-dimension estimate):

| run | N=5 | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|
| NEURAL_DECISION s0 / s1 / s2 | 4.1 / 3.8 / 3.9 | 4.3 / 4.0 / 3.9 | 4.3 / 4.0 / 3.9 | 4.0 / 3.8 / 3.6 | 3.9 / 3.7 / 3.5 | 3.8 / 3.6 / 3.4 | 3.7 / 3.5 / 3.3 |
| NEURAL_RECON s0 / s1 / s2 | 3.2 / 2.8 / 2.8 | 3.1 / 2.7 / 2.7 | 3.0 / 2.7 / 2.7 | 2.9 / 2.7 / 2.7 | 2.9 / 2.7 / 2.8 | 2.8 / 2.7 / 2.7 | 2.7 / 2.7 / 2.7 |

Both encoders use a tiny fraction of the nominal 128 dimensions (the variance is dominated by a few
directions that encode family / exploitability level), and — unlike memory-2 RPS — the reconstruction encoder's
latent is *lower*-dimensional than the decision encoder's here (≈2.7 vs ≈3.5 at N=500).  With d=128 there is
no capacity bottleneck in this experiment, so the RPS capacity phenomenon is not being tested; the numbers are
reported for completeness and should not be over-interpreted.

## 14. Strategic geometry (Figure 7, Table R10)

Computed on the 300 test opponents at ε = 0.10 with 20 000 fixed random pairs; d_resp is the symmetric
response-confusion cost, d_beh the uniform-infoset L2 distance between rank-level policy tables (a
reach-weighted variant uses the reach of each opponent infoset under the learner's blueprint against a
uniform-random opponent); z is averaged over the 8 streams at each N.

* Population structure: ρ(d_beh, d_resp) = 0.686 (reach-weighted 0.728); ρ(‖g−g′‖, d_resp) = 0.689;
  ρ(‖g−g′‖, d_beh) = 0.768.  Behavioral and strategic similarity are correlated but far from identical, so
  behavior-matched pairs with very different confusion costs exist (the matched sets below differ 2.6× in
  d_resp at essentially equal d_beh).
* Behavior-matched strategic separation (mean latent distance of strategically far pairs / strategically near
  pairs within 20 narrow d_beh bins, 6 680 pairs per side, matched d_beh 0.619 vs 0.611):
  true g(q): **1.270** [1.250, 1.289]; decision encoder at N=100: **1.005 / 1.007 / 0.986** (seeds 0/1/2;
  CIs ≈ ±0.014); reconstruction encoder at N=100: **0.894 / 0.875 / 0.879**.  At N=20 and N=500 the picture is
  the same (decision 0.98–1.04, reconstruction 0.87–0.91).
* Correlations at N=100: ρ(d_z, d_beh) ≈ 0.79 for both objectives; ρ(d_z, d_resp) = 0.54–0.56 (decision)
  vs 0.45–0.48 (reconstruction).

Interpretation.  The decision encoder's latent is *somewhat* more aligned with response distance than the
reconstruction encoder's (higher ρ(d_z, d_resp); separation ≈1.0 vs ≈0.88), but neither reproduces the
strategic organization that was the durable signature in memory-2 RPS: once behavioral distance is matched,
the decision latent does not pull strategically far opponents apart (ratio ≈ 1), and the reconstruction latent
actually places strategically far pairs slightly *closer* than strategically near ones (ratio ≈ 0.88).  The
true decision vectors g(q) do carry that structure (1.27), so the target has it and the learned latent does
not.  A plausible reason is that in Leduc the dominant directions of z encode the coarse family/exploitability
structure that is easiest to infer from a few hands, and the finer response-relevant directions live in the
decision head's output rather than in z; this experiment does not test that explanation.

Illustrative pairs (algorithmic selection, see `outputs/eval/test/illustrative_pairs.json`):

* **A — behaviorally distant, response-near** (test opponents 173 & 220, population ids 960 & 1187, both
  STRUCTURED_CORRELATED; d_beh = 0.96, top decile; d_resp = 0.040, bottom decile).  Their tables disagree
  almost completely at round-2 infosets where the opponent holds a paired jack (private J, public J): opponent
  1187 folds them (p ≈ 1.0), opponent 960 calls or raises (p ≈ 0.9–1.0) — e.g. after (raise, raise, call;
  check, raise, raise).  These infosets are rarely reached under the learner's safe response, so the optimal
  safe strategies are nearly interchangeable: deploying 960's oracle strategy against 1187 loses 0.031 of a
  1.183 oracle value, and vice versa 0.050 of 0.966.
* **B — behaviorally similar, response-far** (test opponents 202 & 224, ids 1109 & 1232, both
  STRUCTURED_CORRELATED; d_beh = 0.25, bottom decile; d_resp = 0.36; no pair met the strict top-decile d_resp
  criterion, so the pair with the largest d_resp among bottom-decile d_beh was taken).  Their tables differ at
  a handful of the most-visited infosets: facing the learner's first-round raise, opponent 1109 folds with
  almost any card (J 0.98, Q 0.75, K 0.79), opponent 1232 calls or re-raises (fold ≈ 0).  One tendency at
  high-reach infosets changes the safe response completely (raise every hand against 1109): deploying 1109's
  strategy against 1232 forgoes 0.42 of a 1.79 oracle value, and 1232's against 1109 forgoes 0.29 of 1.19.


## 14b. Family identifiability from short histories (Table R11)

MAP family accuracy of the train-bank posterior on the test histories (chance = 0.25):

| | N=5 | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|
| all families | 0.76 | 0.83 | 0.86 | 0.90 | 0.91 | 0.90 | 0.90 |
| NASH_LOGIT_PERTURB | 0.96 | 0.96 | 0.96 | 0.96 | 0.97 | 0.95 | 0.92 |
| NASH_RANDOM_MIX | 0.27 | 0.47 | 0.65 | 0.77 | 0.83 | 0.85 | 0.85 |
| STRUCTURED_CORRELATED | 0.89 | 0.93 | 0.93 | 0.92 | 0.92 | 0.92 | 0.92 |
| UNSTRUCTURED_DIRICHLET | 0.92 | 0.97 | 0.91 | 0.93 | 0.93 | 0.89 | 0.89 |

Five hands already identify the family in three of four cases; the two Nash-anchored families are confusable at
small N.  This is the relevant caveat for the sample-efficiency result: part of what any population-trained
method "learns" from a few hands is which family the opponent belongs to.

## 15. Bugs discovered, deviations from the plan, limitations

**Bugs found and fixed during development (none affect reported results).**
1. OpenSpiel's Leduc information sets distinguish suits (468 per player), which would have made the
   "unstructured" opponent family unstructured across strategically identical infosets and would have made the
   rank-canonical tokenizer unable to identify which of two physically different infosets the opponent was in.
   Fixed by defining opponent policies on the 144 rank infosets (tied across suits) and symmetrizing the
   blueprints (§2).  OpenSpiel's own `suit_isomorphism=True` variant was tried and rejected: it changes the
   chance structure and crashed on card-id actions.
2. A test compared post-settlement OpenSpiel money at terminal states with pre-settlement contributions
   (test bug; the tree was correct).
3. `torch.as_tensor` on the numpy token table shared memory, so a test that corrupted padded events corrupted
   the global table; encoder buffers are now copies.
4. Evaluation workers set `OMP_NUM_THREADS` after numpy had been imported, so four workers spawned 16 BLAS
   threads and ran 3× slower; the variables are now set before spawning (results unaffected).
5. The first draw of the NASH_LOGIT_PERTURB family (τ ~ U[0.3, 2.5] with an un-normalized feature term) had
   almost no near-equilibrium members (minimum exploitability 0.26, median 2.2 chips); it was discarded before
   any model saw it and regenerated with a unit-scaled feature term and τ ~ LogU[0.15, 2.0].
6. The analysis loader assumed every method's solve arrays cover all histories; it now truncates consistently.

**Deviations from the task specification, all decided before training and documented.**
* Opponent policies and the reconstruction target are defined on 144 rank-level infosets rather than the 468
  physical OpenSpiel infosets (see bug 1).  All game computations (sequence form, LP, audit, simulation) still
  use the full physical game.
* The Transformer layers use fused scaled-dot-product attention without attention-probability dropout (it
  cost 0.9 s per step on CPU); dropout 0.1 is applied to the residual and feed-forward branches.
* The hand encoder is evaluated once per distinct observation type per batch (792 types) and gathered;
  this is the identical function, with dropout masks shared between identical hands.
* N is sampled once per batch (all 32 samples share N) rather than once per sample, to avoid padding; the
  encoder supports per-sample padding and its invariance is tested.
* Blueprints are suit-symmetrized equilibria selected by a documented rule (§2).
* Two tabular-EM variants (uniform and Nash prior) are reported instead of one.
* The classical baselines' LP solves ran at low OS priority concurrently with training (no effect on results).
* Effective-dimension and geometry statistics use z averaged over the 8 test streams, as specified, at every N
  (not only N=100).

**Limitations.**
* Fixed seats (learner always first to act) and a single learner blueprint; the observation policy strongly
  shapes which opponent infosets are ever observed (only 401 of 792 observation types occur).
* One opponent population with four synthetic families; results at small N partly reflect family
  identification (§14b).  No CFR-checkpoint or human-derived opponents.
* Training budget: 6 000 steps; the decision heads' validation loss had plateaued, the reconstruction
  models' cross-entropy was still decreasing slowly (0.6390 → 0.6367 over the last 2 000 steps), so the
  reconstruction results are a lower bound on what a longer budget would give.  No hyperparameter search was
  run for either method; the shared configuration was chosen once.
* Leduc is small: exact LPs and exact audits are possible, but the 1 093-dimensional g and 144-infoset q are
  both tiny compared with real poker, and the observed tokens never exceed 12 events.
* The participation ratio is a linear diagnostic; the geometry analysis uses one ε (0.10) and one pair sample.
* Bootstrap CIs treat the 300 test opponents as the sample and the 8 streams as within-opponent replication;
  the population generator itself is a single draw (no CI over populations).
* All runs are CPU-only; wall-clock numbers reflect a 4-core container.

## 16. Conclusions

**Strongest conclusion actually supported.**  On held-out Leduc opponents, at exactly identical and
independently certified safety, directly learning the decision vector `g(q)` yields lower safe response regret
than reconstructing the opponent's full policy and deriving `g` from it, at every observation budget from 5 to
500 hands and at every ε > 0 (paired 95 % CIs exclude zero throughout; regret 4–12 % lower; area under the
regret curve 6–8 % lower).  Expressed in hands, the reconstruction model needs about twice as many hands to
match the decision model at N ≤ 20, and does not match it within 500 hands beyond N = 50.  The advantage
shows up directly in the accuracy of the decision vector: the decision head's ĝ has 20–25 % lower
standardized error than the ĝ implied by the reconstruction model at every N ≥ 10, even though the
reconstruction model's behavioral error is far below that of any tabular estimator.  Reconstruction does not
automatically preserve what the safe solver needs.

**What the result is not.**  The effect is real but modest in absolute terms: 0.005–0.020 chips per hand,
2–3 % of the oracle-safe gain at ε = 0.10.  It does *not* take the form hypothesized in §1: there is no
low-N advantage that reconstruction later erases.  Instead both amortized methods plateau from N ≈ 100 at a
regret floor (≈ 20–25 % of the oracle gain) that more observations do not remove, and the decision model's
advantage grows with N.  The "reconstruction eventually catches up" phenomenon of the RPS capacity study does
appear in Leduc, but between *amortized* population inference and *per-opponent* estimation: uniform-prior
tabular EM overtakes both neural models at N ≈ 100–200 and is the best method at N = 500, while the exact
train-bank posterior is the best method at N ≤ 10.  A practical system would therefore combine a decision-focused
amortized prior for the first tens of hands with per-opponent estimation thereafter.

**Negative results.**  (1) N80 and N90 are not reached by any method within 500 hands, so the pre-registered
threshold-ratio statistic is empty.  (2) Neither latent space reproduces the strategic geometry seen in
memory-2 RPS: at matched behavioral distance the decision latent does not separate strategically far from near
opponents (ratio ≈ 1.00) and the reconstruction latent slightly inverts the ordering (≈ 0.88), while the true
`g` vectors do separate them (1.27).  (3) The effective latent dimension is 3–4 for both objectives, with
reconstruction *lower* than decision, the opposite of RPS; with d = 128 no bottleneck was being tested.
(4) For the unstructured control family every amortized method plateaus far above per-opponent EM, as expected.

**Outcome classification (task §32).**  Closest to A in direction (decision wins) but without the crossover;
partly B (the advantage is family-dependent, largest for structured and unstructured opponents, absent for the
Nash–random mixture); E is rejected in its strong form (g is more compressible than q: 29 vs 83 PCA dimensions
for 90 % variance) but the compression did not translate into a lower-dimensional *learned* latent.

**What the experiment does not establish.**  It does not show that decision-sufficient representations are
fundamentally more sample-efficient than behavioral ones: the reconstruction models were still improving
slowly at the end of the fixed budget, no hyperparameters were tuned for either method, a single population
and a single blueprint were used, and the tabular estimator's late-N superiority shows that the amortized
floor, not the objective, is the binding constraint at large N.  It does not establish anything about larger
poker games, about active observation policies, or about non-stationary opponents.  It does establish an
exact, audited evaluation pipeline (sequence form, ε-safe LP, OpenSpiel audit, hidden-card-marginalized
baselines, leakage-tested tokenization) on which those questions can be asked.


## 17. Runtime, reproduction, artifacts

**Hardware / software.** 4-core Linux container, no GPU, 15 GB RAM; Python 3.11, open_spiel (pyspiel),
torch 2.14 (CPU), numpy 2.4, scipy 1.17, highspy 1.15.  Everything below ran inside one session
(2026-09-22 17:30 → 2026-09-23 00:11 UTC, ≈ 6.7 h including development and tests).

| stage | wall time | notes |
|---|---|---|
| tests (`pytest leduc_decision_repr/tests`, 31 tests) | ~1 min | all passing |
| population + safe oracle (`build_population`) | 535 s | 6 600 LPs + audits |
| observation streams (`data.datasets`) | 16 s | 7 800 streams, 3.9 M hands |
| training, 6 runs × 6 000 steps | 42 341 core-s; 3.5 h wall in two rounds | 4 × 1-thread then 2 × 2-thread |
| test prediction stage (EM, bank, 6 neural runs) | 586 s | |
| test LP + audit stage, 9 methods × 67 200 | classical 3 × 39 min (low priority, overlapped with training); neural 6 × 19 min = 1.9 h wall, 4 workers | 604 800 LPs, 0 failures |
| analysis + figures | 48 s | |

**Reproduce** (from the repository root):
```bash
pip install open_spiel numpy scipy highspy torch matplotlib pandas pytest cvxpy ecos
python -m pytest leduc_decision_repr/tests -q
python -m leduc_decision_repr.build_population
python -m leduc_decision_repr.data.datasets
leduc_decision_repr/run_training.sh                       # writes outputs/runs/{decision,recon}_s{0,1,2}
R=leduc_decision_repr/outputs/runs
python -m leduc_decision_repr.evaluate predict --split test --runs \
  NEURAL_DECISION_s0=$R/decision_s0,NEURAL_DECISION_s1=$R/decision_s1,NEURAL_DECISION_s2=$R/decision_s2,\
NEURAL_RECON_s0=$R/recon_s0,NEURAL_RECON_s1=$R/recon_s1,NEURAL_RECON_s2=$R/recon_s2
python -m leduc_decision_repr.evaluate solve --split test --workers 4
python -m leduc_decision_repr.analyze
python -m leduc_decision_repr.report_tables leduc_decision_repr/outputs/eval/test > leduc_decision_repr/outputs/eval/test/report_tables.md
```
(`run_pipeline_after_training.sh` chains the last four steps automatically after training.)

**Artifacts.**
* Figures: `leduc_decision_repr/figures/fig{1..11}_*.png|.pdf` with the plotted data in `*_data.json`.
* Raw tables: `leduc_decision_repr/outputs/eval/test/` — `summary.json` (all means, CIs, thresholds, AUCs,
  safety, prediction errors, latent dimensions), `per_opponent_metrics.csv` (every method × opponent × N × ε),
  `geometry.json` + `geometry_raw.npz`, `precondition.json`, `illustrative_pairs.json`, `report_tables.md`,
  `solve_<method>.npz` (u, e_fast, e_os, ok per history × N × ε), `pred_<method>.npz` (g/q errors, z),
  `ghat_<method>.npy`, `solve_summary.json`, `predict_meta.json`, `tokenized_examples.txt`.
* Population / datasets: `outputs/population_4c566ff60c/` (`population.npz`: policies, y_q, g(q), oracle
  strategies and values, blueprints, v*; `params.pkl`; `meta.json`), `outputs/datasets_a630e4d419/`.
* Runs: `outputs/runs/<method>_s<seed>/` (`config.json`, `log.jsonl`, `best.pt`, `last.pt`, `result.json`).
* Configuration: `configs/frozen_v1.json`; plan: `PLAN.md`.


## Appendix A. Full result tables (generated by `report_tables.py`)

### Table R1. Safe response regret (chips/hand, mean over 300 held-out opponents, 8 streams each; 95% paired-bootstrap CI)

**ε = 0.0** (Nash regret = oracle gain = 0.005 [0.004, 0.007])

| method | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 | AUC(logN) |
|---|---|---|---|---|---|---|---|---|
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

**ε = 0.05** (Nash regret = oracle gain = 0.347 [0.314, 0.383])

| method | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 | AUC(logN) |
|---|---|---|---|---|---|---|---|---|
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

**ε = 0.1** (Nash regret = oracle gain = 0.491 [0.445, 0.537])

| method | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 | AUC(logN) |
|---|---|---|---|---|---|---|---|---|
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

**ε = 0.2** (Nash regret = oracle gain = 0.684 [0.629, 0.744])

| method | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 | AUC(logN) |
|---|---|---|---|---|---|---|---|---|
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

### Table R2. Fraction of oracle-safe gain recovered F (opponents with gain ≥ 0.02 only)

**ε = 0.05** (291 of 300 opponents with valid denominator)

| method | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 | N50 | N80 | N90 | AUC_F |
|---|---|---|---|---|---|---|---|---|---|---|---|
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

**ε = 0.1** (299 of 300 opponents with valid denominator)

| method | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 | N50 | N80 | N90 | AUC_F |
|---|---|---|---|---|---|---|---|---|---|---|---|
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

**ε = 0.2** (300 of 300 opponents with valid denominator)

| method | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 | N50 | N80 | N90 | AUC_F |
|---|---|---|---|---|---|---|---|---|---|---|---|
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

### Table R3. Bootstrap distribution of N-thresholds (median [2.5%, 97.5%] over 2000 opponent resamples; 1000 = not reached by 500)

| ε | method | N50 | N80 | N90 |
|---|---|---|---|---|
| 0.05 | NEURAL_DECISION | 5 [5, 5] | 1000 [1000, 1000] | 1000 [1000, 1000] |
| 0.05 | NEURAL_RECON | 5 [5, 10] | 1000 [1000, 1000] | 1000 [1000, 1000] |
| 0.05 | BANK_POSTERIOR | 5 [5, 5] | 1000 [1000, 1000] | 1000 [1000, 1000] |
| 0.05 | TABULAR_EM_NASH | 500 [200, 500] | 1000 [1000, 1000] | 1000 [1000, 1000] |
| 0.05 | TABULAR_EM_UNIFORM | 50 [20, 50] | 1000 [1000, 1000] | 1000 [1000, 1000] |
| 0.1 | NEURAL_DECISION | 5 [5, 5] | 1000 [1000, 1000] | 1000 [1000, 1000] |
| 0.1 | NEURAL_RECON | 5 [5, 10] | 1000 [1000, 1000] | 1000 [1000, 1000] |
| 0.1 | BANK_POSTERIOR | 5 [5, 5] | 1000 [1000, 1000] | 1000 [1000, 1000] |
| 0.1 | TABULAR_EM_NASH | 500 [200, 500] | 1000 [1000, 1000] | 1000 [1000, 1000] |
| 0.1 | TABULAR_EM_UNIFORM | 100 [50, 200] | 1000 [1000, 1000] | 1000 [1000, 1000] |
| 0.2 | NEURAL_DECISION | 5 [5, 10] | 1000 [1000, 1000] | 1000 [1000, 1000] |
| 0.2 | NEURAL_RECON | 10 [5, 10] | 1000 [1000, 1000] | 1000 [1000, 1000] |
| 0.2 | BANK_POSTERIOR | 5 [5, 5] | 1000 [1000, 1000] | 1000 [1000, 1000] |
| 0.2 | TABULAR_EM_NASH | 500 [200, 500] | 1000 [1000, 1000] | 1000 [1000, 1000] |
| 0.2 | TABULAR_EM_UNIFORM | 100 [50, 200] | 1000 [1000, 1000] | 1000 [1000, 1000] |

### Table R4. Paired differences (decision − reconstruction), mean over opponents with 95% CI

| ε | quantity | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 |
|---|---|---|---|---|---|---|---|---|
| 0.0 | regret diff | +0.000 [-0.000,+0.000] | +0.000 [+0.000,+0.001] | +0.001 [+0.001,+0.001] | +0.001 [+0.000,+0.001] | +0.001 [+0.000,+0.001] | +0.001 [+0.000,+0.001] | +0.001 [+0.000,+0.001] |
| 0.05 | regret diff | -0.006 [-0.009,-0.002] | -0.005 [-0.008,-0.003] | -0.005 [-0.008,-0.002] | -0.006 [-0.010,-0.002] | -0.008 [-0.012,-0.004] | -0.007 [-0.012,-0.003] | -0.008 [-0.013,-0.003] |
| 0.05 | F diff | +0.02 [+0.01,+0.03] | +0.01 [-0.00,+0.02] | +0.00 [-0.01,+0.01] | -0.00 [-0.01,+0.01] | +0.00 [-0.01,+0.01] | -0.00 [-0.01,+0.01] | +0.00 [-0.01,+0.01] |
| 0.1 | regret diff | -0.010 [-0.015,-0.005] | -0.009 [-0.012,-0.005] | -0.008 [-0.012,-0.004] | -0.008 [-0.013,-0.003] | -0.012 [-0.018,-0.007] | -0.012 [-0.019,-0.006] | -0.013 [-0.020,-0.006] |
| 0.1 | F diff | +0.03 [+0.02,+0.04] | +0.01 [+0.00,+0.02] | +0.01 [-0.00,+0.02] | +0.00 [-0.01,+0.01] | +0.00 [-0.01,+0.02] | -0.00 [-0.02,+0.01] | -0.00 [-0.02,+0.01] |
| 0.2 | regret diff | -0.013 [-0.020,-0.007] | -0.014 [-0.020,-0.009] | -0.012 [-0.018,-0.006] | -0.014 [-0.022,-0.007] | -0.019 [-0.027,-0.011] | -0.018 [-0.028,-0.010] | -0.020 [-0.030,-0.010] |
| 0.2 | F diff | +0.02 [+0.01,+0.03] | +0.01 [+0.00,+0.02] | +0.01 [-0.00,+0.01] | +0.01 [-0.00,+0.02] | +0.01 [-0.01,+0.02] | -0.00 [-0.01,+0.01] | -0.00 [-0.02,+0.02] |

### Table R5. Per-seed neural results (regret at ε=0.1)

| run | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 | best step |
|---|---|---|---|---|---|---|---|---|
| NEURAL_DECISION_s0 | 0.167 | 0.148 | 0.133 | 0.120 | 0.111 | 0.107 | 0.103 | |
| NEURAL_DECISION_s1 | 0.167 | 0.150 | 0.135 | 0.121 | 0.112 | 0.108 | 0.105 | |
| NEURAL_DECISION_s2 | 0.167 | 0.149 | 0.135 | 0.119 | 0.112 | 0.109 | 0.106 | |
| NEURAL_RECON_s0 | 0.177 | 0.156 | 0.142 | 0.128 | 0.123 | 0.120 | 0.118 | |
| NEURAL_RECON_s1 | 0.179 | 0.159 | 0.144 | 0.130 | 0.125 | 0.121 | 0.118 | |
| NEURAL_RECON_s2 | 0.176 | 0.157 | 0.142 | 0.127 | 0.123 | 0.120 | 0.118 | |

### Table R6. Per-family regret at ε=0.1 (mean) and N80

**NASH_LOGIT_PERTURB** (oracle gain at ε=0.1: 0.194)

| method | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 | N50 | N80 | N90 |
|---|---|---|---|---|---|---|---|---|---|---|
| NEURAL_DECISION | 0.113 | 0.102 | 0.094 | 0.084 | 0.078 | 0.075 | 0.072 | 500 | >500 | >500 |
| NEURAL_RECON | 0.125 | 0.109 | 0.099 | 0.087 | 0.084 | 0.078 | 0.074 | 200 | >500 | >500 |
| BANK_POSTERIOR | 0.108 | 0.095 | 0.089 | 0.079 | 0.083 | 0.076 | 0.075 | 50 | >500 | >500 |
| TABULAR_EM_NASH | 0.141 | 0.133 | 0.120 | 0.108 | 0.096 | 0.086 | 0.071 | >500 | >500 | >500 |
| TABULAR_EM_UNIFORM | 0.142 | 0.136 | 0.132 | 0.119 | 0.111 | 0.102 | 0.090 | >500 | >500 | >500 |

**NASH_RANDOM_MIX** (oracle gain at ε=0.1: 0.286)

| method | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 | N50 | N80 | N90 |
|---|---|---|---|---|---|---|---|---|---|---|
| NEURAL_DECISION | 0.092 | 0.089 | 0.086 | 0.079 | 0.074 | 0.071 | 0.069 | 5 | >500 | >500 |
| NEURAL_RECON | 0.087 | 0.084 | 0.081 | 0.076 | 0.074 | 0.071 | 0.070 | 5 | >500 | >500 |
| BANK_POSTERIOR | 0.091 | 0.086 | 0.083 | 0.076 | 0.080 | 0.081 | 0.080 | 5 | >500 | >500 |
| TABULAR_EM_NASH | 0.252 | 0.232 | 0.215 | 0.184 | 0.163 | 0.141 | 0.114 | 500 | >500 | >500 |
| TABULAR_EM_UNIFORM | 0.098 | 0.096 | 0.094 | 0.087 | 0.081 | 0.074 | 0.064 | 5 | >500 | >500 |

**STRUCTURED_CORRELATED** (oracle gain at ε=0.1: 0.799)

| method | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 | N50 | N80 | N90 |
|---|---|---|---|---|---|---|---|---|---|---|
| NEURAL_DECISION | 0.202 | 0.158 | 0.120 | 0.094 | 0.082 | 0.077 | 0.073 | 5 | 20 | >500 |
| NEURAL_RECON | 0.224 | 0.178 | 0.139 | 0.108 | 0.098 | 0.094 | 0.091 | 5 | 50 | >500 |
| BANK_POSTERIOR | 0.184 | 0.148 | 0.127 | 0.117 | 0.113 | 0.111 | 0.107 | 5 | 20 | >500 |
| TABULAR_EM_NASH | 0.646 | 0.567 | 0.482 | 0.354 | 0.271 | 0.214 | 0.177 | 100 | >500 | >500 |
| TABULAR_EM_UNIFORM | 0.300 | 0.263 | 0.225 | 0.158 | 0.119 | 0.083 | 0.062 | 5 | 100 | 500 |

**UNSTRUCTURED_DIRICHLET** (oracle gain at ε=0.1: 0.684)

| method | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 | N50 | N80 | N90 |
|---|---|---|---|---|---|---|---|---|---|---|
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

### Table R9. Effective latent dimension (participation ratio of z across test opponents) vs N

| run | N=5 | N=10 | N=20 | N=50 | N=100 | N=200 | N=500 |
|---|---|---|---|---|---|---|---|
| NEURAL_DECISION_s0 | 4.1 | 4.3 | 4.3 | 4.0 | 3.9 | 3.8 | 3.7 |
| NEURAL_DECISION_s1 | 3.8 | 4.0 | 4.0 | 3.8 | 3.7 | 3.6 | 3.5 |
| NEURAL_DECISION_s2 | 3.9 | 3.9 | 3.9 | 3.6 | 3.5 | 3.4 | 3.3 |
| NEURAL_RECON_s0 | 3.2 | 3.1 | 3.0 | 2.9 | 2.9 | 2.8 | 2.7 |
| NEURAL_RECON_s1 | 2.8 | 2.7 | 2.7 | 2.7 | 2.7 | 2.7 | 2.7 |
| NEURAL_RECON_s2 | 2.8 | 2.7 | 2.7 | 2.7 | 2.8 | 2.7 | 2.7 |

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
