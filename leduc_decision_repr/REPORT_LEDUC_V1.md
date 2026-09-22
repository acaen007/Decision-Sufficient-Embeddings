# REPORT_LEDUC_V1 — Decision-sufficient opponent representations in two-player Leduc poker

*Sample efficiency of decision-focused vs. full-behavioral opponent representations at identical, exactly certified safety.*

> Status: sections 1–9 describe the frozen design; sections 10–16 report results from the single held-out test evaluation.

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
