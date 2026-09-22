# Leduc decision-sufficient representation experiment — implementation plan (Phase 0)

## Repository state at start
The repository contained only a README stub. Everything below is built from scratch under
`leduc_decision_repr/`. Environment: Python 3.11, open_spiel 1.x (pyspiel), torch (CPU only),
scipy/HiGHS + highspy, 4 CPU cores, 15 GB RAM, no GPU. Compute budgets below are chosen for this.

## Fixed design decisions (documented choices)
1. **Game**: `pyspiel.load_game("leduc_poker")` (2 players, 6 cards = 3 ranks x 2 suits, ante 1,
   raise sizes 2 / 4, max 2 raises per round). Rank = card // 2 (verified: 0,1 -> J, 2,3 -> Q, 4,5 -> K).
   468 information sets per player, 9457 histories, 5520 terminals.
2. **Learner = player 0** (acts first in each round; OpenSpiel Leduc has a fixed starting player).
   Positions do not alternate across hands: every observed hand is an independent Leduc game with
   the learner in seat 0. This is the simplest faithful setting; documented as a limitation.
3. **Game representation**: the whole OpenSpiel tree is enumerated once into arrays
   (`game/leduc_tree.py`). All later code (sequence form, simulator, tokenizer, EM baseline) is
   table-driven from that enumeration, so it is fast and deterministic. OpenSpiel itself is used
   for (a) the enumeration and (b) independent validation (C++ TabularBestResponse, OpenSpiel's
   own sequence-form LP, CFR).
4. **Sequence form** (`game/sequence_form.py`): sequences = empty + one per (infoset, legal action);
   `E x = e`, `F y = f`, payoff `A[s0, s1] = sum_z chance(z) u0(z)` over terminals with last
   sequences (s0, s1). `g(q) = A y_q`.
5. **Safe LP** (`game/safe_lp.py`): `max g^T x` s.t. `E x = e`, `x >= 0`, `F^T v - A^T x <= 0`,
   `v_root >= v* - eps` (dual of the opponent best-response LP `min_y x^T A y`, `F y = f`, `y >= 0`).
   Solved with HiGHS (highspy, dual simplex, tightened tolerances, warm-started model reuse).
   Every deployed strategy is converted to a behavioral policy (uniform at self-unreachable
   infosets) and that policy is the object that is evaluated and audited.
6. **Independent audit**: `pyspiel.TabularBestResponse(game, 1, policy).value("")` gives the
   opponent's best-response value; `e(x) = v* + BR_1(x)`. Required: `e(x) <= eps + 1e-7`.
   A second, fast exact tree best response (derived from the enumerated tree, not from A) is also
   computed for every strategy and cross-checked against the OpenSpiel value.
7. **Equilibrium selection** (blueprints are not unique): v* from the game-value LP; the learner's
   Nash blueprint is the eps=0 safe-LP solution that maximizes value against a *uniform random*
   opponent (a population-independent, deterministic selection rule). The opponent's Nash
   (used by the Nash-based families) is selected symmetrically. Unreachable infosets: uniform.
8. **Opponent population**: 1650 policies, 4 families x ~412, stratified split 1200/150/300
   (300/37-38/75 per family). Families: NASH_LOGIT_PERTURB, NASH_RANDOM_MIX, STRUCTURED_CORRELATED,
   UNSTRUCTURED_DIRICHLET. Generator parameters and seeds saved.
9. **Observation streams**: learner plays the Nash blueprint, opponent plays q; one stream = 500
   completed hands; prefixes give N in {5,10,20,50,100,200,500}. train 4 / val 4 / test 8 streams.
   Stream seed = f(split, opponent id, stream id). A hand is stored as its terminal-history index;
   the tokenizer maps a terminal history to the player-0 *observation* event sequence.
10. **Tokenizer**: event tokens HAND_START / ACTION / PUBLIC_CARD / SHOWDOWN / HAND_END with
    categorical + numeric fields; the opponent's private rank appears only in a SHOWDOWN token.
    Distinct player-0 observation sequences are deduplicated into "hand observation types"; the
    hand encoder is evaluated once per distinct type per batch and gathered (mathematically the
    same model, far cheaper on CPU; dropout masks are shared between identical hands).
11. **Models**: shared hierarchical Transformer (event -> hand CLS -> history CLS -> z in R^128),
    reconstruction decoder (z + infoset embedding/features -> masked softmax over 3 actions for all
    468 opponent infosets), decision head (z -> standardized g, dims of near-zero train variance
    excluded from the loss and predicted at their train mean).
12. **Training**: one model per objective, N sampled per batch from the 7 budgets (uniform),
    AdamW 3e-4, weight decay 0.01, grad clip 1.0, validation-loss checkpoint selection, 3 seeds.
13. **Baselines**: NASH, ORACLE_SAFE, TABULAR_EM (uniform Dirichlet pseudocount prior, and a
    Nash-mean prior variant), TRAIN_BANK_POSTERIOR (exact marginalized likelihoods).
14. **Evaluation**: 300 test opponents x 8 streams x 7 N x 4 eps per method; per-opponent averaging
    over streams before statistics; paired bootstrap over opponents.

## Execution order
Phases 0-16 exactly as in the task statement; tests live in `leduc_decision_repr/tests/` and are
run with `pytest` before data generation and before training.
