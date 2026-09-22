# leduc_decision_repr — decision-sufficient opponent representations in Leduc poker

Controlled sample-efficiency experiment: does directly learning the opponent-dependent decision
vector `g(q) = A y_q` (what a certified ε-safe response needs) reduce the number of observed
Leduc hands required for safe exploitation, compared with first reconstructing the opponent's
full behavioral policy `q` and deriving `g` from it?  Both representations feed the *same*
exact ε-safe sequence-form LP; every deployed strategy is audited with OpenSpiel's C++ best
response.  Full write-up: `REPORT_LEDUC_V1.md`.

## Layout
```
leduc_decision_repr/
  PLAN.md                     Phase-0 implementation plan and design decisions
  REPORT_LEDUC_V1.md          final report
  configs/frozen_v1.json      frozen experimental configuration
  common.py                   paths, constants (eps, N budgets), config hashing
  game/leduc_tree.py          enumeration of the OpenSpiel Leduc tree into arrays
  game/sequence_form.py       E, F, e, f, A; realization <-> behavioral; exact tree best response
  game/safe_lp.py             game-value LP, eps-safe LP (HiGHS), OpenSpiel C++ audit
  game/symmetry.py            rank-level infosets and suit-permutation symmetrization
  game/policy_utils.py        infoset features, participation ratio, distances, rank-policy -> g
  data/opponents.py           four opponent families, stratified split
  data/simulator.py           vectorized hand simulator (learner = Nash blueprint)
  data/tokenizer.py           structured event tokenizer (player-0 observables only)
  data/datasets.py            fixed observation streams (train 4 / val 4 / test 8 x 500 hands)
  baselines/likelihood.py     hidden-card-marginalized hand likelihoods
  baselines/tabular_em.py     tabular EM (uniform / Nash prior)
  baselines/bank_posterior.py train-bank Bayesian posterior
  models/encoder.py           hierarchical Transformer (event -> hand -> history -> z)
  models/heads.py             reconstruction decoder and decision head
  models/torch_g.py           differentiable rank-policy -> g map
  build_population.py         Phase 3
  train.py                    Phase 13
  evaluate.py                 Phase 14 (predict + solve stages)
  analyze.py                  Phase 15-16 (metrics, geometry, figures)
  tests/                      game-theoretic, tokenizer/leakage, baseline and model tests
  outputs/                    populations, datasets, runs, eval tables (large arrays git-ignored)
  figures/                    figures (PNG + PDF) with the plotted data as JSON
```

## Reproduce
```bash
pip install open_spiel numpy scipy highspy torch matplotlib pandas pytest cvxpy ecos
cd <repo root>
python -m pytest leduc_decision_repr/tests -q                 # all validation tests
python -m leduc_decision_repr.build_population                # opponents, g(q), safe oracle  (~9 min)
python -m leduc_decision_repr.data.datasets                   # observation streams          (<1 min)
leduc_decision_repr/run_training.sh                           # 3 seeds x 2 objectives       (~2 h, 4 CPU)
python -m leduc_decision_repr.evaluate predict --split test \
   --runs NEURAL_DECISION_s0=leduc_decision_repr/outputs/runs/decision_s0,...  # see report for the full list
python -m leduc_decision_repr.evaluate solve --split test --workers 4         # ~3 h, 4 CPU
python -m leduc_decision_repr.analyze                          # summary.json, geometry, figures
```
