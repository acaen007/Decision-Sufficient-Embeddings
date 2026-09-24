# Decision-Sufficient-Embeddings

hello
## Leduc pipeline (`leduc_decision_repr/`)

Reports, in order: `leduc_decision_repr/REPORT_LEDUC_V1.md` (decision vs reconstruction representations, exact ε-safe
exploitation), `REPORT_LEDUC_V2.md` (SAFE_REGRET: differentiable regularized safe solver), `REPORT_LEDUC_V3.md`
(pre-registered follow-ups T1–T5: parameter matching and weighted reconstruction, covariance gap and censoring,
ε-rank oracle curve, SPO+, empirical-Bayes hybrids).  Figures are in `leduc_decision_repr/figures/`, raw evaluation
arrays and run logs in `leduc_decision_repr/outputs/`.

V3 entry points: `v3_scheduler.py` (queue of training runs + evaluation), `t3_eps_rank.py`, `t2_covariance_gap.py`,
`t5_hybrids.py`, `t4_val_select.py`, `v3_analysis.py {t1,t2,t4,t5}`, `v3_figures.py {t1,t2,t4,t5}`, `v3_audit_table.py`.
