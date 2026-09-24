#!/bin/bash
# Post hoc pilot of the recommended next experiment: learned-prior EM with the Jacobian-weighted RECON-889k q-hat (seed 0).
cd /home/user/Decision-Sufficient-Embeddings
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
R=leduc_decision_repr/outputs/runs_v3
nice -n 12 python3 -m leduc_decision_repr.t5_hybrids --recon_runs $R/recjac889k_s0 --suffix _JAC > leduc_decision_repr/outputs/t5_prior_em_jac.log 2>&1
while pgrep -f "evaluate predic[t]" > /dev/null; do sleep 30; done
python3 -m leduc_decision_repr.merge_hyb_meta
nice -n 8 python3 -m leduc_decision_repr.evaluate solve --methods HYB_PRIOR_EM_JAC --workers 2 --eps_idx 2 > leduc_decision_repr/outputs/t5_solve_priorjac.log 2>&1
echo "T5 PRIORJAC DONE $(date -u)" >> leduc_decision_repr/outputs/t5_solve_priorjac.log
