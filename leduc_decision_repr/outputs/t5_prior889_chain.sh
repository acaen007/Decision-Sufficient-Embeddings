#!/bin/bash
# T5c extra: learned-prior EM with the RECON-889k models (3 seeds), after the scheduler has evaluated them.
cd /home/user/Decision-Sufficient-Embeddings
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
R=leduc_decision_repr/outputs/runs_v3
until [ -f $R/rec889k_s0/EVALUATED ] && [ -f $R/rec889k_s1/EVALUATED ] && [ -f $R/rec889k_s2/EVALUATED ]; do sleep 120; done
nice -n 12 python3 -m leduc_decision_repr.t5_hybrids --recon_runs $R/rec889k_s0,$R/rec889k_s1,$R/rec889k_s2 --suffix _889K > leduc_decision_repr/outputs/t5_prior_em_889k.log 2>&1
while pgrep -f "evaluate predic[t]" > /dev/null; do sleep 30; done
python3 -m leduc_decision_repr.merge_hyb_meta
nice -n 8 python3 -m leduc_decision_repr.evaluate solve --methods HYB_PRIOR_EM_889K --workers 2 --eps_idx 2 > leduc_decision_repr/outputs/t5_solve_prior889.log 2>&1
echo "T5 PRIOR889 DONE $(date -u)" >> leduc_decision_repr/outputs/t5_solve_prior889.log
