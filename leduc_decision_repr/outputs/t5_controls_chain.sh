#!/bin/bash
# T5c controls: single-seed prior-EM (ensemble confound) and ensemble-recon without EM.
cd /home/user/Decision-Sufficient-Embeddings
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nice -n 12 python3 -m leduc_decision_repr.t5_hybrids --recon_runs leduc_decision_repr/outputs/runs/recon_s0 --suffix _S0 > leduc_decision_repr/outputs/t5_prior_em_s0.log 2>&1
nice -n 12 python3 -m leduc_decision_repr.t5_hybrids --recon_runs leduc_decision_repr/outputs/runs/recon_s0,leduc_decision_repr/outputs/runs/recon_s1,leduc_decision_repr/outputs/runs/recon_s2 --ensemble_only > leduc_decision_repr/outputs/t5_ensrec.log 2>&1
while pgrep -f "evaluate predic[t]" > /dev/null; do sleep 30; done
python3 -m leduc_decision_repr.merge_hyb_meta
nice -n 8 python3 -m leduc_decision_repr.evaluate solve --methods HYB_PRIOR_EM_S0,HYB_ENSREC --workers 2 --eps_idx 2 > leduc_decision_repr/outputs/t5_solve_controls.log 2>&1
echo "T5 CONTROLS DONE $(date -u)" >> leduc_decision_repr/outputs/t5_solve_controls.log
