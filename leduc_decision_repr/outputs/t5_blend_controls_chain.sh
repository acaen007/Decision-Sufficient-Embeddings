#!/bin/bash
# T5a controls: single-seed BLEND (ensemble confound) and 3-seed decision ensemble without EM (lambda = 1).
cd /home/user/Decision-Sufficient-Embeddings
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
R=leduc_decision_repr/outputs/runs_v3
nice -n 12 python3 -m leduc_decision_repr.t5_hybrids --net_runs $R/dec133k_s0 --suffix _S0 > leduc_decision_repr/outputs/t5_blend_s0.log 2>&1
nice -n 12 python3 -m leduc_decision_repr.t5_hybrids --net_runs $R/dec133k_s0,$R/dec133k_s1,$R/dec133k_s2 --lams 1.0 --suffix _ENSDEC > leduc_decision_repr/outputs/t5_ensdec.log 2>&1
while pgrep -f "evaluate predic[t]" > /dev/null; do sleep 30; done
python3 -m leduc_decision_repr.merge_hyb_meta
nice -n 8 python3 -m leduc_decision_repr.evaluate solve --methods HYB_BLEND_S0,HYB_BLEND_ENSDEC --workers 2 --eps_idx 2 > leduc_decision_repr/outputs/t5_solve_blend_controls.log 2>&1
echo "T5 BLEND CONTROLS DONE $(date -u)" >> leduc_decision_repr/outputs/t5_solve_blend_controls.log
