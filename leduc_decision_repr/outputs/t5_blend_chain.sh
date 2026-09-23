#!/bin/bash
# T5a: BLEND with the matched DEC-133k models (3 seeds), after the scheduler has evaluated them.
cd /home/user/Decision-Sufficient-Embeddings
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
R=leduc_decision_repr/outputs/runs_v3
until [ -f $R/dec133k_s0/EVALUATED ] && [ -f $R/dec133k_s1/EVALUATED ] && [ -f $R/dec133k_s2/EVALUATED ]; do sleep 120; done
nice -n 12 python3 -m leduc_decision_repr.t5_hybrids --net_runs $R/dec133k_s0,$R/dec133k_s1,$R/dec133k_s2 > leduc_decision_repr/outputs/t5_blend.log 2>&1
while pgrep -f "evaluate predic[t]" > /dev/null; do sleep 30; done
python3 -m leduc_decision_repr.merge_hyb_meta
nice -n 8 python3 -m leduc_decision_repr.evaluate solve --methods HYB_BLEND --workers 2 --eps_idx 2 > leduc_decision_repr/outputs/t5_solve_blend.log 2>&1
echo "T5 BLEND DONE $(date -u)" >> leduc_decision_repr/outputs/t5_solve_blend.log
