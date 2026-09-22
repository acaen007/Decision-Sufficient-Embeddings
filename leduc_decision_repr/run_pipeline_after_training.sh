#!/bin/bash
# Phase 14-16 automation: waits for training, then predict (neural) -> solve (neural) -> analyze.
cd "$(dirname "$0")/.."
R=leduc_decision_repr/outputs/runs
O=leduc_decision_repr/outputs
until [ -f $R/DONE ]; do sleep 60; done
echo "training done at $(date)" > $O/pipeline.log
export OMP_NUM_THREADS=1
RUNS=NEURAL_DECISION_s0=$R/decision_s0,NEURAL_DECISION_s1=$R/decision_s1,NEURAL_DECISION_s2=$R/decision_s2,NEURAL_RECON_s0=$R/recon_s0,NEURAL_RECON_s1=$R/recon_s1,NEURAL_RECON_s2=$R/recon_s2
python3 -m leduc_decision_repr.evaluate predict --split test --skip_classical --threads 4 --runs $RUNS > $O/eval_predict_neural.log 2>&1
echo "neural predict done at $(date)" >> $O/pipeline.log
python3 -m leduc_decision_repr.evaluate solve --split test --workers 4 --methods NEURAL_DECISION_s0,NEURAL_DECISION_s1,NEURAL_DECISION_s2,NEURAL_RECON_s0,NEURAL_RECON_s1,NEURAL_RECON_s2 > $O/eval_solve_neural.log 2>&1
echo "neural solve done at $(date)" >> $O/pipeline.log
until [ -f $O/eval/test/solve_BANK_POSTERIOR.npz ] && [ -f $O/eval/test/solve_TABULAR_EM_NASH.npz ] && [ -f $O/eval/test/solve_TABULAR_EM_UNIFORM.npz ]; do sleep 60; done
echo "classical solve done at $(date)" >> $O/pipeline.log
python3 -m leduc_decision_repr.analyze > $O/analyze.log 2>&1
echo "analysis done at $(date)" >> $O/pipeline.log
touch $O/PIPELINE_DONE
