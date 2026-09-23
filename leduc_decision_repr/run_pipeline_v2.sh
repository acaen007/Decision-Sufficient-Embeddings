#!/bin/bash
# V2 Phases B-D automation: sweep selection (validation only) -> seeds 1,2 -> test evaluation -> analysis.
cd "$(dirname "$0")/.."
R=leduc_decision_repr/outputs/runs_v2
O=leduc_decision_repr/outputs
until [ -f $R/SWEEP_DONE ]; do sleep 60; done
echo "sweep done at $(date)" > $O/pipeline_v2.log
export OMP_NUM_THREADS=1
SWEEP=SR_t0.01=$R/safe_regret_t0.01_s0,SR_t0.05=$R/safe_regret_t0.05_s0,SR_t0.10=$R/safe_regret_t0.10_s0,SR_const1e-4=$R/safe_regret_const1e-4_s0
python3 -m leduc_decision_repr.sweep_select_v2 --runs $SWEEP --workers 4 > $O/sweep_select_v2.log 2>&1
echo "sweep selection done at $(date)" >> $O/pipeline_v2.log
SEL=$(python3 -c "import json; print(json.load(open('$O/eval/val_sweep_v2/sweep_selection.json'))['_selected'])")
SELDIR=$(python3 -c "print(dict(kv.split('=') for kv in '$SWEEP'.split(','))['$SEL'])")
TAU=$(python3 -c "import json; print(json.load(open('$SELDIR/config.json'))['cfg']['tau_start'])")
SCHED=$(python3 -c "import json; print(json.load(open('$SELDIR/config.json'))['cfg']['tau_schedule'])")
echo "selected $SEL ($SELDIR) tau_start=$TAU schedule=$SCHED" >> $O/pipeline_v2.log
export OMP_NUM_THREADS=2
python3 -m leduc_decision_repr.train --method safe_regret --seed 1 --steps 6000 --threads 2 --tau_start $TAU --tau_schedule $SCHED --out $R/safe_regret_final_s1 > $R/safe_regret_final_s1.log 2>&1 &
python3 -m leduc_decision_repr.train --method safe_regret --seed 2 --steps 6000 --threads 2 --tau_start $TAU --tau_schedule $SCHED --out $R/safe_regret_final_s2 > $R/safe_regret_final_s2.log 2>&1 &
wait
echo "seeds 1,2 done at $(date)" >> $O/pipeline_v2.log
export OMP_NUM_THREADS=1
RUNS=NEURAL_SAFE_REGRET_s0=$SELDIR,NEURAL_SAFE_REGRET_s1=$R/safe_regret_final_s1,NEURAL_SAFE_REGRET_s2=$R/safe_regret_final_s2
python3 -m leduc_decision_repr.evaluate predict --split test --skip_classical --threads 4 --runs $RUNS > $O/eval_predict_v2.log 2>&1
echo "test predict done at $(date)" >> $O/pipeline_v2.log
python3 -m leduc_decision_repr.evaluate solve --split test --workers 4 --methods NEURAL_SAFE_REGRET_s0,NEURAL_SAFE_REGRET_s1,NEURAL_SAFE_REGRET_s2 > $O/eval_solve_v2.log 2>&1
echo "test solve done at $(date)" >> $O/pipeline_v2.log
python3 -m leduc_decision_repr.analyze --sweep_dirs $SWEEP > $O/analyze_v2.log 2>&1
python3 -m leduc_decision_repr.report_tables leduc_decision_repr/outputs/eval/test > $O/eval/test/report_tables.md 2>/dev/null
echo "analysis done at $(date)" >> $O/pipeline_v2.log
touch $O/PIPELINE_V2_DONE
