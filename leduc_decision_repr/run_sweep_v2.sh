#!/bin/bash
# V2 Phase B: validation-only tau sweep for SAFE_REGRET (seed 0, full budget, identical otherwise).
set -e
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
R=leduc_decision_repr/outputs/runs_v2
mkdir -p $R
STEPS=6000
python3 -m leduc_decision_repr.train --method safe_regret --seed 0 --steps $STEPS --threads 1 --tau_start 0.01 --tau_schedule linear --out $R/safe_regret_t0.01_s0 > $R/safe_regret_t0.01_s0.log 2>&1 &
python3 -m leduc_decision_repr.train --method safe_regret --seed 0 --steps $STEPS --threads 1 --tau_start 0.05 --tau_schedule linear --out $R/safe_regret_t0.05_s0 > $R/safe_regret_t0.05_s0.log 2>&1 &
python3 -m leduc_decision_repr.train --method safe_regret --seed 0 --steps $STEPS --threads 1 --tau_start 0.10 --tau_schedule linear --out $R/safe_regret_t0.10_s0 > $R/safe_regret_t0.10_s0.log 2>&1 &
python3 -m leduc_decision_repr.train --method safe_regret --seed 0 --steps $STEPS --threads 1 --tau_start 0.0001 --tau_schedule const --out $R/safe_regret_const1e-4_s0 > $R/safe_regret_const1e-4_s0.log 2>&1 &
wait
echo "SWEEP DONE" > $R/SWEEP_DONE
