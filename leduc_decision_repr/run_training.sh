#!/bin/bash
# Phase 13: train 3 seeds x 2 objectives with the frozen configuration (configs/frozen_v1.json).
set -e
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
R=leduc_decision_repr/outputs/runs
mkdir -p $R
STEPS=6000
# round 1: four single-thread runs in parallel
python3 -m leduc_decision_repr.train --method decision --seed 0 --steps $STEPS --threads 1 --out $R/decision_s0 > $R/decision_s0.log 2>&1 &
python3 -m leduc_decision_repr.train --method decision --seed 1 --steps $STEPS --threads 1 --out $R/decision_s1 > $R/decision_s1.log 2>&1 &
python3 -m leduc_decision_repr.train --method decision --seed 2 --steps $STEPS --threads 1 --out $R/decision_s2 > $R/decision_s2.log 2>&1 &
python3 -m leduc_decision_repr.train --method recon --seed 0 --steps $STEPS --threads 1 --out $R/recon_s0 > $R/recon_s0.log 2>&1 &
wait
# round 2: two double-thread runs
export OMP_NUM_THREADS=2
python3 -m leduc_decision_repr.train --method recon --seed 1 --steps $STEPS --threads 2 --out $R/recon_s1 > $R/recon_s1.log 2>&1 &
python3 -m leduc_decision_repr.train --method recon --seed 2 --steps $STEPS --threads 2 --out $R/recon_s2 > $R/recon_s2.log 2>&1 &
wait
echo "ALL TRAINING DONE" > $R/DONE
