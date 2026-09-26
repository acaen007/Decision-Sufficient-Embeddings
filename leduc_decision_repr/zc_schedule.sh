#!/bin/bash
# Decision-code-from-hands study (REPORT_LEDUC_ZCODE.md): two training waves of 4 runs (1 thread each), then test
# evaluation at eps = 0.10, then the PRIOR-EM (ZC prior) hybrid with the V3 recipe.  Restart-safe: finished runs and
# evaluations are skipped.
set -u
cd "$(dirname "$0")/.."
R=leduc_decision_repr/outputs/runs_zcode; mkdir -p $R
AE8=$PWD/leduc_decision_repr/outputs/dcomp/ae_runs/REGRET_d8_s0.pt
AE4=$PWD/leduc_decision_repr/outputs/dcomp/ae_runs/REGRET_d4_s0.pt
stamp() { date -u +%H:%M:%S; }
train() {   # name seed mode d ae
  if [ -f $R/$1/result.json ]; then echo "$(stamp) skip $1 (done)"; return; fi
  echo "$(stamp) start $1"
  python -m leduc_decision_repr.train --method zcode --seed $2 --zc_mode $3 --zc_d $4 --zc_ae $5 --select_by regret \
      --recon_hidden 834 --threads 1 --out $R/$1 > $R/$1.log 2>&1
  echo "$(stamp) trained $1 rc=$?"
}
train zcdec8_s0 0 dec 8 $AE8 & train zcdec8_s1 1 dec 8 $AE8 & train zcdec8_s2 2 dec 8 $AE8 & train zcdist8_s0 0 distill 8 $AE8 & wait
train zcdist8_s1 1 distill 8 $AE8 & train zcdist8_s2 2 distill 8 $AE8 & train zcdec4_s0 0 dec 4 $AE4 & train zcdec4_s1 1 dec 4 $AE4 & wait
for spec in ZC_DEC8:zcdec8 ZC_DIST8:zcdist8 ZC_DEC4:zcdec4; do
  name=${spec%%:*}; run=${spec##*:}
  for s in 0 1 2; do
    [ -f $R/${run}_s$s/best.pt ] || continue
    [ -f $R/${run}_s$s/EVALUATED ] && { echo "$(stamp) skip eval ${run}_s$s"; continue; }
    echo "$(stamp) eval ${run}_s$s"
    python -m leduc_decision_repr.ft_eval --name NEURAL_${name}_s$s --run $R/${run}_s$s --eps_idx 2 --workers 4 > $R/eval_${run}_s$s.log 2>&1
    echo "$(stamp) evaluated ${run}_s$s rc=$?"
  done
done
if [ ! -f $R/EMPRIOR_DONE ]; then
  echo "$(stamp) prior-EM (ZC prior)"
  python -m leduc_decision_repr.t5_hybrids --recon_runs $R/zcdec8_s0,$R/zcdec8_s1,$R/zcdec8_s2 --suffix _ZC > $R/emprior.log 2>&1
  python -m leduc_decision_repr.merge_hyb_meta >> $R/emprior.log 2>&1
  python -c "from pathlib import Path; from leduc_decision_repr.common import OUT; from leduc_decision_repr.evaluate import solve; solve(OUT / 'eval' / 'test', ['HYB_PRIOR_EM_ZC'], workers=4, eps_idx=[2])" >> $R/emprior.log 2>&1
  echo "$(stamp) prior-EM done rc=$?"; touch $R/EMPRIOR_DONE
fi
echo "$(stamp) ALL DONE"
