"""V2 Phase B: evaluate the SAFE_REGRET sweep runs on the VALIDATION split with the exact safe LP
(eps in {0.05, 0.1, 0.2}) and report validation safe regret per run.  Selection only; never test."""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np

from .common import OUT, N_BUDGETS, EPSILONS, save_json, load_json
from .evaluate import predict, solve
from .data.datasets import load_population
from .analysis.metrics import EvalData


def main(runs: dict, out_dir: Path, workers: int):
    predict("val", runs, out_dir, skip_classical=True, threads=workers)
    solve(out_dir, list(runs.keys()), workers=workers, eps_idx=[1, 2, 3])
    pop = load_population(); ed = EvalData(out_dir, pop)
    res = {}
    for m in runs:
        R = ed.regret(m)                                   # (n_opp, nN, nE)
        per_eps = {str(EPSILONS[k]): {"by_N": np.nanmean(R[:, :, k], 0).tolist(), "mean": float(np.nanmean(R[:, :, k]))} for k in [1, 2, 3]}
        p = ed.pred[m]
        res[m] = {"val_regret_mean_all": float(np.nanmean(R[:, :, 1:])), "per_eps": per_eps,
                  "g_nmse_by_N": ed.per_opp(p["g_nmse"]).mean(0).tolist(),
                  "best_step": int(load_json(Path(runs[m]) / "result.json")["best_step"]) + 1}
    best = min(res, key=lambda m: res[m]["val_regret_mean_all"])
    res["_selected"] = best
    save_json(res, out_dir / "sweep_selection.json")
    for m, r in res.items():
        if m.startswith("_"):
            continue
        print(f"{m:28s} val regret (mean over N, eps>0) = {r['val_regret_mean_all']:.4f}  by eps: "
              + ", ".join(f"{e}: {v['mean']:.4f}" for e, v in r["per_eps"].items()) + f"  best step {r['best_step']}")
    print("SELECTED:", best)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--out", default=str(OUT / "eval" / "val_sweep_v2"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    main(dict(kv.split("=") for kv in a.runs.split(",")), Path(a.out), a.workers)
