"""Summarize the supplementary deployment comparison: regularized-QP deployment (tau fixed) vs the official
LP-vertex deployment from the same g_hat, per method, with paired bootstrap CIs over opponents."""
from __future__ import annotations

import glob, json, sys
from pathlib import Path

import numpy as np

from .common import OUT, N_BUDGETS, EPSILONS, save_json
from .data.datasets import load_population
from .analysis.metrics import EvalData, bootstrap_mean


def main(eval_dir: Path, tau: float):
    pop = load_population(); ed = EvalData(eval_dir, pop)
    rng = np.random.default_rng(0)
    out = {"tau": tau, "methods": {}}
    for f in sorted(glob.glob(str(eval_dir / f"deployqp_tau{tau}_*.npz"))):
        m = Path(f).stem.replace(f"deployqp_tau{tau}_", "")
        d = np.load(f)
        u_qp = ed.per_opp(d["u"]); u_lp = ed.u(m)                    # (n_opp, nN, nE)
        R_qp = ed.V[:, None, :] - u_qp; R_lp = ed.V[:, None, :] - u_lp
        viol = (d["e_os"] - np.array(EPSILONS)[None, None])
        entry = {"max_violation": float(np.nanmax(viol)), "n_violations_gt_1e-6": int(np.nansum(viol > 1e-6)), "qp_failures": int((~d["ok"][:, :, 1:]).sum())}
        for k, eps in enumerate(EPSILONS):
            if k == 0:
                continue
            mq, loq, hiq, _ = bootstrap_mean(R_qp[:, :, k], rng, n_boot=1000)
            ml = np.nanmean(R_lp[:, :, k], 0)
            md, lod, hid, _ = bootstrap_mean(R_qp[:, :, k] - R_lp[:, :, k], rng, n_boot=1000)
            entry[str(eps)] = {"regret_qp": mq.tolist(), "regret_qp_lo": loq.tolist(), "regret_qp_hi": hiq.tolist(), "regret_lp": ml.tolist(),
                               "diff_qp_minus_lp": md.tolist(), "diff_lo": lod.tolist(), "diff_hi": hid.tolist()}
        out["methods"][m] = entry
        print(m, f"max viol {entry['max_violation']:.1e}", "| eps=0.1 LP:", " ".join(f"{v:.3f}" for v in entry["0.1"]["regret_lp"]),
              "| QP:", " ".join(f"{v:.3f}" for v in entry["0.1"]["regret_qp"]), "| diff:", " ".join(f"{a:+.3f}[{lo:+.3f},{hi:+.3f}]" for a, lo, hi in zip(entry["0.1"]["diff_qp_minus_lp"], entry["0.1"]["diff_lo"], entry["0.1"]["diff_hi"])))
    save_json(out, eval_dir / f"deployqp_tau{tau}_comparison.json")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "eval" / "test", float(sys.argv[2]) if len(sys.argv) > 2 else 0.01)
