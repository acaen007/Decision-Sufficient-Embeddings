"""Supplementary V2 analysis: deploy the *regularized* exactly-safe solution (tau fixed) from the same g_hat
instead of the LP vertex, for selected methods, with the same exact audits.  Both deployments are exactly
epsilon-safe; the official protocol uses the LP vertex (evaluate.py).  Never used for selection."""
from __future__ import annotations

import argparse, multiprocessing as mp, os, time
from pathlib import Path

import numpy as np

from .common import OUT, N_BUDGETS, EPSILONS, save_json, load_json
from .game.leduc_tree import get_tree
from .game.sequence_form import get_sequence_form
from .data.datasets import load_population


def _worker(args):
    out_dir, method, h_lo, h_hi, tau, eps_idx, wid = args
    from .game.safe_lp import LeducSafeSolver, OpenSpielAuditor
    from .game.safe_qp import SafeQP
    tree, sf = get_tree(), get_sequence_form()
    L = LeducSafeSolver(sf); aud = OpenSpielAuditor(sf, L.v_star); qp = SafeQP(sf, L.v_star)
    pop = load_population(); hist_opp = np.load(Path(out_dir) / "hist_opp.npy")
    ghat = np.load(Path(out_dir) / f"ghat_{method}.npy", mmap_mode="r")
    nH = h_hi - h_lo; nN = len(N_BUDGETS); nE = len(EPSILONS)
    u = np.full((nH, nN, nE), np.nan); e_os = np.full((nH, nN, nE), np.nan); ok = np.zeros((nH, nN, nE), bool)
    t0 = time.time()
    for i, h in enumerate(range(h_lo, h_hi)):
        g_true = pop["G"][hist_opp[h]]
        for j in range(nN):
            g = np.asarray(ghat[h, j], dtype=np.float64)
            for k in eps_idx:
                sol = qp.solve(g, EPSILONS[k], tau)
                pol = sf.realization_to_behavioral(0, sol["x"]); xd = sf.behavioral_to_realization(0, pol)
                ok[i, j, k] = sol["ok"]; u[i, j, k] = xd @ g_true; e_os[i, j, k] = aud.exploitability_of_learner(pol)
        if wid == 0 and (i + 1) % 20 == 0:
            print(f"  [{method}] worker0 {i+1}/{nH} ({time.time()-t0:.0f}s)", flush=True)
    return h_lo, u, e_os, ok


def main(out_dir: Path, methods, tau: float, workers: int, eps_idx, max_hist=None):
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    meta = load_json(out_dir / "predict_meta.json"); H = meta["H"] if max_hist is None else min(meta["H"], max_hist)
    chunks = np.linspace(0, H, workers + 1).astype(int)
    summary = {}
    for m in methods:
        t0 = time.time()
        jobs = [(str(out_dir), m, int(chunks[w]), int(chunks[w + 1]), tau, eps_idx, w) for w in range(workers)]
        with mp.get_context("spawn").Pool(workers) as pool:
            res = pool.map(_worker, jobs)
        u = np.full((H, len(N_BUDGETS), len(EPSILONS)), np.nan); e = u.copy(); ok = np.zeros(u.shape, bool)
        for h_lo, uu, ee, kk in res:
            n = uu.shape[0]; u[h_lo:h_lo + n] = uu; e[h_lo:h_lo + n] = ee; ok[h_lo:h_lo + n] = kk
        np.savez(out_dir / f"deployqp_tau{tau}_{m}.npz", u=u, e_os=e, ok=ok, tau=tau)
        viol = (e - np.array(EPSILONS)[None, None])[:, :, eps_idx]
        summary[m] = {"max_violation": float(np.nanmax(viol)), "n_violations_1e-6": int((viol > 1e-6).sum()), "qp_failures": int((~ok[:, :, eps_idx]).sum()),
                      "runtime_s": time.time() - t0}
        print(f"{m}: regularized deployment done in {time.time()-t0:.0f}s; max violation {np.nanmax(viol):.2e}", flush=True)
    save_json(summary, out_dir / f"deployqp_tau{tau}_summary.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "eval" / "test")); ap.add_argument("--methods", required=True)
    ap.add_argument("--tau", type=float, default=0.01); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eps_idx", default="1,2,3"); ap.add_argument("--max_hist", type=int, default=None)
    a = ap.parse_args()
    main(Path(a.out), a.methods.split(","), a.tau, a.workers, [int(v) for v in a.eps_idx.split(",")], a.max_hist)
