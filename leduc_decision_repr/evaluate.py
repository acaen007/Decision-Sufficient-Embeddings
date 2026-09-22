"""Phase 14: evaluate every method on a split (test by default) — run ONCE on test.

Stage `predict`: for each method and each (opponent, stream, N) produce g_hat (and z, q_hat
metrics where applicable).  Stage `solve`: feed every g_hat into the same epsilon-safe LP,
deploy the behavioral policy, compute u(x_hat, q), the fast exact exploitability and the
independent OpenSpiel best-response audit.  Results are saved as raw arrays.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from .common import OUT, N_BUDGETS, EPSILONS, save_json, load_json, environment_info
from .game.leduc_tree import get_tree
from .game.sequence_form import get_sequence_form
from .game.symmetry import get_symmetry
from .game.policy_utils import RankPolicyToG
from .data.tokenizer import get_token_table
from .data.datasets import load_population, find_dataset_dir, load_split
from .baselines.likelihood import HandLikelihood, type_counts
from .baselines.tabular_em import TabularEM
from .baselines.bank_posterior import BankPosterior

EVAL_DIR = OUT / "eval"


def predict(split: str, runs: dict, out_dir: Path, em_alpha: float = 1.0, em_iter: int = 200):
    """runs: {method_name: run_dir} for neural runs."""
    import torch
    torch.set_num_threads(4)
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    tree, sf, sym, tab = get_tree(), get_sequence_form(), get_symmetry(), get_token_table()
    pop = load_population(); ds = find_dataset_dir(); data = load_split(ds, split)
    opp_ids = data["opp_ids"]; obs = data["obs_types"]
    n_opp, n_streams, _ = obs.shape
    H = n_opp * n_streams
    hist_opp = np.repeat(opp_ids, n_streams)                     # opponent id of each history
    flat_obs = obs.reshape(H, -1)
    G_true = pop["G"][hist_opp]                                  # (H, n0)
    Q_true = pop["rank_policies"][hist_opp]                      # (H, 144, 3)
    r2g = RankPolicyToG(tree, sf, sym)
    lik = HandLikelihood(tab, sym)
    train_ids = np.flatnonzero(pop["split"] == 0)
    g_stats = load_json(Path(list(runs.values())[0]) / "config.json") if runs else None
    G_train = pop["G"][train_ids]
    g_mean, g_std = G_train.mean(0), G_train.std(0)
    valid = g_std > 1e-3 * g_std.max()
    g_std_safe = np.where(valid, g_std, 1.0)
    meta = {"split": split, "n_opp": int(n_opp), "n_streams": int(n_streams), "H": int(H), "N_budgets": N_BUDGETS,
            "hist_opp": hist_opp.tolist(), "methods": {}, "g_valid_dims": int(valid.sum())}
    np.save(out_dir / "hist_opp.npy", hist_opp)

    def g_err(g_hat):
        nmse = (((g_hat - G_true) / g_std_safe) ** 2)[:, valid].mean(1)
        raw = np.sqrt(((g_hat - G_true) ** 2).sum(1))
        return nmse, raw

    def q_err(q_hat):
        mask = sym.rank_legal_mask[1]
        d = ((q_hat - Q_true) * mask[None]) ** 2
        return np.sqrt(d.sum((1, 2)) / mask.shape[0])            # uniform-infoset RMS over infosets

    # ---------------- classical baselines
    uniform = sym.rank_legal_mask[1] / sym.rank_legal_mask[1].sum(1, keepdims=True)
    nash_rank = pop["blueprint1_rank"]
    bank = BankPosterior(lik, pop["rank_policies"][train_ids], pop["G"][train_ids])
    fam_train = pop["family_index"][train_ids]
    for name, prior in [("TABULAR_EM_UNIFORM", uniform), ("TABULAR_EM_NASH", nash_rank)]:
        em = TabularEM(lik, sym.rank_legal_mask[1], prior, alpha=em_alpha, n_iter=em_iter)
        ghat = np.zeros((H, len(N_BUDGETS), G_true.shape[1]), dtype=np.float32)
        gn = np.zeros((H, len(N_BUDGETS))); gr = np.zeros((H, len(N_BUDGETS))); qe = np.zeros((H, len(N_BUDGETS)))
        for j, N in enumerate(N_BUDGETS):
            counts = type_counts(flat_obs[:, :N], tab.n_types)
            q_hat = em.fit(counts)
            g_hat = r2g.g(q_hat)
            ghat[:, j] = g_hat; gn[:, j], gr[:, j] = g_err(g_hat); qe[:, j] = q_err(q_hat)
            print(f"{name} N={N} done ({time.time()-t0:.0f}s)", flush=True)
        np.save(out_dir / f"ghat_{name}.npy", ghat)
        np.savez(out_dir / f"pred_{name}.npz", g_nmse=gn, g_raw=gr, q_err=qe)
        meta["methods"][name] = {"kind": "classical", "prior": name.split("_")[-1], "alpha": em_alpha}
    # bank posterior
    name = "BANK_POSTERIOR"
    ghat = np.zeros((H, len(N_BUDGETS), G_true.shape[1]), dtype=np.float32)
    gn = np.zeros((H, len(N_BUDGETS))); gr = np.zeros((H, len(N_BUDGETS)))
    fam_mass = np.zeros((H, len(N_BUDGETS), 4)); ent = np.zeros((H, len(N_BUDGETS)))
    for j, N in enumerate(N_BUDGETS):
        counts = type_counts(flat_obs[:, :N], tab.n_types)
        g_bar, post = bank.g_bar(counts)
        ghat[:, j] = g_bar; gn[:, j], gr[:, j] = g_err(g_bar)
        for f in range(4):
            fam_mass[:, j, f] = post[:, fam_train == f].sum(1)
        ent[:, j] = -(post * np.log(np.maximum(post, 1e-300))).sum(1)
    np.save(out_dir / f"ghat_{name}.npy", ghat)
    np.savez(out_dir / f"pred_{name}.npz", g_nmse=gn, g_raw=gr, family_mass=fam_mass, posterior_entropy=ent)
    meta["methods"][name] = {"kind": "classical"}
    print(f"{name} done ({time.time()-t0:.0f}s)", flush=True)
    # ---------------- neural runs
    from .train import load_trained
    from .models.torch_g import TorchRankPolicyToG
    tg = TorchRankPolicyToG(r2g)
    for name, run_dir in runs.items():
        enc, head, ck = load_trained(run_dir)
        method = ck["method"]
        ghat = np.zeros((H, len(N_BUDGETS), G_true.shape[1]), dtype=np.float32)
        Z = np.zeros((H, len(N_BUDGETS), enc.history.proj.out_features), dtype=np.float32)
        gn = np.zeros((H, len(N_BUDGETS))); gr = np.zeros((H, len(N_BUDGETS))); qe = np.full((H, len(N_BUDGETS)), np.nan)
        with torch.no_grad():
            for j, N in enumerate(N_BUDGETS):
                for start in range(0, H, 100):
                    sl = slice(start, min(start + 100, H))
                    x = torch.as_tensor(flat_obs[sl, :N].astype(np.int64))
                    z = enc(x)
                    Z[sl, j] = z.numpy()
                    if method == "recon":
                        q_hat = head(z)
                        g_hat = tg(q_hat).numpy()
                        qe[sl, j] = q_err(q_hat.numpy())[0:0].sum() if False else np.sqrt((((q_hat.numpy() - Q_true[sl]) * sym.rank_legal_mask[1][None]) ** 2).sum((1, 2)) / 144)
                    else:
                        g_hat = head(z).numpy()
                    ghat[sl, j] = g_hat
                nm, rw = g_err(ghat[:, j].astype(np.float64)); gn[:, j] = nm; gr[:, j] = rw
                print(f"{name} N={N} done ({time.time()-t0:.0f}s)", flush=True)
        np.save(out_dir / f"ghat_{name}.npy", ghat)
        np.savez(out_dir / f"pred_{name}.npz", g_nmse=gn, g_raw=gr, q_err=qe, z=Z)
        meta["methods"][name] = {"kind": "neural", "objective": method, "run_dir": str(run_dir), "best_step": int(ck["step"])}
    meta["predict_runtime_s"] = time.time() - t0
    save_json(meta, out_dir / "predict_meta.json")
    print("predict stage done", time.time() - t0)


# ---------------------------------------------------------------------------- solve stage
def _solve_worker(args):
    out_dir, method, h_lo, h_hi, worker_id = args
    from .game.safe_lp import LeducSafeSolver, OpenSpielAuditor
    tree, sf = get_tree(), get_sequence_form()
    L = LeducSafeSolver(sf); aud = OpenSpielAuditor(sf, L.v_star)
    pop = load_population()
    hist_opp = np.load(Path(out_dir) / "hist_opp.npy")
    ghat = np.load(Path(out_dir) / f"ghat_{method}.npy", mmap_mode="r")
    nH = h_hi - h_lo; nN = len(N_BUDGETS); nE = len(EPSILONS)
    u = np.zeros((nH, nN, nE)); e_fast = np.zeros((nH, nN, nE)); e_os = np.zeros((nH, nN, nE))
    ok = np.zeros((nH, nN, nE), dtype=bool)
    t0 = time.time()
    for i, h in enumerate(range(h_lo, h_hi)):
        g_true = pop["G"][hist_opp[h]]
        for j in range(nN):
            g = np.asarray(ghat[h, j], dtype=np.float64)
            for k, eps in enumerate(EPSILONS):
                okk, pol, x = L.solve_safe(g, eps)
                ok[i, j, k] = okk
                u[i, j, k] = x @ g_true
                e_fast[i, j, k] = L.exploitability(x)
                e_os[i, j, k] = aud.exploitability_of_learner(pol)
        if worker_id == 0 and (i + 1) % 20 == 0:
            print(f"  [{method}] worker0 {i+1}/{nH} ({time.time()-t0:.0f}s)", flush=True)
    return h_lo, u, e_fast, e_os, ok, L.lp0.n_fail


def solve(out_dir: Path, methods, workers: int = 4):
    t0 = time.time()
    meta = load_json(out_dir / "predict_meta.json")
    H = meta["H"]
    chunks = np.linspace(0, H, workers + 1).astype(int)
    summary = {}
    for method in methods:
        if (out_dir / f"solve_{method}.npz").exists():
            print(f"{method}: already solved, skipping"); continue
        tm = time.time()
        jobs = [(str(out_dir), method, int(chunks[w]), int(chunks[w + 1]), w) for w in range(workers)]
        with mp.get_context("spawn").Pool(workers) as pool:
            res = pool.map(_solve_worker, jobs)
        nN, nE = len(N_BUDGETS), len(EPSILONS)
        u = np.zeros((H, nN, nE)); ef = np.zeros((H, nN, nE)); eo = np.zeros((H, nN, nE)); ok = np.zeros((H, nN, nE), bool)
        fails = 0
        for h_lo, uu, ff, oo, kk, nf in res:
            n = uu.shape[0]
            u[h_lo:h_lo + n] = uu; ef[h_lo:h_lo + n] = ff; eo[h_lo:h_lo + n] = oo; ok[h_lo:h_lo + n] = kk; fails += nf
        np.savez(out_dir / f"solve_{method}.npz", u=u, e_fast=ef, e_os=eo, ok=ok, lp_failures=fails)
        viol = eo - np.array(EPSILONS)[None, None]
        summary[method] = {"lp_failures": int(fails), "max_violation_os": float(viol.max()),
                           "n_violations_1e-7": int((viol > 1e-7).sum()), "max_abs_fast_vs_os": float(np.abs(ef - eo).max()),
                           "runtime_s": time.time() - tm}
        print(f"{method}: solved {H*nN*nE} LPs in {time.time()-tm:.0f}s; max OpenSpiel violation {viol.max():.2e}; "
              f"LP failures {fails}", flush=True)
        save_json(summary, out_dir / "solve_summary_partial.json")
    save_json({"summary": summary, "runtime_s": time.time() - t0, "env": environment_info()}, out_dir / "solve_summary.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["predict", "solve"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    ap.add_argument("--runs", default="", help="comma list name=run_dir for neural runs")
    ap.add_argument("--methods", default="", help="comma list of method names to solve (default: all predicted)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out_dir = Path(args.out) if args.out else EVAL_DIR / args.split
    if args.stage == "predict":
        runs = dict(kv.split("=") for kv in args.runs.split(",") if kv)
        predict(args.split, runs, out_dir)
    else:
        meta = load_json(out_dir / "predict_meta.json")
        methods = args.methods.split(",") if args.methods else list(meta["methods"].keys())
        solve(out_dir, methods, args.workers)
