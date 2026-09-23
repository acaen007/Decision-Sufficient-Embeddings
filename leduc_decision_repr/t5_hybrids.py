"""T5 hybrids that need no new training.
(a) BLEND:  g_hat = lam_N * g_net + (1 - lam_N) * g_EM,  lam_N per N fitted on the validation split.
(c) PRIOR-EM: tabular EM with a Dirichlet prior whose mean is the reconstruction model's q_hat and whose
    concentration kappa_N is fitted on validation; the estimate is consistent as N -> infinity.
Both produce g_hat arrays for the test split that go through the standard exact LP + audit stage."""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import torch

from .common import OUT, N_BUDGETS, EPSILONS, save_json, load_json
from .game.leduc_tree import get_tree
from .game.sequence_form import get_sequence_form
from .game.safe_lp import get_solver
from .game.symmetry import get_symmetry
from .game.policy_utils import RankPolicyToG
from .data.tokenizer import get_token_table
from .data.datasets import load_population, find_dataset_dir, load_split
from .baselines.likelihood import HandLikelihood, type_counts
from .baselines.tabular_em import TabularEM
from .train import load_trained
from .models.torch_g import TorchRankPolicyToG

VAL_STREAMS = 1     # validation subset for fitting (150 opponents x 1 stream) to keep the LP cost small


def val_regret(L, pop, g_hat, opp_ids, eps=0.1):
    k = EPSILONS.index(eps); r = np.zeros(len(opp_ids))
    for i in range(len(opp_ids)):
        _, _, x = L.solve_safe(g_hat[i], eps); r[i] = pop["V_oracle"][opp_ids[i], k] - x @ pop["G"][opp_ids[i]]
    return r


def net_ghat(run_dirs, obs, N, method_is_recon, tg):
    """Mean over seeds of the network's g_hat (or q_hat for recon) on (n, N) obs types."""
    outs = []
    for rd in run_dirs:
        enc, head, ck = load_trained(rd)
        with torch.no_grad():
            z = enc(torch.as_tensor(obs[:, :N].astype(np.int64)))
            outs.append((head(z) if not method_is_recon else head(z)).numpy())
    return np.mean(outs, 0)


def main(net_runs, recon_runs, out_dir: Path, kappas=(1.0, 3.0, 10.0, 30.0, 100.0), lams=None, suffix="", ensemble_only=False):
    t0 = time.time(); out_dir.mkdir(parents=True, exist_ok=True)
    tree, sf, sym, tab = get_tree(), get_sequence_form(), get_symmetry(), get_token_table()
    L = get_solver(); pop = load_population(); r2g = RankPolicyToG(tree, sf, sym); tg = TorchRankPolicyToG(r2g)
    lik = HandLikelihood(tab, sym); mask = sym.rank_legal_mask[1]
    uniform = mask / mask.sum(1, keepdims=True)
    ds = find_dataset_dir(); val = load_split(ds, "val"); test = load_split(ds, "test")
    lams = lams or [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    res = {"net_runs": net_runs, "recon_runs": recon_runs, "blend": {}, "prior_em": {}}
    # ---------------- validation fits
    vobs = val["obs_types"][:, :VAL_STREAMS].reshape(-1, val["obs_types"].shape[2]); vids = np.repeat(val["opp_ids"], VAL_STREAMS)
    em_u = TabularEM(lik, mask, uniform, alpha=1.0, n_iter=200)
    lam_sel, kappa_sel = {}, {}
    for N in N_BUDGETS:
        counts = type_counts(vobs[:, :N], tab.n_types)
        g_em = r2g.g(em_u.fit(counts))
        g_net = net_ghat(net_runs, vobs, N, False, tg) if net_runs else None
        if g_net is not None:
            regs = {lam: float(val_regret(L, pop, lam * g_net + (1 - lam) * g_em, vids).mean()) for lam in lams}
            lam_sel[N] = min(regs, key=regs.get); res["blend"][str(N)] = {"val_regret_by_lambda": regs, "selected": lam_sel[N]}
        if recon_runs and not ensemble_only:
            q_hat = net_ghat(recon_runs, vobs, N, True, tg)                 # (n, 144, 3)
            regs = {}
            for kappa in kappas:
                # per-history prior: EM with prior mean q_hat_h (vectorized: TabularEM takes one prior; loop over kappa with per-row prior)
                Q = fit_em_with_row_prior(lik, mask, q_hat, kappa, counts)
                regs[kappa] = float(val_regret(L, pop, r2g.g(Q), vids).mean())
            kappa_sel[N] = min(regs, key=regs.get); res["prior_em"][str(N)] = {"val_regret_by_kappa": regs, "selected": kappa_sel[N]}
        print(f"N={N}: blend {res['blend'].get(str(N), {}).get('selected')}  prior-EM kappa {res['prior_em'].get(str(N), {}).get('selected')}  ({time.time()-t0:.0f}s)", flush=True)
    # ---------------- test predictions with the selected hyper-parameters
    tobs = test["obs_types"].reshape(-1, test["obs_types"].shape[2]); H = tobs.shape[0]
    n0 = sf.n_seq[0]
    if net_runs:
        ghat_blend = np.zeros((H, len(N_BUDGETS), n0), dtype=np.float32)
    if recon_runs:
        ghat_prior = np.zeros((H, len(N_BUDGETS), n0), dtype=np.float32)
    for j, N in enumerate(N_BUDGETS):
        counts = type_counts(tobs[:, :N], tab.n_types)
        if net_runs:
            g_em = r2g.g(em_u.fit(counts)); g_net = net_ghat(net_runs, tobs, N, False, tg)
            ghat_blend[:, j] = lam_sel[N] * g_net + (1 - lam_sel[N]) * g_em
        if recon_runs:
            q_hat = net_ghat(recon_runs, tobs, N, True, tg)
            ghat_prior[:, j] = r2g.g(q_hat) if ensemble_only else r2g.g(fit_em_with_row_prior(lik, mask, q_hat, kappa_sel[N], counts))
        print(f"test N={N} done ({time.time()-t0:.0f}s)", flush=True)
    # write into the standard eval dir as pseudo-methods (g errors as in predict)
    G_true = pop["G"][np.repeat(test["opp_ids"], test["obs_types"].shape[1])]
    train_ids = np.flatnonzero(pop["split"] == 0); G_train = pop["G"][train_ids]; g_mean, g_std = G_train.mean(0), G_train.std(0)
    valid = g_std > 1e-3 * g_std.max(); g_std_safe = np.where(valid, g_std, 1.0)
    hm = out_dir / "hyb_meta.json"
    meta = load_json(hm) if hm.exists() else {"methods": {}}
    prior_name = ("HYB_ENSREC" if ensemble_only else "HYB_PRIOR_EM") + suffix
    for name, arr in [("HYB_BLEND" + suffix, ghat_blend if net_runs else None), (prior_name, ghat_prior if recon_runs else None)]:
        if arr is None:
            continue
        np.save(out_dir / f"ghat_{name}.npy", arr)
        gn = (((arr - G_true[:, None]) / g_std_safe) ** 2)[:, :, valid].mean(2); gr = np.sqrt(((arr - G_true[:, None]) ** 2).sum(2))
        np.savez(out_dir / f"pred_{name}.npz", g_nmse=gn, g_raw=gr)
        meta["methods"][name] = {"kind": "classical", "hybrid": True, "runs": net_runs if name.startswith("HYB_BLEND") else recon_runs,
                                 "selection": res["blend"] if name.startswith("HYB_BLEND") else res["prior_em"]}
    save_json(meta, hm)
    save_json(res, out_dir / (("t5_hybrid_selection_blend" if net_runs else "t5_hybrid_selection_prior") + suffix + ".json"))
    print("done", time.time() - t0)


def fit_em_with_row_prior(lik, mask, q_prior, kappa, counts, n_iter=200, tol=1e-7):
    """Tabular EM where history h has its own Dirichlet prior kappa * q_prior[h] (vectorized over histories)."""
    n = counts.shape[0]
    prior = np.where(mask[None], np.maximum(q_prior, 1e-6), 0.0); prior = prior / prior.sum(2, keepdims=True)
    Q = prior.copy(); c_pair = counts[:, lik.pair_type]; alpha_prior = kappa * prior
    for it in range(n_iter):
        logq = np.log(np.maximum(Q, 1e-300)).reshape(n, -1)
        ll = lik.pair_loglik(logq) + lik.pair_logprior[None]
        m = np.full((n, lik.tab.n_types), -np.inf); np.maximum.at(m.T, lik.pair_type, ll.T)
        w = np.exp(ll - m[:, lik.pair_type]); z = (lik.S @ w.T).T; w = w / np.maximum(z[:, lik.pair_type], 1e-300)
        exp_counts = (lik.M.T @ (w * c_pair).T).T.reshape(n, -1, 3)
        num = exp_counts + alpha_prior; Qn = np.where(mask[None], num / num.sum(2, keepdims=True), 0.0)
        delta = np.abs(Qn - Q).max(); Q = Qn
        if delta < tol:
            break
    return Q


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--net_runs", default=""); ap.add_argument("--recon_runs", default="")
    ap.add_argument("--out", default=str(OUT / "eval" / "test"))
    ap.add_argument("--suffix", default=""); ap.add_argument("--ensemble_only", action="store_true")
    a = ap.parse_args()
    main([r for r in a.net_runs.split(",") if r], [r for r in a.recon_runs.split(",") if r], Path(a.out), suffix=a.suffix, ensemble_only=a.ensemble_only)
