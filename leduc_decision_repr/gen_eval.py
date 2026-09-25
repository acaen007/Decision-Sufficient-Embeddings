"""Generalization study: g_hat for every method on every history of outputs/gen/families.npz, then the exact
eps-safe LP (eps = 0.10), u(x, q) and the OpenSpiel exploitability audit of every deployed strategy.
All checkpoints / kappa are the ones selected on the ORIGINAL validation set (seed 0; kappa = 3).
Writes outputs/gen/ghat_<M>.npy, outputs/gen/solve.npz and outputs/gen/eval_meta.json."""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, time, multiprocessing as mp
import numpy as np
from .common import OUT, N_BUDGETS, save_json

D = OUT / "gen"; EPS = 0.10; KAPPA = 3.0
NEURAL = {"DEC-889k": "runs/decision_s0", "RECON-889k": "runs_v3/rec889k_s0", "RECON-JAC-global": "runs_jacopp/a0_s0", "JAC-opp": "runs_jacopp/a3_s0"}
PRIORS = {"PRIOR-EM (RECON-131k)": "runs/recon_s0", "PRIOR-EM (JAC-opp)": "runs_jacopp/a3_s0"}
METHODS = ["BANK", "TAB-EM"] + list(NEURAL) + list(PRIORS)          # modelling methods (FIXED-NE handled separately)


def fname(m):
    return m.replace(" ", "_").replace("(", "").replace(")", "")


def predict():
    import torch
    from .game.leduc_tree import get_tree
    from .game.sequence_form import get_sequence_form
    from .game.symmetry import get_symmetry
    from .game.policy_utils import RankPolicyToG
    from .data.tokenizer import get_token_table
    from .data.datasets import load_population
    from .baselines.likelihood import HandLikelihood, type_counts
    from .baselines.tabular_em import TabularEM
    from .baselines.bank_posterior import BankPosterior
    from .train import load_trained
    from .models.torch_g import TorchRankPolicyToG
    from .t5_hybrids import fit_em_with_row_prior
    torch.set_num_threads(4); t0 = time.time(); times = {}
    F = np.load(D / "families.npz"); obs = F["obs"]; H = obs.shape[0]
    tree, sf, sym, tab = get_tree(), get_sequence_form(), get_symmetry(), get_token_table(); pop = load_population()
    r2g = RankPolicyToG(tree, sf, sym); tg = TorchRankPolicyToG(r2g); lik = HandLikelihood(tab, sym); mask = sym.rank_legal_mask[1]
    tr = np.flatnonzero(pop["split"] == 0); nN = len(N_BUDGETS); n0 = sf.n_seq[0]
    out = {m: np.zeros((H, nN, n0), dtype=np.float32) for m in METHODS}
    counts = {N: type_counts(obs[:, :N], tab.n_types) for N in N_BUDGETS}
    # classical
    t = time.time(); bank = BankPosterior(lik, pop["rank_policies"][tr], pop["G"][tr])
    for j, N in enumerate(N_BUDGETS):
        out["BANK"][:, j] = bank.g_bar(counts[N])[0]
    times["BANK"] = time.time() - t; t = time.time()
    em = TabularEM(lik, mask, mask / mask.sum(1, keepdims=True), alpha=1.0, n_iter=200)
    for j, N in enumerate(N_BUDGETS):
        out["TAB-EM"][:, j] = r2g.g(em.fit(counts[N]))
    times["TAB-EM"] = time.time() - t
    print(f"classical done ({time.time()-t0:.0f}s)", flush=True)
    # neural (seed 0) and the q_hat priors
    qhat = {}
    def run_net(run, want_q):
        enc, head, ck = load_trained(OUT / run); enc.eval(); head.eval(); G_ = np.zeros((H, nN, n0), dtype=np.float32); Q_ = np.zeros((H, nN, 144, 3)) if want_q else None
        with torch.no_grad():
            for j, N in enumerate(N_BUDGETS):
                for s in range(0, H, 200):
                    z = enc(torch.as_tensor(obs[s:s + 200, :N].astype(np.int64)))
                    if ck["method"] == "recon":
                        q = head(z); G_[s:s + 200, j] = tg(q).numpy()
                        if want_q:
                            Q_[s:s + 200, j] = q.numpy()
                    else:
                        G_[s:s + 200, j] = head(z).numpy()
        return G_, Q_, ck
    for m, run in NEURAL.items():
        t = time.time(); G_, Q_, ck = run_net(run, m == "JAC-opp"); out[m] = G_; times[m] = time.time() - t
        if m == "JAC-opp":
            qhat["PRIOR-EM (JAC-opp)"] = Q_
        print(f"{m} done (method {ck['method']}, step {ck['step']}) ({time.time()-t0:.0f}s)", flush=True)
    _, qhat["PRIOR-EM (RECON-131k)"], _ = run_net(PRIORS["PRIOR-EM (RECON-131k)"], True)
    for m in PRIORS:
        t = time.time()
        for j, N in enumerate(N_BUDGETS):
            out[m][:, j] = r2g.g(fit_em_with_row_prior(lik, mask, qhat[m][:, j], KAPPA, counts[N]))
        times[m] = time.time() - t
        print(f"{m} done ({time.time()-t0:.0f}s)", flush=True)
    for m in METHODS:
        np.save(D / f"ghat_{fname(m)}.npy", out[m])
    return times


def _worker(args):
    wid, methods, h_lo, h_hi = args
    from .game.safe_lp import get_solver, OpenSpielAuditor
    from .game.sequence_form import get_sequence_form
    L = get_solver(); aud = OpenSpielAuditor(get_sequence_form(), L.v_star)
    F = np.load(D / "families.npz"); G = F["G"]; hist_opp = F["hist_opp"]
    gh = {m: np.load(D / f"ghat_{fname(m)}.npy", mmap_mode="r") for m in methods}
    nN = len(N_BUDGETS); n = h_hi - h_lo
    u = np.zeros((len(methods), n, nN)); ex = np.zeros((len(methods), n, nN)); ok = np.zeros((len(methods), n, nN), dtype=bool)
    t0 = time.time()
    for i, h in enumerate(range(h_lo, h_hi)):
        g_true = G[hist_opp[h]]
        for a, m in enumerate(methods):
            for j in range(nN):
                okk, pol, x = L.solve_safe(np.asarray(gh[m][h, j], dtype=np.float64), EPS)
                ok[a, i, j] = okk; u[a, i, j] = x @ g_true; ex[a, i, j] = aud.exploitability_of_learner(pol)
        if wid == 0 and (i + 1) % 25 == 0:
            print(f"  worker0 {i+1}/{n} ({time.time()-t0:.0f}s)", flush=True)
    return h_lo, u, ex, ok, L.lp0.n_fail


def solve(workers=4):
    F = np.load(D / "families.npz"); H = F["obs"].shape[0]; nN = len(N_BUDGETS)
    bounds = np.linspace(0, H, workers + 1).astype(int)
    with mp.get_context("spawn").Pool(workers) as pool:
        res = pool.map(_worker, [(w, METHODS, int(bounds[w]), int(bounds[w + 1])) for w in range(workers)])
    u = np.zeros((len(METHODS), H, nN)); ex = np.zeros_like(u); ok = np.zeros(u.shape, dtype=bool); fails = 0
    for h_lo, uu, ee, kk, nf in res:
        u[:, h_lo:h_lo + uu.shape[1]] = uu; ex[:, h_lo:h_lo + uu.shape[1]] = ee; ok[:, h_lo:h_lo + uu.shape[1]] = kk; fails += nf
    # FIXED-NE: our equilibrium blueprint, no modelling; audited once
    from .data.datasets import load_population
    from .game.safe_lp import get_solver, OpenSpielAuditor
    from .game.sequence_form import get_sequence_form
    pop = load_population(); L = get_solver(); aud = OpenSpielAuditor(get_sequence_form(), L.v_star)
    u_ne = F["G"][F["hist_opp"]] @ pop["x_nash"]; ex_ne = aud.exploitability_of_learner(pop["blueprint0"])
    np.savez_compressed(D / "solve.npz", methods=np.array(METHODS), u=u, expl=ex, ok=ok, lp_failures=fails, u_fixed_ne=u_ne, expl_fixed_ne=ex_ne)
    return fails, ex_ne


if __name__ == "__main__":
    t0 = time.time(); meta = {"eps": EPS, "kappa": KAPPA, "neural_runs": NEURAL, "prior_runs": PRIORS, "methods": ["FIXED-NE"] + METHODS}
    meta["predict_time_s"] = predict(); meta["predict_wall_s"] = time.time() - t0
    t1 = time.time(); fails, ex_ne = solve(); meta["solve_wall_s"] = time.time() - t1; meta["lp_failures"] = int(fails); meta["expl_fixed_ne"] = float(ex_ne)
    save_json(meta, D / "eval_meta.json"); print(json.dumps(meta, indent=1)); print(f"total {time.time()-t0:.0f}s")
