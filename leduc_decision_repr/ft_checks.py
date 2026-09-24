"""Pre-training checks for decision-aware fine-tuning (re-run of the V3 SPO+ checks plus end-to-end chain checks)."""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import time, json
import numpy as np, torch
import torch.nn.functional as F
from .common import OUT, save_json
from .data.datasets import load_population
from .game.safe_lp import get_solver
from .game.spo_plus import SPOPlus
from .train import load_trained, TorchRankPolicyToG, RankPolicyToG
from .game.leduc_tree import get_tree
from .game.sequence_form import get_sequence_form
from .game.symmetry import get_symmetry
from .data.datasets import find_dataset_dir, load_split

torch.set_num_threads(1)
EPS = 0.10
pop = load_population(); L = get_solver(); spo = SPOPlus(L, EPS)
X = np.load(OUT / "weights_v3" / "xstar_train_eps0.1.npy"); tr = np.flatnonzero(pop["split"] == 0)
rng = np.random.default_rng(0); res = {}

# ---------------------------------------------------------------- 1. SPO+ properties on g_hat directly
opps = rng.choice(tr, 20, replace=False)
cache_ok = [abs(pop["G"][k] @ X[k] - pop["V_oracle"][k, 2]) for k in opps]
loss_truth, nonneg, convex, fd_err, bound = [], [], [], [], []
for k in opps:
    g = pop["G"][k]; xg = X[k]
    l0, _ = spo.loss_and_grad(g, g, xg); loss_truth.append(abs(l0))
    for scale in [0.05, 0.2, 1.0]:
        a = g + scale * rng.normal(size=g.shape) * np.abs(g).mean(); b = g + scale * rng.normal(size=g.shape) * np.abs(g).mean()
        la, ga = spo.loss_and_grad(a, g, xg); lb, _ = spo.loss_and_grad(b, g, xg); lm, _ = spo.loss_and_grad(0.5 * (a + b), g, xg)
        nonneg.append(min(la, lb)); convex.append(0.5 * (la + lb) - lm)
        x_hat = L.solve_safe(a, EPS)[2]; regret = g @ xg - g @ x_hat; bound.append(la - regret)
        u = rng.normal(size=g.shape); u /= np.linalg.norm(u); h = 1e-6
        fd = (spo.loss_and_grad(a + h * u, g, xg)[0] - spo.loss_and_grad(a - h * u, g, xg)[0]) / (2 * h)
        fd_err.append(abs(fd - ga @ u) / max(abs(fd), 1e-9))
res["direct"] = {"n_opponents": 20, "cache_value_vs_V_oracle_max_abs": float(max(cache_ok)), "loss_at_truth_max_abs": float(max(loss_truth)),
                 "min_loss": float(min(nonneg)), "min_convexity_gap": float(min(convex)), "grad_vs_fd_rel_err_median": float(np.median(fd_err)),
                 "grad_vs_fd_rel_err_max": float(max(fd_err)), "min_loss_minus_regret": float(min(bound)),
                 "all_pass": bool(max(loss_truth) < 1e-9 and min(nonneg) > -1e-9 and min(convex) > -1e-9 and min(bound) > -1e-9)}
print("direct", res["direct"], flush=True)

# ---------------------------------------------------------------- 2. end-to-end chain checks (double precision, eval mode)
ds = find_dataset_dir(); train = load_split(ds, "train")
fn = spo.torch_function()
def chain_check(run_dir, method):
    enc, head, ck = load_trained(run_dir); enc.double(); head.double(); enc.eval(); head.eval()
    r2g = TorchRankPolicyToG(RankPolicyToG(get_tree(), get_sequence_form(), get_symmetry())).double()
    idx = rng.choice(len(train["opp_ids"]), 4, replace=False); ids = train["opp_ids"][idx]
    x = torch.as_tensor(train["obs_types"][idx, 0, :100].astype(np.int64))
    G = torch.as_tensor(pop["G"][ids], dtype=torch.float64); Xs = torch.as_tensor(X[ids], dtype=torch.float64)
    params = list(enc.parameters()) + list(head.parameters())
    def f():
        out = head(enc(x)); g_hat = r2g(out) if method == "recon" else out
        return fn.apply(g_hat, G, Xs).mean(), g_hat
    val, g_hat = f(); grads = torch.autograd.grad(val, params)
    out = []
    for trial in range(3):
        u = [torch.randn_like(p) for p in params]; nu = torch.sqrt(sum((v ** 2).sum() for v in u)); u = [v / nu for v in u]
        ana = float(sum((g * v).sum() for g, v in zip(grads, u)))
        for h in [1e-4, 1e-5]:
            with torch.no_grad():
                for p, v in zip(params, u): p.add_(h * v)
                fp = float(f()[0])
                for p, v in zip(params, u): p.sub_(2 * h * v)
                fm = float(f()[0])
                for p, v in zip(params, u): p.add_(h * v)
            fd = (fp - fm) / (2 * h); out.append({"trial": trial, "h": h, "analytic": ana, "fd": fd, "rel_err": abs(fd - ana) / max(abs(fd), 1e-12)})
    # regret bound at the model's own predictions
    gh = g_hat.detach().numpy(); lb = []
    for i, k in enumerate(ids):
        l, _ = spo.loss_and_grad(gh[i], pop["G"][k], X[k]); xh = L.solve_safe(gh[i], EPS)[2]
        lb.append(l - (pop["G"][k] @ X[k] - pop["G"][k] @ xh))
    return {"directional": out, "median_rel_err": float(np.median([o["rel_err"] for o in out])), "min_loss_minus_regret": float(min(lb)),
            "g_hat_scale": float(np.abs(gh).mean()), "g_true_scale": float(np.abs(pop["G"][ids]).mean())}
res["chain_recon_jac"] = chain_check(OUT / "runs_v3" / "recjac889k_s0", "recon"); print("recon chain", {k: v for k, v in res["chain_recon_jac"].items() if k != "directional"}, flush=True)
res["chain_dec889k"] = chain_check(OUT / "runs" / "decision_s0", "decision"); print("dec chain", {k: v for k, v in res["chain_dec889k"].items() if k != "directional"}, flush=True)

# ---------------------------------------------------------------- 3. LP timing on a full batch of 32 (RECON-JAC predictions)
enc, head, ck = load_trained(OUT / "runs_v3" / "recjac889k_s0")
r2g = TorchRankPolicyToG(RankPolicyToG(get_tree(), get_sequence_form(), get_symmetry()))
idx = rng.choice(len(train["opp_ids"]), 32, replace=False); ids = train["opp_ids"][idx]
with torch.no_grad():
    gh = r2g(head(enc(torch.as_tensor(train["obs_types"][idx, 0, :50].astype(np.int64))))).numpy().astype(np.float64)
t = time.time()
for i, k in enumerate(ids):
    spo.loss_and_grad(gh[i], pop["G"][k], X[k])
res["lp_timing"] = {"batch": 32, "seconds_per_batch": time.time() - t, "ms_per_lp": (time.time() - t) / 32 * 1000}
print("timing", res["lp_timing"], flush=True)
save_json(res, OUT / "ft" / "checks.json")
