"""Part B of the decision-compression study (REPORT_LEDUC_DECISION_COMPRESSION.md): policy autoencoder q -> z -> q_hat with
four losses (RECON, JAC-OPP, G-MSE, REGRET response-bank) at d in {2, 4, 8, 16, 32, 64}, seeds {0, 1, 2}.
Stages: prep (training data + Jacobian weights), select (REGRET beta/lambda on val at d = 8, seed 0), train (all runs),
eval (exact LPs + audit on val / test / OOD, behavioural and policy metrics, test latents).
Outputs in outputs/dcomp/: ae_data.npz, ae_select.json, ae_runs/*.pt, ae_eval/*.npz."""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, sys, time, multiprocessing as mp
import numpy as np
from .common import save_json
from .dc_common import D, EPS_LIST, log, load_sets, HandDist, kl_rows, policy_kl, fraction, LPPool

ARMS = ["RECON", "JAC-OPP", "G-MSE", "REGRET"]; DIMS = [2, 4, 8, 16, 32, 64]; SEEDS = [0, 1, 2]
POSTHOC_ARMS = ["OBS-RECON"]            # post-hoc behaviour-optimal arm: KL between observable hand distributions (declared deviation)
STEPS = 6000; BATCH = 256; LR = 1e-3; WD = 1e-4; WARM = 200; N_GEN = 5000
SEL_GRID = [(100.0, 0.0), (100.0, 0.1), (1000.0, 0.0), (1000.0, 0.1)]
RUNS = D / "ae_runs"; EVALS = D / "ae_eval"


def run_name(arm, d, seed, beta=None, lam=None):
    return f"{arm}_d{d}_s{seed}" + (f"_b{beta:g}_l{lam:g}" if beta is not None else "")


# ----------------------------------------------------------------------------------------------- data
def _jac_chunk(Q):
    from .jacopp_gate import Jac
    jc = Jac(); out = np.zeros((len(Q), 144))
    for i, q in enumerate(Q):
        J, _, _, _ = jc.jac(q); out[i], _ = jc.proj_norm2(J)
    return out


def prep():
    from .data.opponents import OpponentGenerator, FAMILIES
    from .game.leduc_tree import get_tree
    from .game.symmetry import get_symmetry
    from .game.sequence_form import get_sequence_form
    from .game.policy_utils import RankPolicyToG
    S, pop = load_sets(); tree, sym, sf = get_tree(), get_symmetry(), get_sequence_form(); r2g = RankPolicyToG(tree, sf, sym)
    gen = OpponentGenerator(tree, sym, pop["blueprint1_rank"], 2024)
    assert np.abs(gen.generate(FAMILIES[0], 0)[0] - pop["rank_policies"][0]).max() == 0
    Qg = np.array([gen.generate(f, 10000 + k)[0] for f in FAMILIES for k in range(N_GEN)])
    Q = np.concatenate([S["train"]["Q"], Qg]); G = r2g.g(Q)
    with mp.get_context("spawn").Pool(4) as p:
        W = np.concatenate(p.map(_jac_chunk, np.array_split(Q, 40)))
    Wn = W / np.maximum(W.mean(1, keepdims=True), 1e-12)
    np.savez_compressed(D / "ae_data.npz", Q=Q.astype(np.float32), G=G.astype(np.float32), W=Wn.astype(np.float32),
                        B=S["train"]["X"][0.10].astype(np.float32), n_orig=len(S["train"]["Q"]))
    log(f"prep: {len(Q)} training policies ({len(Qg)} generated), Jacobian weights mean-1 per opponent")


def prep_obs():
    """Targets for OBS-RECON: exact observable hand distributions of the training policies, plus the likelihood tables."""
    S, pop = load_sets(); hd = HandDist(pop); A = np.load(D / "ae_data.npz"); Q = A["Q"].astype(np.float64)
    P = np.concatenate([hd(Q[i:i + 2000]) for i in range(0, len(Q), 2000)]); lik = hd.lik; M = lik.M.tocoo()
    np.savez_compressed(D / "ae_obs.npz", P=P.astype(np.float32), M_row=M.row, M_col=M.col, n_pairs=lik.n_pairs, pair_logprior=lik.pair_logprior,
                        pair_type=lik.pair_type, reach=hd.reach, logC=hd.logC, n_types=lik.tab.n_types)
    log(f"prep_obs: {P.shape} observable targets, {lik.n_pairs} likelihood pairs")


class TorchHandDist:
    """log p_q(o) over the reachable observation types, differentiable in q (mirrors dc_common.HandDist)."""
    def __init__(self, O):
        import torch
        self.t = torch; n = int(O["n_pairs"])
        self.M = torch.sparse_coo_tensor(np.stack([O["M_row"], O["M_col"]]), torch.ones(len(O["M_row"])), (n, 432)).coalesce()
        self.lp = torch.as_tensor(O["pair_logprior"], dtype=torch.float32); self.pt = torch.as_tensor(O["pair_type"], dtype=torch.long)
        self.reach = torch.as_tensor(O["reach"], dtype=torch.bool); self.logC = torch.as_tensor(O["logC"], dtype=torch.float32); self.T = int(O["n_types"])

    def log_p(self, lq):                                             # lq (B, 144, 3) log-policy, 0 on illegal slots
        t = self.t; B = lq.shape[0]; ll = t.sparse.mm(self.M, lq.reshape(B, -1).T).T + self.lp[None]          # (B, n_pairs)
        idx = self.pt[None].expand(B, -1); m = t.full((B, self.T), -1e30).scatter_reduce(1, idx, ll, "amax").detach()
        s = t.zeros((B, self.T)).scatter_add(1, idx, t.exp(ll - m.gather(1, idx))); lt = (m + t.log(s.clamp_min(1e-38)))[:, self.reach] + self.logC[None]
        return lt - t.logsumexp(lt, 1, keepdim=True)


# ----------------------------------------------------------------------------------------------- model / training
def make_model(d, legal):
    import torch, torch.nn as nn
    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(432, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, d))
            self.dec = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 432))
            self.register_buffer("legal", torch.as_tensor(legal, dtype=torch.bool))
        def forward(self, q):
            z = self.enc(q.reshape(len(q), -1)); lg = self.dec(z).reshape(-1, 144, 3).masked_fill(~self.legal[None], float("-inf"))
            return z, torch.log_softmax(lg, -1)
    return AE()


def train_one(args):
    arm, d, seed, beta, lam, name = args
    import torch
    from .game.leduc_tree import get_tree
    from .game.symmetry import get_symmetry
    from .game.sequence_form import get_sequence_form
    from .game.policy_utils import RankPolicyToG
    from .models.torch_g import TorchRankPolicyToG
    torch.set_num_threads(1); torch.manual_seed(seed); rng = np.random.default_rng([seed, 77])
    if (RUNS / f"{name}.pt").exists():
        return name, 0.0
    t0 = time.time(); A = np.load(D / "ae_data.npz"); sym = get_symmetry(); legal = np.asarray(sym.rank_legal_mask[1], bool)
    Q = torch.as_tensor(A["Q"]); G = torch.as_tensor(A["G"]); W = torch.as_tensor(A["W"]); B = torch.as_tensor(A["B"])
    tg = TorchRankPolicyToG(RankPolicyToG(get_tree(), get_sequence_form(), sym)) if arm in ("G-MSE", "REGRET") else None
    if arm == "OBS-RECON":
        O = np.load(D / "ae_obs.npz"); thd = TorchHandDist(O); Pt = torch.as_tensor(O["P"]); lPt = torch.log(Pt.clamp_min(1e-30))
    model = make_model(d, legal); opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARM) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / STEPS))))
    lq = torch.log(Q.clamp_min(1e-30)); hist = []
    for step in range(STEPS):
        ix = torch.as_tensor(rng.integers(0, len(Q), BATCH)); q = Q[ix]
        z, lqh = model(q)
        lqh0 = torch.where(model.legal[None], lqh, torch.zeros_like(lqh))                          # finite on illegal slots
        kl = torch.where(model.legal[None], q * (lq[ix] - lqh0), torch.zeros_like(q)).sum(2)         # (B, 144)
        if arm == "RECON":
            loss = kl.mean()
        elif arm == "JAC-OPP":
            loss = (kl * W[ix]).mean()
        elif arm == "OBS-RECON":
            loss = (Pt[ix] * (lPt[ix] - thd.log_p(lqh0))).sum(1).mean()
        else:
            gh = tg(lqh.exp()); gmse = ((gh - G[ix]) ** 2).sum(1).mean()
            if arm == "G-MSE":
                loss = gmse
            else:
                val = G[ix] @ B.T; p = torch.softmax(beta * (gh @ B.T), 1)
                loss = (val.max(1).values - (p * val).sum(1)).mean() + lam * gmse
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 500 == 0 or step == STEPS - 1:
            hist.append((step, loss.item()))
    torch.save({"state": model.state_dict(), "arm": arm, "d": d, "seed": seed, "beta": beta, "lam": lam, "hist": hist, "train_s": time.time() - t0},
               RUNS / f"{name}.pt")
    return name, time.time() - t0


def train_many(jobs, workers=4):
    RUNS.mkdir(parents=True, exist_ok=True)
    with mp.get_context("spawn").Pool(workers) as p:
        for name, s in p.imap_unordered(train_one, jobs):
            log(f"  trained {name} ({s:.0f}s)")


# ----------------------------------------------------------------------------------------------- evaluation
class Evaluator:
    def __init__(self):
        import torch
        from .game.leduc_tree import get_tree
        from .game.symmetry import get_symmetry
        from .game.sequence_form import get_sequence_form
        from .game.policy_utils import RankPolicyToG
        torch.set_num_threads(4); self.torch = torch
        self.S, pop = load_sets(); self.hd = HandDist(pop); self.sym = get_symmetry(); self.legal = np.asarray(self.sym.rank_legal_mask[1], bool)
        self.r2g = RankPolicyToG(get_tree(), get_sequence_form(), self.sym)
        tr = self.S["train"]; self.pbar = self.hd(tr["Q"]).mean(0); self.qbar = tr["Q"].mean(0); self.gbar = tr["G"].mean(0)
        for s in self.S.values():
            s["P"] = self.hd(s["Q"])
        self.pool = LPPool(4, audit=True)

    def evaluate(self, name, sets=("val", "test", "ood")):
        torch = self.torch; ck = torch.load(RUNS / f"{name}.pt", weights_only=False); model = make_model(ck["d"], self.legal)
        model.load_state_dict(ck["state"]); model.eval(); out = {}
        for sname in sets:
            s = self.S[sname]
            with torch.no_grad():
                z, lqh = model(torch.as_tensor(s["Q"], dtype=torch.float32))
            qh = lqh.exp().double().numpy(); qh = np.where(self.legal[None], qh, 0.0); qh /= qh.sum(2, keepdims=True); gh = self.r2g.g(qh)
            out[f"{sname}_kl_beh"] = kl_rows(s["P"], self.hd(qh)); out[f"{sname}_kl_beh0"] = kl_rows(s["P"], np.broadcast_to(self.pbar, s["P"].shape))
            out[f"{sname}_kl_pol"] = policy_kl(s["Q"], qh, self.legal); out[f"{sname}_kl_pol0"] = policy_kl(s["Q"], np.broadcast_to(self.qbar, s["Q"].shape), self.legal)
            out[f"{sname}_gerr"] = ((gh - s["G"]) ** 2).sum(1); out[f"{sname}_gvar"] = ((s["G"] - self.gbar) ** 2).sum(1)
            for e in (EPS_LIST if sname == "test" else [0.10]):
                X, ex, ok = self.pool.solve(gh, e); out[f"{sname}_u_{e}"] = (X * s["G"]).sum(1); out[f"{sname}_expl_{e}"] = ex; out[f"{sname}_ok_{e}"] = ok
            if sname == "test":
                out["test_z"] = z.numpy()
        EVALS.mkdir(parents=True, exist_ok=True); np.savez_compressed(EVALS / f"{name}.npz", **out)
        return out


def main(stages):
    D.mkdir(parents=True, exist_ok=True); wall_f = D / "ae_wallclock.json"; wall = json.loads(wall_f.read_text()) if wall_f.exists() else {}
    if "prep" in stages and not (D / "ae_data.npz").exists():
        t = time.time(); prep(); wall["prep_s"] = time.time() - t; save_json(wall, wall_f)
    if "select" in stages and not (D / "ae_select.json").exists():
        t = time.time(); jobs = [("REGRET", 8, 0, b, l, run_name("REGRET", 8, 0, b, l)) for b, l in SEL_GRID]; train_many(jobs)
        ev = Evaluator(); sel = {}
        for *_, name in jobs:
            o = ev.evaluate(name, sets=("val",)); s = ev.S["val"]; sel[name] = fraction(o["val_u_0.1"], s["V0"], s["V"][0.10])
            log(f"  select {name}: val fraction {sel[name]:.4f}")
        ev.pool.close(); best = max(sel, key=sel.get); b, l = [(b, l) for b, l in SEL_GRID if run_name("REGRET", 8, 0, b, l) == best][0]
        save_json({"val_fraction": sel, "selected": best, "beta": b, "lam": l, "wall_s": time.time() - t}, D / "ae_select.json"); wall["select_s"] = time.time() - t; save_json(wall, wall_f)
    if "train" in stages:
        selj = json.loads((D / "ae_select.json").read_text()); beta, lam = selj["beta"], selj["lam"]
        t = time.time(); jobs = []
        for d in DIMS:
            for seed in SEEDS:
                for arm in ARMS:
                    name = run_name(arm, d, seed)
                    if arm == "REGRET" and d == 8 and seed == 0 and not (RUNS / f"{name}.pt").exists():
                        import shutil; shutil.copy(RUNS / f"{selj['selected']}.pt", RUNS / f"{name}.pt")      # the selected run is the d = 8, seed 0 run
                    jobs.append((arm, d, seed, beta if arm == "REGRET" else None, lam if arm == "REGRET" else None, name))
        jobs.sort(key=lambda j: j[0] in ("G-MSE", "REGRET"), reverse=True)                            # long jobs first
        train_many(jobs); wall["train_s"] = wall.get("train_s", 0) + time.time() - t; save_json(wall, wall_f)
    if "posthoc" in stages:                                         # declared post-hoc arm OBS-RECON: train + evaluate
        t = time.time()
        if not (D / "ae_obs.npz").exists():
            prep_obs()
        train_many([(a, d, seed, None, None, run_name(a, d, seed)) for a in POSTHOC_ARMS for d in DIMS for seed in SEEDS])
        ev = Evaluator()
        for a in POSTHOC_ARMS:
            for d in DIMS:
                for seed in SEEDS:
                    name = run_name(a, d, seed)
                    if not (EVALS / f"{name}.npz").exists():
                        ev.evaluate(name); log(f"  evaluated {name}")
        ev.pool.close(); wall["posthoc_s"] = time.time() - t; save_json(wall, wall_f)
    if "eval" in stages:
        t = time.time(); ev = Evaluator(); n = 0
        for d in DIMS:
            for seed in SEEDS:
                for arm in ARMS:
                    name = run_name(arm, d, seed)
                    if (RUNS / f"{name}.pt").exists() and not (EVALS / f"{name}.npz").exists():
                        ev.evaluate(name); n += 1; log(f"  evaluated {name}")
        ev.pool.close(); wall["eval_s"] = wall.get("eval_s", 0) + time.time() - t; save_json(wall, wall_f)


if __name__ == "__main__":
    main(sys.argv[1:] or ["prep", "select", "train", "eval"])
