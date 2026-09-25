"""Non-stationary opponents (REPORT_LEDUC_NONSTATIONARY.md section 0): build SWITCH / DRIFT processes, simulate them,
compute oracle values of the current opponent at every checkpoint, produce g_hat for every method from the hands seen
so far (full history, windows, discounting, post-switch oracles), then exact eps-safe LP + OpenSpiel audit.
Writes outputs/nonstat/{processes.npz, solve.npz, eval_meta.json}."""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, time, multiprocessing as mp
import numpy as np
from .common import OUT, save_json

D = OUT / "nonstat"; EPS = 0.10; KAPPA = 3.0; H_LEN = 500; SWITCH_AT = 200
SW_T = [100, 200, 205, 210, 220, 250, 300, 400, 500]; DR_T = [20, 50, 100, 200, 300, 400, 500]
WINS = [25, 50, 100]; GAMMAS = [0.95, 0.98, 0.99]
BASE = ["BANK", "TAB-EM"] + [f"WIN-EM W={w}" for w in WINS] + [f"DISC-EM g={g}" for g in GAMMAS] + \
       ["JAC-opp", "JAC-opp-W50", "DEC-889k", "PRIOR-EM", "PRIOR-EM-WIN"]
ORACLES = ["POST-EM", "POST-PRIOR-EM"]
METHODS = {"SWITCH": BASE + ORACLES, "DRIFT": BASE}
CKPT = {"SWITCH": SW_T, "DRIFT": DR_T}


def lam_schedule(kind):
    t = np.arange(1, H_LEN + 1)
    return (t > SWITCH_AT).astype(float) if kind == "SWITCH" else (t - 1) / (H_LEN - 1)


def simulate_mix(tree, pol0, polA, polB, lam, n_streams, split_id, opp_id, base_seed=7):
    """Simulator with a per-hand mixture of two player-1 policies (mixture of node tables = per-infoset interpolation)."""
    from .data.simulator import node_probability_table, stream_seed, MAX_STEPS
    from .game.leduc_tree import TERMINAL, NUM_CARDS
    PA = node_probability_table(tree, pol0, polA); PB = node_probability_table(tree, pol0, polB)
    cA, cB = np.cumsum(PA, 1), np.cumsum(PB, 1)
    U = np.stack([np.random.default_rng(stream_seed(base_seed, split_id, opp_id, s)).random((len(lam), MAX_STEPS)) for s in range(n_streams)]).reshape(-1, MAX_STEPS)
    Lr = np.tile(lam, n_streams); nodes = np.zeros(U.shape[0], dtype=np.int64)
    for step in range(MAX_STEPS):
        idx = np.flatnonzero(tree.node_type[nodes] != TERMINAL)
        if len(idx) == 0:
            break
        l = Lr[idx, None]; c = (1 - l) * cA[nodes[idx]] + l * cB[nodes[idx]]
        a = np.minimum((U[idx, step][:, None] >= c).sum(1), NUM_CARDS - 1)
        child = tree.child_table[nodes[idx], a]
        for j in np.flatnonzero(child < 0):
            P = (1 - Lr[idx[j]]) * PA[nodes[idx[j]]] + Lr[idx[j]] * PB[nodes[idx[j]]]
            child[j] = tree.child_table[nodes[idx[j]], np.flatnonzero(P > 0)[-1]]
        nodes[idx] = child
    assert np.all(tree.node_type[nodes] == TERMINAL)
    return nodes.reshape(n_streams, len(lam))


def build():
    from .game.leduc_tree import get_tree
    from .game.sequence_form import get_sequence_form
    from .game.symmetry import get_symmetry
    from .game.policy_utils import RankPolicyToG
    from .game.safe_lp import get_solver
    from .data.datasets import load_population
    from .data.tokenizer import get_token_table
    t0 = time.time(); tree, sf, sym, tab = get_tree(), get_sequence_form(), get_symmetry(), get_token_table()
    pop = load_population(); L = get_solver(); r2g = RankPolicyToG(tree, sf, sym)
    GF = np.load(OUT / "gen" / "families.npz", allow_pickle=True); arch = np.flatnonzero(GF["family"] == "FAR-ARCH")
    test = np.flatnonzero(pop["split"] == 2)
    Qp = np.concatenate([pop["rank_policies"][test], GF["rank_policies"][arch]]); Gp = np.concatenate([pop["G"][test], GF["G"][arch]])
    src = [f"test:{i}" for i in test] + [f"FAR-ARCH:{GF['subfamily'][i]}:{i}" for i in arch]
    d2 = (Gp ** 2).sum(1)[:, None] + (Gp ** 2).sum(1)[None] - 2 * Gp @ Gp.T; Dist = np.sqrt(np.maximum(d2, 0))
    med = float(np.median(Dist[np.triu_indices(len(Gp), 1)]))
    rng = np.random.default_rng([2027, 1]); used = set()
    def draw(contrast):
        while True:
            a, b = (int(v) for v in rng.choice(len(Gp), 2, replace=False))
            if (a, b) in used or (contrast and Dist[a, b] <= med):
                continue
            used.add((a, b)); return a, b
    procs = [("SWITCH", "contrasting", *draw(True)) for _ in range(40)] + [("SWITCH", "random", *draw(False)) for _ in range(20)] + \
            [("DRIFT", "contrasting", *draw(True)) for _ in range(27)] + [("DRIFT", "random", *draw(False)) for _ in range(13)]
    kind = np.array([p[0] for p in procs]); sel = np.array([p[1] for p in procs]); A = np.array([p[2] for p in procs]); B = np.array([p[3] for p in procs])
    obs = np.zeros((len(procs), 2, H_LEN), dtype=np.int16)
    for i, (k, _, a, b) in enumerate(procs):
        terms = simulate_mix(tree, pop["blueprint0"], sym.expand(1, Qp[a]), sym.expand(1, Qp[b]), lam_schedule(k), 2, {"SWITCH": 30, "DRIFT": 31}[k], i)
        obs[i] = tab.obs_type_of_terminal[terms]
    # current-opponent decision vectors and oracle values at every checkpoint
    nT = max(len(SW_T), len(DR_T)); Gt = np.full((len(procs), nT, sf.n_seq[0]), np.nan); V0 = np.full((len(procs), nT), np.nan); Ve = np.full_like(V0, np.nan)
    for i, (k, _, a, b) in enumerate(procs):
        for j, t in enumerate(CKPT[k]):
            lam = lam_schedule(k)[t - 1]
            q = (1 - lam) * Qp[a] + lam * Qp[b]
            g = r2g.g(q[None])[0]; Gt[i, j] = g
            for eps, arr in ((0.0, V0), (EPS, Ve)):
                ok, pol, x = L.solve_safe(g, eps); assert ok; arr[i, j] = x @ g
    np.savez_compressed(D / "processes.npz", kind=kind, selection=sel, A=A, B=B, pair_dist=Dist[A, B], median_pool_dist=med, source=np.array(src),
                        obs=obs, G_t=Gt, V0=V0, Veps=Ve, pool_Q=Qp)
    print(f"built {len(procs)} processes ({time.time()-t0:.0f}s); median pool distance {med:.3f}", flush=True)
    return time.time() - t0


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
    torch.set_num_threads(4); t0 = time.time()
    P = np.load(D / "processes.npz", allow_pickle=True); tree, sf, sym, tab = get_tree(), get_sequence_form(), get_symmetry(), get_token_table()
    pop = load_population(); r2g = RankPolicyToG(tree, sf, sym); tg = TorchRankPolicyToG(r2g); lik = HandLikelihood(tab, sym); mask = sym.rank_legal_mask[1]
    tr = np.flatnonzero(pop["split"] == 0); bank = BankPosterior(lik, pop["rank_policies"][tr], pop["G"][tr])
    em = TabularEM(lik, mask, mask / mask.sum(1, keepdims=True), alpha=1.0, n_iter=200); T_ = tab.n_types
    nets = {n: load_trained(OUT / r) for n, r in (("JAC-opp", "runs_jacopp/a3_s0"), ("DEC-889k", "runs/decision_s0"))}
    def net(name, seqs):                                           # seqs: (n, L) obs types
        enc, head, ck = nets[name]; enc.eval(); head.eval()
        with torch.no_grad():
            z = enc(torch.as_tensor(seqs.astype(np.int64))); out = head(z)
            return (tg(out).numpy(), out.numpy()) if ck["method"] == "recon" else (out.numpy(), None)
    def disc_counts(o, t, g):
        w = g ** (t - 1 - np.arange(t)); return np.stack([np.bincount(o[h, :t].astype(np.int64), weights=w, minlength=T_) for h in range(len(o))])
    res = {}
    for kind in ("SWITCH", "DRIFT"):
        pidx = np.flatnonzero(P["kind"] == kind); o = P["obs"][pidx].reshape(-1, H_LEN); n = len(o); Ts = CKPT[kind]
        out = {m: np.zeros((n, len(Ts), sf.n_seq[0]), dtype=np.float32) for m in METHODS[kind]}
        for j, t in enumerate(Ts):
            full = o[:, :t]; win = lambda W: o[:, max(0, t - W):t]; post = o[:, SWITCH_AT:t] if (kind == "SWITCH" and t > SWITCH_AT) else full
            cf = type_counts(full, T_); out["BANK"][:, j] = bank.g_bar(cf)[0]; out["TAB-EM"][:, j] = r2g.g(em.fit(cf))
            for W in WINS:
                out[f"WIN-EM W={W}"][:, j] = r2g.g(em.fit(type_counts(win(W), T_)))
            for g in GAMMAS:
                out[f"DISC-EM g={g}"][:, j] = r2g.g(em.fit(disc_counts(o, t, g)))
            gj, qj = net("JAC-opp", full); out["JAC-opp"][:, j] = gj
            gw, qw = net("JAC-opp", win(50)); out["JAC-opp-W50"][:, j] = gw
            out["DEC-889k"][:, j] = net("DEC-889k", full)[0]
            out["PRIOR-EM"][:, j] = r2g.g(fit_em_with_row_prior(lik, mask, qj, KAPPA, cf))
            out["PRIOR-EM-WIN"][:, j] = r2g.g(fit_em_with_row_prior(lik, mask, qw, KAPPA, type_counts(win(50), T_)))
            if kind == "SWITCH":
                cp = type_counts(post, T_); out["POST-EM"][:, j] = r2g.g(em.fit(cp))
                out["POST-PRIOR-EM"][:, j] = r2g.g(fit_em_with_row_prior(lik, mask, net("JAC-opp", post)[1], KAPPA, cp))
            print(f"{kind} t={t} done ({time.time()-t0:.0f}s)", flush=True)
        for m, arr in out.items():
            np.save(D / f"ghat_{kind}_{m.replace(' ', '_').replace('=', '')}.npy", arr)
        res[kind] = n
    return time.time() - t0


def _worker(args):
    wid, kind, lo, hi = args
    from .game.safe_lp import get_solver, OpenSpielAuditor
    from .game.sequence_form import get_sequence_form
    L = get_solver(); aud = OpenSpielAuditor(get_sequence_form(), L.v_star)
    P = np.load(D / "processes.npz", allow_pickle=True); pidx = np.flatnonzero(P["kind"] == kind); Gt = P["G_t"][pidx]
    ms = METHODS[kind]; nT = len(CKPT[kind])
    gh = {m: np.load(D / f"ghat_{kind}_{m.replace(' ', '_').replace('=', '')}.npy", mmap_mode="r") for m in ms}
    u = np.zeros((len(ms), hi - lo, nT)); ex = np.zeros_like(u); ok = np.zeros(u.shape, dtype=bool); t0 = time.time()
    for i, h in enumerate(range(lo, hi)):
        p = h // 2
        for a, m in enumerate(ms):
            for j in range(nT):
                okk, pol, x = L.solve_safe(np.asarray(gh[m][h, j], dtype=np.float64), EPS)
                ok[a, i, j] = okk; u[a, i, j] = x @ Gt[p, j]; ex[a, i, j] = aud.exploitability_of_learner(pol)
        if wid == 0 and (i + 1) % 10 == 0:
            print(f"  [{kind}] worker0 {i+1}/{hi-lo} ({time.time()-t0:.0f}s)", flush=True)
    return lo, u, ex, ok


def solve(workers=4):
    from .data.datasets import load_population
    t0 = time.time(); P = np.load(D / "processes.npz", allow_pickle=True); pop = load_population(); out = {}
    for kind in ("SWITCH", "DRIFT"):
        n = 2 * int((P["kind"] == kind).sum()); b = np.linspace(0, n, workers + 1).astype(int)
        with mp.get_context("spawn").Pool(workers) as pool:
            res = pool.map(_worker, [(w, kind, int(b[w]), int(b[w + 1])) for w in range(workers)])
        ms = METHODS[kind]; nT = len(CKPT[kind]); u = np.zeros((len(ms), n, nT)); ex = np.zeros_like(u); ok = np.zeros(u.shape, dtype=bool)
        for lo, uu, ee, kk in res:
            u[:, lo:lo + uu.shape[1]] = uu; ex[:, lo:lo + uu.shape[1]] = ee; ok[:, lo:lo + uu.shape[1]] = kk
        pidx = np.flatnonzero(P["kind"] == kind); u_ne = np.einsum("ptn,n->pt", P["G_t"][pidx][:, :nT], pop["x_nash"])
        out[f"{kind}_u"] = u; out[f"{kind}_expl"] = ex; out[f"{kind}_ok"] = ok; out[f"{kind}_u_fixed_ne"] = u_ne; out[f"{kind}_methods"] = np.array(ms)
    np.savez_compressed(D / "solve.npz", **out)
    return time.time() - t0


if __name__ == "__main__":
    D.mkdir(parents=True, exist_ok=True); meta = {"eps": EPS, "kappa": KAPPA, "checkpoints": CKPT, "methods": METHODS}
    meta["build_s"] = build(); meta["predict_s"] = predict(); meta["solve_s"] = solve()
    save_json(meta, D / "eval_meta.json"); print(json.dumps({k: v for k, v in meta.items() if k.endswith("_s")}))
