"""Change detection + reset (REPORT_LEDUC_CHANGEDETECT.md section 0).
Stages (each resumable; outputs in outputs/cpd/):
  build      calibration processes (CAL-SWITCH, CAL-STAT) and the TEST-STAT control, exact values   -> processes.npz
  calibrate  CUSUM grid on CAL processes, exact LPs (not audited), constrained selection             -> calib.npz
  test       all methods on test SWITCH / TEST-STAT / DRIFT, exact LP + OpenSpiel audit              -> test.npz
  floor      exact per-hand KL between the switch pairs, Lorden delays, known-model CUSUM delays    -> floor.npz
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, sys, time, multiprocessing as mp
import numpy as np
from .common import OUT, save_json
from .ns_eval import simulate_mix, lam_schedule, H_LEN, SWITCH_AT, EPS, KAPPA, SW_T, DR_T

D = OUT / "cpd"; NS = OUT / "nonstat"; BLOCK = 5; CLIP = 30.0; KNOWN_H = float(np.log(1000.0)); KEY = 1024
SW_FINE = sorted(set(SW_T) | {215, 225, 230, 240, 260, 280}); ST_T = [100, 200, 300, 400, 500]
GRID = [(wa, c, tau) for wa in (0, 10, 25) for c in (0.0, 0.05, 0.15) for tau in (2.0, 3.0, 4.0, 6.0, 8.0)]
CKPTS = {"SW": SW_FINE, "ST": ST_T, "DR": DR_T, "CSW": SW_T, "CST": ST_T}
DISC = {0.98: 50, 0.95: 20}                                        # gamma -> prior window 1/(1-gamma)
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def cfg_name(c):
    return f"CPD[Wa={c[0]},c={c[1]},tau={c[2]:g}]"


# ----------------------------------------------------------------------------------------------- build
def build():
    from .game.leduc_tree import get_tree
    from .game.symmetry import get_symmetry
    from .game.safe_lp import get_solver
    from .data.datasets import load_population
    from .data.tokenizer import get_token_table
    from .game.policy_utils import RankPolicyToG
    from .game.sequence_form import get_sequence_form
    from .gen_families import Builder
    tree, sym, tab, sf = get_tree(), get_symmetry(), get_token_table(), get_sequence_form()
    pop = load_population(); L = get_solver(); r2g = RankPolicyToG(tree, sf, sym)
    NP = np.load(NS / "processes.npz", allow_pickle=True); out = {}

    def values(g):
        v = []
        for eps in (0.0, EPS):
            ok, pol, x = L.solve_safe(g, eps); assert ok; v.append(float(x @ g))
        return v

    def sim(qa, qb, switch, split_id, i):
        lam = lam_schedule("SWITCH") if switch else np.zeros(H_LEN)
        terms = simulate_mix(tree, pop["blueprint0"], sym.expand(1, qa), sym.expand(1, qb), lam, 2, split_id, i)
        return tab.obs_type_of_terminal[terms]

    def pairs(G, rng, n_con, n_rand):
        d2 = (G ** 2).sum(1)[:, None] + (G ** 2).sum(1)[None] - 2 * G @ G.T; Dist = np.sqrt(np.maximum(d2, 0))
        med = float(np.median(Dist[np.triu_indices(len(G), 1)])); used = set()
        def draw(contrast):
            while True:
                a, b = (int(v) for v in rng.choice(len(G), 2, replace=False))
                if (a, b) in used or (contrast and Dist[a, b] <= med):
                    continue
                used.add((a, b)); return a, b
        pr = [(*draw(True), "contrasting") for _ in range(n_con)] + [(*draw(False), "random") for _ in range(n_rand)]
        return pr, Dist, med

    # ---- test SWITCH / DRIFT: reuse the non-stationary processes unchanged
    sw = np.flatnonzero(NP["kind"] == "SWITCH"); dr = np.flatnonzero(NP["kind"] == "DRIFT"); Qp = NP["pool_Q"]
    out.update(SW_obs=NP["obs"][sw], SW_QA=Qp[NP["A"][sw]], SW_QB=Qp[NP["B"][sw]], SW_sel=NP["selection"][sw], SW_dist=NP["pair_dist"][sw],
               SW_GA=NP["G_t"][sw][:, 1], SW_GB=NP["G_t"][sw][:, 2], SW_V0A=NP["V0"][sw][:, 1], SW_V0B=NP["V0"][sw][:, 2],
               SW_VeA=NP["Veps"][sw][:, 1], SW_VeB=NP["Veps"][sw][:, 2],
               DR_obs=NP["obs"][dr], DR_G=NP["G_t"][dr][:, :len(DR_T)], DR_V0=NP["V0"][dr][:, :len(DR_T)], DR_Ve=NP["Veps"][dr][:, :len(DR_T)],
               DR_sel=NP["selection"][dr])
    # ---- TEST-STAT: 40 opponents from the same test pool
    GF = np.load(OUT / "gen" / "families.npz", allow_pickle=True); arch = np.flatnonzero(GF["family"] == "FAR-ARCH"); test = np.flatnonzero(pop["split"] == 2)
    assert np.allclose(Qp, np.concatenate([pop["rank_policies"][test], GF["rank_policies"][arch]]))
    Gp = np.concatenate([pop["G"][test], GF["G"][arch]])
    st = np.random.default_rng([2027, 3]).choice(len(Qp), 40, replace=False)
    obs = np.stack([sim(Qp[a], Qp[a], False, 34, i) for i, a in enumerate(st)]); V = np.array([values(Gp[a]) for a in st])
    out.update(ST_obs=obs, ST_QA=Qp[st], ST_GA=Gp[st], ST_V0A=V[:, 0], ST_VeA=V[:, 1], ST_pool_idx=st)
    log("TEST-STAT built")
    # ---- calibration pool: 150 validation opponents + 48 fresh archetypes (k = 100..111)
    B = Builder(); val = np.flatnonzero(pop["split"] == 1)
    Qa, src = [], [f"val:{i}" for i in val]
    for kind in ("ROCK", "CALLING_STATION", "MANIAC", "TAG"):
        for k in range(100, 112):
            q, _ = B.arch(kind, k); B.check(q); Qa.append(q); src.append(f"arch:{kind}:{k}")
    Qc = np.concatenate([pop["rank_policies"][val], np.array(Qa)]); Gc = np.concatenate([pop["G"][val], r2g.g(np.array(Qa))])
    pr, Dist, med = pairs(Gc, np.random.default_rng([2027, 2]), 40, 20)
    A = np.array([p[0] for p in pr]); Bb = np.array([p[1] for p in pr])
    obs = np.stack([sim(Qc[a], Qc[b], True, 32, i) for i, (a, b, _) in enumerate(pr)])
    VA = np.array([values(Gc[a]) for a in A]); VB = np.array([values(Gc[b]) for b in Bb])
    out.update(CSW_obs=obs, CSW_QA=Qc[A], CSW_QB=Qc[Bb], CSW_sel=np.array([p[2] for p in pr]), CSW_dist=Dist[A, Bb], CSW_GA=Gc[A], CSW_GB=Gc[Bb],
               CSW_V0A=VA[:, 0], CSW_V0B=VB[:, 0], CSW_VeA=VA[:, 1], CSW_VeB=VB[:, 1], CAL_median_dist=med, CAL_src=np.array(src))
    cst = np.random.default_rng([2027, 4]).choice(len(Qc), 40, replace=False)
    obs = np.stack([sim(Qc[a], Qc[a], False, 33, i) for i, a in enumerate(cst)]); V = np.array([values(Gc[a]) for a in cst])
    out.update(CST_obs=obs, CST_QA=Qc[cst], CST_GA=Gc[cst], CST_V0A=V[:, 0], CST_VeA=V[:, 1], CST_pool_idx=cst)
    np.savez_compressed(D / "processes.npz", **out)
    log(f"calibration built: pool {len(Qc)}, median pair distance {med:.3f}")


def cur_values(P, s, T_list):
    """Per process: (G, V0, Veps) of the current opponent at each checkpoint -> (n_proc, nT, dim), (n_proc, nT), (n_proc, nT)."""
    if s == "DR":
        return P["DR_G"], P["DR_V0"], P["DR_Ve"]
    post = np.array([t > SWITCH_AT for t in T_list]) if s in ("SW", "CSW") else np.zeros(len(T_list), bool)
    has_b = f"{s}_GB" in P.files
    GB = P[f"{s}_GB"] if has_b else P[f"{s}_GA"]; V0B = P[f"{s}_V0B"] if has_b else P[f"{s}_V0A"]; VeB = P[f"{s}_VeB"] if has_b else P[f"{s}_VeA"]
    G = np.where(post[None, :, None], GB[:, None], P[f"{s}_GA"][:, None])
    return G, np.where(post[None], V0B[:, None], P[f"{s}_V0A"][:, None]), np.where(post[None], VeB[:, None], P[f"{s}_VeA"][:, None])


# ----------------------------------------------------------------------------------------------- predictive models
class Pred:
    def __init__(self):
        import torch
        from .train import load_trained
        from .game.symmetry import get_symmetry
        from .data.tokenizer import get_token_table
        from .data.datasets import load_population
        from .baselines.likelihood import HandLikelihood
        torch.set_num_threads(4); self.torch = torch
        self.enc, self.head, _ = load_trained(OUT / "runs_jacopp" / "a3_s0")
        self.sym, self.tab = get_symmetry(), get_token_table(); self.lik = HandLikelihood(self.tab, self.sym)
        pop = load_population(); Qtr = pop["rank_policies"][pop["split"] == 0]
        ll = self.type_ll(Qtr); m = ll.max(0); self.pop_ll = m + np.log(np.exp(ll - m).mean(0))      # exact population prior predictive

    def type_ll(self, q):
        return self.lik.type_loglik(np.log(np.maximum(q, 1e-300)).reshape(len(q), -1))

    def q(self, O, reqs, chunk_tokens=60000):
        """reqs (m, 3) rows (h, s, e), e > s -> JAC-opp reconstruction-head policies (m, 144, 3) from hands O[h, s:e]."""
        out = np.zeros((len(reqs), 144, 3)); lens = reqs[:, 2] - reqs[:, 1]; order = np.argsort(lens, kind="stable"); i0 = 0
        while i0 < len(order):
            i1 = i0 + 1
            while i1 < len(order) and (i1 - i0 + 1) * lens[order[i1]] <= chunk_tokens:
                i1 += 1
            ch = order[i0:i1]; Lm = int(lens[ch].max()); x = np.zeros((len(ch), Lm), np.int64); m = np.zeros((len(ch), Lm), bool)
            for r, i in enumerate(ch):
                h, s, e = reqs[i]; x[r, :e - s] = O[h, s:e]; m[r, :e - s] = True
            with self.torch.no_grad():
                out[ch] = self.head(self.enc(self.torch.as_tensor(x), self.torch.as_tensor(m))).numpy()
            i0 = i1
        return out


# ----------------------------------------------------------------------------------------------- detectors
def run_detector(pred, O, cfgs, tag):
    """Lockstep CUSUM for all configurations over histories O (n, 500).  Returns alarms (m, 4): cfg, history, hand t (0-based), new start."""
    n = len(O); K = len(cfgs); WA = np.array([c[0] for c in cfgs]); C = np.array([c[1] for c in cfgs]); TAU = np.array([c[2] for c in cfgs])
    k = np.zeros((K, n), np.int64); S = np.zeros((K, n)); z = np.zeros((K, n), np.int64); hold = np.zeros((K, n), bool)
    hh = np.broadcast_to(np.arange(n), (K, n)); alarms = []; n_enc = 0; t0 = time.time()
    for e in range(0, H_LEN, BLOCK):
        S[hold] = 0.0; z[hold] = e; hold[:] = False
        nb = min(BLOCK, H_LEN - e); nxt = O[:, e:e + nb].astype(np.int64)
        alt_s = np.maximum(0, e - WA)
        keys = np.unique(np.concatenate([(hh * KEY + k).ravel()] + [np.arange(n) * KEY + max(0, e - wa) for wa in set(WA.tolist()) if wa > 0]))
        kh, ks = keys // KEY, keys % KEY; LL = np.empty((len(keys), nb)); emp = ks >= e
        LL[emp] = pred.pop_ll[nxt[kh[emp]]]
        if (~emp).any():
            ne = np.flatnonzero(~emp); q = pred.q(O, np.stack([kh[ne], ks[ne], np.full(len(ne), e)], 1)); n_enc += len(ne)
            LL[ne] = np.take_along_axis(pred.type_ll(q), nxt[kh[ne]], 1)
        pos = lambda key: np.minimum(np.searchsorted(keys, key), len(keys) - 1)
        cur = LL[pos(hh * KEY + k)]
        alt = np.where((WA == 0)[:, None, None], pred.pop_ll[nxt][None], LL[pos(np.arange(n)[None] * KEY + alt_s[:, None])])
        for j in range(nb):
            t = e + j; act = ~hold
            inc = np.clip(alt[..., j] - cur[..., j], -CLIP, CLIP) - C[:, None]
            S = np.where(act, np.maximum(0.0, S + inc), S)
            z = np.where(act & (S <= 0.0), t + 1, z)
            fire = act & (S > TAU[:, None])
            if fire.any():
                ci, hi = np.nonzero(fire)
                alarms += [(int(a), int(b), t, int(z[a, b])) for a, b in zip(ci, hi)]
                k[fire] = z[fire]; hold |= fire
        if (e // BLOCK) % 20 == 19:
            log(f"  detector[{tag}] hand {e + nb}/{H_LEN}: {len(keys)} segments this block, {n_enc} encodings, {len(alarms)} alarms ({time.time() - t0:.0f}s)")
    return np.array(alarms, dtype=np.int64).reshape(-1, 4)


def known_cusum(pred, O, QA, QB):
    """CUSUM on the exact per-hand LLR log p(h|q_B)/p(h|q_A), threshold log 1000; restarts after pre-switch alarms,
    stops after the first alarm after hand 200.  O (n_hist, 500) with history h from process h // 2."""
    llA, llB = pred.type_ll(QA), pred.type_ll(QB); p = np.arange(len(O)) // 2
    inc = np.clip(np.take_along_axis(llB[p], O.astype(np.int64), 1) - np.take_along_axis(llA[p], O.astype(np.int64), 1), -CLIP, CLIP)
    n = len(O); S = np.zeros(n); z = np.zeros(n, np.int64); act = np.ones(n, bool); alarms = []
    for t in range(H_LEN):
        S = np.where(act, np.maximum(0.0, S + inc[:, t]), S); z = np.where(act & (S <= 0), t + 1, z)
        fire = act & (S > KNOWN_H)
        for h in np.flatnonzero(fire):
            alarms.append((0, int(h), t, int(z[h])))
        S[fire] = 0.0; z[fire] = t + 1; act &= ~(fire & (t >= SWITCH_AT))
    return np.array(alarms, dtype=np.int64).reshape(-1, 4), inc


def seg_starts(alarms, cfg, n, T_list):
    """Segment start at each checkpoint T (hands 0..T-1 seen) from the latest alarm with t <= T-1 -> (n, nT)."""
    out = np.zeros((n, len(T_list)), np.int64); a = alarms[alarms[:, 0] == cfg]
    for h in np.unique(a[:, 1]):
        ah = a[a[:, 1] == h]; ah = ah[np.argsort(ah[:, 2], kind="stable")]
        for j, T in enumerate(T_list):
            m = ah[:, 2] <= T - 1
            if m.any():
                out[h, j] = ah[m][-1, 3]
    return out


# ----------------------------------------------------------------------------------------------- deployment
def ghat(pred, O, reqs):
    """reqs (m, 4) rows (h, T, kind, param): kind 0 = PRIOR-EM on hands [param, T); kind 1 = DISC-PRIOR-EM with gamma = param / 1000."""
    from .game.leduc_tree import get_tree
    from .game.sequence_form import get_sequence_form
    from .game.policy_utils import RankPolicyToG
    from .baselines.likelihood import type_counts
    from .t5_hybrids import fit_em_with_row_prior
    r2g = RankPolicyToG(get_tree(), get_sequence_form(), pred.sym); mask = pred.sym.rank_legal_mask[1]; T_ = pred.tab.n_types
    out = np.zeros((len(reqs), r2g.g(mask[None] / mask.sum(1, keepdims=True)[None]).shape[1]))
    for c0 in range(0, len(reqs), 512):
        R = reqs[c0:c0 + 512]; seg = np.zeros((len(R), 3), np.int64); counts = np.zeros((len(R), T_))
        for i, (h, T, kind, prm) in enumerate(R):
            if kind == 0:
                seg[i] = (h, prm, T); counts[i] = np.bincount(O[h, prm:T].astype(np.int64), minlength=T_)
            else:
                g = prm / 1000.0; Lw = DISC[g]; seg[i] = (h, max(0, T - Lw), T)
                counts[i] = np.bincount(O[h, :T].astype(np.int64), weights=g ** (T - 1 - np.arange(T)), minlength=T_)
        q = pred.q(O, seg)
        out[c0:c0 + len(R)] = r2g.g(fit_em_with_row_prior(pred.lik, mask, q, KAPPA, counts))
        if (c0 // 512) % 10 == 9:
            log(f"  ghat {c0 + len(R)}/{len(reqs)}")
    return out


_LP = {}


def _lp_init(audit):
    from .game.safe_lp import get_solver, OpenSpielAuditor
    from .game.sequence_form import get_sequence_form
    _LP["L"] = get_solver(); _LP["aud"] = OpenSpielAuditor(get_sequence_form(), _LP["L"].v_star) if audit else None


def _lp_chunk(args):
    gh, Gc = args; L, aud = _LP["L"], _LP["aud"]; u = np.zeros(len(gh)); ex = np.full(len(gh), np.nan); ok = np.zeros(len(gh), bool)
    for i in range(len(gh)):
        okk, pol, x = L.solve_safe(np.asarray(gh[i], dtype=np.float64), EPS); ok[i] = okk; u[i] = x @ Gc[i]
        if aud is not None:
            ex[i] = aud.exploitability_of_learner(pol)
    return u, ex, ok


def solve_lps(G_hat, G_cur, audit, workers=4):
    chunks = [(G_hat[i:i + 100], G_cur[i:i + 100]) for i in range(0, len(G_hat), 100)]; t0 = time.time()
    with mp.get_context("spawn").Pool(workers, initializer=_lp_init, initargs=(audit,)) as pool:
        res = []
        for j, r in enumerate(pool.imap(_lp_chunk, chunks)):
            res.append(r)
            if j % 40 == 39:
                log(f"  LPs {100 * (j + 1)}/{len(G_hat)} ({time.time() - t0:.0f}s)")
    return tuple(np.concatenate([r[i] for r in res]) for i in range(3))


class Requests:
    """Collect deployment requests (set, h, T-index, kind, param) for named methods; deduplicate; evaluate."""
    def __init__(self):
        self.rows = {}; self.methods = {}

    def add(self, method, s, h, j, T, kind, prm):
        key = (s, int(h), int(T), int(kind), int(prm)); idx = self.rows.setdefault(key, len(self.rows))
        self.methods.setdefault((method, s), []).append((int(h), j, idx))

    def evaluate(self, pred, P, Os, audit):
        keys = list(self.rows); by_set = {}
        for i, (s, h, T, kind, prm) in enumerate(keys):
            by_set.setdefault(s, []).append((i, h, T, kind, prm))
        G_hat = np.zeros((len(keys), 1093)); G_cur = np.zeros_like(G_hat)
        for s, lst in by_set.items():
            arr = np.array(lst, dtype=np.int64); log(f"  {s}: {len(arr)} unique estimates")
            G_hat[arr[:, 0]] = ghat(pred, Os[s], arr[:, 1:])
            Gt, _, _ = cur_values(P, s, CKPTS[s]); tix = {T: j for j, T in enumerate(CKPTS[s])}
            G_cur[arr[:, 0]] = Gt[arr[:, 1] // 2, [tix[T] for T in arr[:, 2]]]
        u, ex, ok = solve_lps(G_hat, G_cur, audit)
        res = {}
        for (m, s), lst in self.methods.items():
            n = len(Os[s]); nT = len(CKPTS[s]); U = np.full((n, nT), np.nan); E = np.full((n, nT), np.nan); K = np.zeros((n, nT), bool)
            for h, j, idx in lst:
                U[h, j], E[h, j], K[h, j] = u[idx], ex[idx], ok[idx]
            res[(m, s)] = (U, E, K)
        return res, G_hat, keys, u, ex, ok


# ----------------------------------------------------------------------------------------------- stages
def calibrate(pred, P):
    Osw = P["CSW_obs"].reshape(-1, H_LEN); Ost = P["CST_obs"].reshape(-1, H_LEN); O = np.concatenate([Osw, Ost]); nsw = len(Osw)
    t0 = time.time(); alarms = run_detector(pred, O, GRID, "CAL"); t_det = time.time() - t0
    Os = {"CSW": Osw, "CST": Ost}; R = Requests()
    for ci, cfg in enumerate(GRID):
        for s, off in (("CSW", 0), ("CST", nsw)):
            n = len(Os[s]); a = alarms[(alarms[:, 1] >= off) & (alarms[:, 1] < off + n)].copy(); a[:, 1] -= off
            st = seg_starts(a, ci, n, CKPTS[s])
            for h in range(n):
                for j, T in enumerate(CKPTS[s]):
                    R.add(cfg_name(cfg), s, h, j, T, 0, st[h, j])
    for s in ("CSW", "CST"):
        for h in range(len(Os[s])):
            for j, T in enumerate(CKPTS[s]):
                R.add("PRIOR-EM", s, h, j, T, 0, 0)
    t0 = time.time(); res, _, keys, u, _, ok = R.evaluate(pred, P, Os, audit=False); t_lp = time.time() - t0
    F = {}
    for m in [cfg_name(c) for c in GRID] + ["PRIOR-EM"]:
        f = {}
        for s in ("CSW", "CST"):
            U = res[(m, s)][0]; U = U.reshape(-1, 2, U.shape[1]).mean(1); _, V0, Ve = cur_values(P, s, CKPTS[s])
            keep = (Ve - V0) >= 0.01                                                   # headroom exclusion as in the non-stationary study
            f[s] = float(np.mean(np.where(keep, U - V0, 0).sum(0) / np.where(keep, Ve - V0, 0).sum(0)))   # mean over checkpoints of the pooled fraction
        F[m] = f
    cost = {m: F["PRIOR-EM"]["CST"] - F[m]["CST"] for m in F}
    ok_cfg = [c for c in GRID if cost[cfg_name(c)] <= 0.02]
    sel = max(ok_cfg, key=lambda c: F[cfg_name(c)]["CSW"]) if ok_cfg else min(GRID, key=lambda c: cost[cfg_name(c)])
    n_fa = {cfg_name(c): int(((alarms[:, 0] == ci) & (alarms[:, 1] >= nsw)).sum()) for ci, c in enumerate(GRID)}
    out = {"F": F, "stationary_cost": cost, "qualifying": [cfg_name(c) for c in ok_cfg], "selected": list(sel), "selected_name": cfg_name(sel),
           "cal_stat_alarms": n_fa, "n_unique_estimates": len(keys), "lp_failures": int((~ok).sum()), "detector_s": t_det, "lp_s": t_lp}
    np.save(D / "calib_alarms.npy", alarms); save_json(out, D / "calib.json")
    log(f"calibration: selected {cfg_name(sel)}  F_switch {F[cfg_name(sel)]['CSW']:.4f}  stationary cost {cost[cfg_name(sel)]:+.4f}  ({len(ok_cfg)} qualify)")
    return sel


def test(pred, P, sel):
    Osw = P["SW_obs"].reshape(-1, H_LEN); Ost = P["ST_obs"].reshape(-1, H_LEN); Odr = P["DR_obs"].reshape(-1, H_LEN); nsw = len(Osw)
    Os = {"SW": Osw, "ST": Ost, "DR": Odr}; t0 = time.time()
    alarms = run_detector(pred, np.concatenate([Osw, Ost]), GRID, "TEST"); si = GRID.index(tuple(sel))
    alarms_dr = run_detector(pred, Odr, [tuple(sel)], "DRIFT")
    alarms_known, llr = known_cusum(pred, Osw, P["SW_QA"], P["SW_QB"]); t_det = time.time() - t0
    np.save(D / "test_alarms.npy", alarms); np.save(D / "test_alarms_drift.npy", alarms_dr); np.save(D / "test_alarms_known.npy", alarms_known)
    R = Requests()
    def add_fixed(s):
        for h in range(len(Os[s])):
            for j, T in enumerate(CKPTS[s]):
                R.add("PRIOR-EM", s, h, j, T, 0, 0); R.add("PRIOR-EM-WIN", s, h, j, T, 0, max(0, T - 50))
                for g in DISC:
                    R.add(f"DISC-PRIOR-EM g={g}", s, h, j, T, 1, int(round(g * 1000)))
                if s == "SW":
                    R.add("POST-PRIOR-EM", s, h, j, T, 0, SWITCH_AT if T > SWITCH_AT else 0)
    for s in ("SW", "ST", "DR"):
        add_fixed(s)
    for ci, cfg in enumerate(GRID):
        for s, off in (("SW", 0), ("ST", nsw)):
            n = len(Os[s]); a = alarms[(alarms[:, 1] >= off) & (alarms[:, 1] < off + n)].copy(); a[:, 1] -= off
            st = seg_starts(a, ci, n, CKPTS[s])
            for h in range(n):
                for j, T in enumerate(CKPTS[s]):
                    R.add(cfg_name(cfg), s, h, j, T, 0, st[h, j])
                    if ci == si:
                        R.add("CPD-PRIOR-EM", s, h, j, T, 0, st[h, j])
    st = seg_starts(alarms_dr, 0, len(Odr), CKPTS["DR"])
    for h in range(len(Odr)):
        for j, T in enumerate(CKPTS["DR"]):
            R.add("CPD-PRIOR-EM", "DR", h, j, T, 0, st[h, j])
    st = seg_starts(alarms_known, 0, nsw, CKPTS["SW"])
    for h in range(nsw):
        for j, T in enumerate(CKPTS["SW"]):
            R.add("KNOWN-CUSUM-RESET", "SW", h, j, T, 0, st[h, j])
    t1 = time.time(); res, G_hat, keys, u, ex, ok = R.evaluate(pred, P, Os, audit=True); t_lp = time.time() - t1
    arrs = {}
    for (m, s), (U, E, K) in res.items():
        arrs[f"{s}::{m}::u"] = U; arrs[f"{s}::{m}::expl"] = E; arrs[f"{s}::{m}::ok"] = K
    # verification against the non-stationary study's g_hat at the original checkpoints
    ver = {}
    for m, f in (("PRIOR-EM", "PRIOR-EM"), ("PRIOR-EM-WIN", "PRIOR-EM-WIN"), ("POST-PRIOR-EM", "POST-PRIOR-EM")):
        old = np.load(NS / f"ghat_SWITCH_{f}.npy"); d = 0.0
        for jo, T in enumerate(SW_T):
            for h in range(nsw):
                s0 = {"PRIOR-EM": 0, "PRIOR-EM-WIN": max(0, T - 50), "POST-PRIOR-EM": SWITCH_AT if T > SWITCH_AT else 0}[m]
                d = max(d, float(np.abs(G_hat[R.rows[("SW", h, T, 0, s0)]] - old[h, jo]).max()))
        ver[m] = d
    np.savez_compressed(D / "test.npz", **arrs, keys=np.array(keys, dtype=object), u_all=u, expl_all=ex, ok_all=ok, llr_known=llr.astype(np.float32))
    meta = {"selected": list(sel), "selected_name": cfg_name(sel), "n_unique_estimates": len(keys), "detector_s": t_det, "lp_audit_s": t_lp,
            "verification_max_abs_ghat_diff": ver, "methods": sorted({m for (m, s) in res}), "checkpoints": CKPTS}
    save_json(meta, D / "test_meta.json"); log(f"test done: {len(keys)} unique deployed strategies; verification {ver}")


def type_dist(tree, tab, pol0, pol1):
    from .data.simulator import node_probability_table
    from .game.leduc_tree import TERMINAL
    P = node_probability_table(tree, pol0, pol1); reach = np.zeros(tree.n_nodes); reach[0] = 1.0
    par, slot = np.nonzero(tree.child_table >= 0); ch = tree.child_table[par, slot]; order = np.argsort(ch)   # children have larger ids than parents
    for p_, s_, c_ in zip(par[order], slot[order], ch[order]):
        reach[c_] = reach[p_] * P[p_, s_]
    term = np.flatnonzero(tree.node_type == TERMINAL)
    return np.bincount(tab.obs_type_of_terminal[term], weights=reach[term], minlength=tab.n_types)


def floor(pred, P):
    from .game.leduc_tree import get_tree
    from .data.datasets import load_population
    tree = get_tree(); pop = load_population(); assert np.all(tree.parent[1:] < np.arange(1, tree.n_nodes))
    out = {"KL_BA": [], "KL_AB": [], "check_max_abs": 0.0, "mc_llr_post": []}
    llr = np.load(D / "test.npz", allow_pickle=True)["llr_known"].reshape(-1, 2, H_LEN)
    for i in range(len(P["SW_QA"])):
        pA = type_dist(tree, pred.tab, pop["blueprint0"], pred.sym.expand(1, P["SW_QA"][i])); pB = type_dist(tree, pred.tab, pop["blueprint0"], pred.sym.expand(1, P["SW_QB"][i]))
        assert abs(pA.sum() - 1) < 1e-9 and abs(pB.sum() - 1) < 1e-9
        lA, lB = pred.type_ll(P["SW_QA"][i][None])[0], pred.type_ll(P["SW_QB"][i][None])[0]; nz = (pA > 1e-15) & (pB > 1e-15)
        out["check_max_abs"] = max(out["check_max_abs"], float(np.abs((np.log(pB[nz]) - np.log(pA[nz])) - (lB[nz] - lA[nz])).max()))
        out["KL_BA"].append(float((pB[nz] * (np.log(pB[nz]) - np.log(pA[nz]))).sum())); out["KL_AB"].append(float((pA[nz] * (np.log(pA[nz]) - np.log(pB[nz]))).sum()))
        out["mc_llr_post"].append(float(llr[i][:, SWITCH_AT:].mean()))
    out = {k: (np.array(v) if isinstance(v, list) else v) for k, v in out.items()}
    np.savez_compressed(D / "floor.npz", **out); log(f"floor: median KL(B||A) {np.median(out['KL_BA']):.3f} nats/hand; exact-vs-likelihood check {out['check_max_abs']:.2e}")


if __name__ == "__main__":
    D.mkdir(parents=True, exist_ok=True); stages = sys.argv[1:] or ["build", "calibrate", "test", "floor"]; wall = {}
    wall_f = D / "wallclock.json"; wall = json.loads(wall_f.read_text()) if wall_f.exists() else {}
    if "build" in stages and not (D / "processes.npz").exists():
        t = time.time(); build(); wall["build_s"] = time.time() - t; save_json(wall, wall_f)
    P = np.load(D / "processes.npz", allow_pickle=True); pred = Pred()
    if "calibrate" in stages and not (D / "calib.json").exists():
        t = time.time(); calibrate(pred, P); wall["calibrate_s"] = time.time() - t; save_json(wall, wall_f)
    sel = tuple(json.loads((D / "calib.json").read_text())["selected"]) if (D / "calib.json").exists() else None
    if "test" in stages and sel is not None and not (D / "test.npz").exists():
        t = time.time(); test(pred, P, sel); wall["test_s"] = time.time() - t; save_json(wall, wall_f)
    if "floor" in stages and (D / "test.npz").exists() and not (D / "floor.npz").exists():
        t = time.time(); floor(pred, P); wall["floor_s"] = time.time() - t; save_json(wall, wall_f)
    log(f"done: {wall}")
