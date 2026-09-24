"""Safe-set geometry diagnostic (no training).  Stages:
  build   -> S_eps support points, M_support, M_resp, M_rand, M_gvar, inflation factor s   (outputs/geometry/M.npz, build.json)
  score   -> per (method, N, opponent): R, ||d||_2, ||d||_M (4 M's), exact W(d) via two LPs   (outputs/geometry/points.npz)
  analyze -> Spearman / Kendall tables, paradox pairs, bound checks                        (outputs/geometry/analysis.json)
  stretch -> realizability projection of DEC-889k g_hat; oracle safe gain at small eps       (outputs/geometry/stretch.json)
Every LP solution is audited with OpenSpiel's best response (Expl <= eps + 1e-7)."""
from __future__ import annotations
import os, sys, json, time
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import multiprocessing as mp
from pathlib import Path
import numpy as np

from .common import OUT, N_BUDGETS, save_json, load_json

EPS = 0.10; EPS_IDX = 2
GD = OUT / "geometry"; GD.mkdir(parents=True, exist_ok=True)
EVAL = OUT / "eval" / "test"
METHODS = {"DEC-133k": "NEURAL_DEC133K_s0", "DEC-889k": "NEURAL_DECISION_s0", "RECON-131k": "NEURAL_RECON_s0",
           "RECON-889k": "NEURAL_REC889K_s0", "RECON-JAC-889k": "NEURAL_RECJAC889K_s0", "MSE+0.3SPO+": "NEURAL_SPO03_133K_s0",
           "tabular EM": "TABULAR_EM_UNIFORM", "bank posterior": "BANK_POSTERIOR", "learned-prior EM": "HYB_PRIOR_EM_S0"}
N_SEL = [5, 20, 100, 500]


# ------------------------------------------------------------------ worker: support LPs + audit
_W = {}
def _init():
    from .game.safe_lp import get_solver, OpenSpielAuditor
    from .game.sequence_form import get_sequence_form
    S = get_sequence_form(); L = get_solver()
    _W.update(S=S, L=L, aud=OpenSpielAuditor(S, L.v_star))


def _support(u, eps=EPS):
    """raw argmax_{S_eps} u^T x (LP solution) + OpenSpiel audit of the corresponding behavioral policy."""
    S, L, aud = _W["S"], _W["L"], _W["aud"]
    ok, x, _ = L.lp0.solve_safe(u, eps, L.v_star)
    pol = S.realization_to_behavioral(0, x)
    e_os = aud.exploitability_of_learner(pol)
    return ok, x, e_os


def _support_batch(args):
    U, eps = args
    out = [_support(u, eps) for u in U]
    return np.array([o[0] for o in out]), np.array([o[1] for o in out]), np.array([o[2] for o in out])


def _width_batch(args):
    """W(d) = max d^T x - min d^T x over S_eps; returns W, the two support values, audit and LP status."""
    D, eps = args
    W = np.zeros(len(D)); hi = np.zeros(len(D)); lo = np.zeros(len(D)); ex = np.zeros((len(D), 2)); ok = np.ones(len(D), bool)
    for i, d in enumerate(D):
        o1, x1, e1 = _support(d, eps); o2, x2, e2 = _support(-d, eps)
        hi[i] = d @ x1; lo[i] = d @ x2; W[i] = hi[i] - lo[i]; ex[i] = (e1, e2); ok[i] = o1 and o2
    return W, hi, lo, ex, ok


def _pmap(fn, items, workers=4):
    with mp.get_context("spawn").Pool(workers, initializer=_init) as pool:
        return pool.map(fn, items)


def _split(A, k):
    return [A[i::k] for i in range(k)], [np.arange(len(A))[i::k] for i in range(k)]


def _unsplit(parts, idx, n, tail_shape=()):
    out = np.zeros((n,) + tail_shape, dtype=parts[0].dtype)
    for p, ix in zip(parts, idx):
        out[ix] = p
    return out


# ------------------------------------------------------------------ stage 1: build the metrics
def build(workers=4, seed=0):
    from .data.datasets import load_population
    t0 = time.time(); rng = np.random.default_rng(seed)
    pop = load_population(); G = pop["G"]; tr = np.flatnonzero(pop["split"] == 0); n0 = G.shape[1]
    Gtr = G[tr]; gbar = Gtr.mean(0)
    U_, sv, Vt = np.linalg.svd(Gtr - gbar, full_matrices=False)
    lam = sv ** 2 / (len(tr) - 1); rank_g = int((sv > 1e-8 * sv[0]).sum())
    gauss = rng.normal(size=(250, n0)); gauss /= np.linalg.norm(gauss, axis=1, keepdims=True)
    pca = Vt[:250]
    D = np.concatenate([gauss, -gauss, pca, -pca])                      # 1000 directions
    kinds = np.array(["gauss"] * 500 + ["pca"] * 500)
    # fresh set for the inflation factor: 100 new Gaussian + 100 PCA-weighted combinations
    fg = rng.normal(size=(100, n0)); fg /= np.linalg.norm(fg, axis=1, keepdims=True)
    fp = (rng.normal(size=(100, rank_g)) * np.sqrt(lam[:rank_g])) @ Vt[:rank_g]; fp /= np.linalg.norm(fp, axis=1, keepdims=True)
    F = np.concatenate([fg, fp])
    allD = np.concatenate([D, F])
    parts, idx = _split(allD, workers)
    res = _pmap(_support_batch, [(p, EPS) for p in parts], workers)
    ok = _unsplit([r[0] for r in res], idx, len(allD)); X = _unsplit([r[1] for r in res], idx, len(allD), (n0,)); eos = _unsplit([r[2] for r in res], idx, len(allD))
    Xb, Xf = X[:1000], X[1000:]
    M_support = np.cov(Xb, rowvar=False)
    xs = np.load(OUT / "weights_v3" / "xstar_train_eps0.1.npy")[tr]
    M_resp = np.cov(xs, rowvar=False)
    tr_s = np.trace(M_support)
    M_gvar = np.cov(Gtr, rowvar=False); M_gvar *= tr_s / np.trace(M_gvar)
    ev_s, Q_s = np.linalg.eigh(M_support)
    Qr, _ = np.linalg.qr(rng.normal(size=(n0, n0)))
    M_rand = (Qr * ev_s) @ Qr.T
    M_resp_scaled_trace = float(np.trace(M_resp))
    # inflation factor: fresh points inside {c + (sM)^{1/2} v}; pseudo-inverse on range(M_support)
    tol = 1e-10 * ev_s.max(); keep = ev_s > tol; rank_s = int(keep.sum())
    Minv = (Q_s[:, keep] / ev_s[keep]) @ Q_s[:, keep].T; P = Q_s[:, keep] @ Q_s[:, keep].T
    def inflation(c):
        d = Xf - c
        q = np.einsum("ij,jk,ik->i", d, Minv, d)
        resid = np.linalg.norm(d - d @ P, axis=1) / np.maximum(np.linalg.norm(d, axis=1), 1e-12)
        return float(q.max()), float(np.median(q)), float(resid.max())
    s_fresh, s_med_fresh, resid_fresh = inflation(Xf.mean(0))
    s_build, s_med_build, resid_build = inflation(Xb.mean(0))
    q_build_self = np.einsum("ij,jk,ik->i", Xb - Xb.mean(0), Minv, Xb - Xb.mean(0))
    def spectrum(M, name):
        ev = np.clip(np.linalg.eigvalsh(M)[::-1], 0, None); r = int((ev > 1e-10 * ev[0]).sum())
        return {"name": name, "trace": float(ev.sum()), "rank_tol1e-10": r, "top10": ev[:10].tolist(),
                "condition_number_nonzero": float(ev[0] / ev[r - 1]), "participation_ratio": float(ev.sum() ** 2 / (ev ** 2).sum()),
                "frac_trace_top10": float(ev[:10].sum() / ev.sum()), "frac_trace_top50": float(ev[:50].sum() / ev.sum()),
                "n_eig_for_90pct_trace": int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.9) + 1), "eig_at_rank": float(ev[r - 1])}
    spec = {k: spectrum(M, k) for k, M in [("M_support", M_support), ("M_resp", M_resp), ("M_gvar", M_gvar), ("M_rand", M_rand)]}
    # alignment between the S_eps shape and the g-variance shape (how much of M_support's trace lies in the training-g subspace)
    Pg = Vt[:rank_g].T @ Vt[:rank_g]
    align = {"frac_trace_M_support_in_g_span": float(np.trace(Pg @ M_support) / tr_s), "frac_trace_M_resp_in_g_span": float(np.trace(Pg @ M_resp) / np.trace(M_resp)),
             "g_span_dim": rank_g, "affine_hull_dim_realization_plans": int(n0 - 469)}
    info = {"n_directions": 1000, "n_fresh": 200, "lp_fail_build": int((~ok[:1000]).sum()), "lp_fail_fresh": int((~ok[1000:]).sum()),
            "audit_max_expl_minus_eps": float(np.max(eos - EPS)), "audit_n_violations_1e-7": int((eos - EPS > 1e-7).sum()), "n_audited": int(len(eos)),
            "M_support_rank": rank_s, "inflation_s_fresh_centre": s_fresh, "inflation_median_fresh_centre": s_med_fresh, "resid_outside_range_fresh": resid_fresh,
            "inflation_s_build_centre": s_build, "inflation_median_build_centre": s_med_build, "resid_outside_range_build_centre": resid_build,
            "build_points_self_max_q": float(q_build_self.max()), "build_points_self_median_q": float(np.median(q_build_self)),
            "trace_M_support": float(tr_s), "trace_M_resp": M_resp_scaled_trace, "spectra": spec, "alignment": align,
            "support_value_spread": {"gauss_dir_mean_norm_x": float(np.linalg.norm(Xb[kinds == 'gauss'] - Xb.mean(0), axis=1).mean()),
                                     "pca_dir_mean_norm_x": float(np.linalg.norm(Xb[kinds == 'pca'] - Xb.mean(0), axis=1).mean())},
            "runtime_s": time.time() - t0}
    np.savez_compressed(GD / "M.npz", M_support=M_support, M_resp=M_resp, M_rand=M_rand, M_gvar=M_gvar, X_build=Xb, X_fresh=Xf, D=D, F=F,
                        kinds=kinds, s_fresh=s_fresh, s_build=s_build, gbar=gbar, Vt_g=Vt[:rank_g], lam_g=lam[:rank_g])
    save_json(info, GD / "build.json")
    print(json.dumps({k: v for k, v in info.items() if k != "spectra"}, indent=1)); print(json.dumps(spec, indent=1)[:3000])


# ------------------------------------------------------------------ stage 2: score existing predictions
def score(workers=4):
    from .data.datasets import load_population
    t0 = time.time(); pop = load_population()
    Mz = np.load(GD / "M.npz"); Ms = {k: Mz[k] for k in ["M_support", "M_resp", "M_rand", "M_gvar"]}
    meta = load_json(EVAL / "predict_meta.json"); ns = meta["n_streams"]
    hist_opp = np.load(EVAL / "hist_opp.npy"); rows = np.arange(0, len(hist_opp), ns)            # first stream of every opponent
    opp = hist_opp[rows]; Gt = pop["G"][opp].astype(np.float64); V = pop["V_oracle"][opp, EPS_IDX]
    jsel = [N_BUDGETS.index(N) for N in N_SEL]
    rec = {k: [] for k in ["method", "N", "opp", "R", "l2", "M_support", "M_resp", "M_rand", "M_gvar"]}; deltas = []
    for mname, key in METHODS.items():
        gh = np.load(EVAL / f"ghat_{key}.npy", mmap_mode="r")
        u = np.load(EVAL / f"solve_{key}.npz")["u"]
        for j, N in zip(jsel, N_SEL):
            g_hat = np.asarray(gh[rows, j], dtype=np.float64)
            d = Gt - g_hat                                                                          # raw chips
            R = V - u[rows, j, EPS_IDX]
            rec["method"] += [mname] * len(opp); rec["N"] += [N] * len(opp); rec["opp"] += opp.tolist(); rec["R"] += R.tolist()
            rec["l2"] += np.linalg.norm(d, axis=1).tolist()
            for k, M in Ms.items():
                rec[k] += np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, M, d), 0)).tolist()
            deltas.append(d)
    D = np.concatenate(deltas); n = len(D)
    print(f"{n} points; solving {2 * n} width LPs on {workers} workers", flush=True)
    parts, idx = _split(D, workers * 6)
    res = _pmap(_width_batch, [(p, EPS) for p in parts], workers)
    W = _unsplit([r[0] for r in res], idx, n); hi = _unsplit([r[1] for r in res], idx, n); lo = _unsplit([r[2] for r in res], idx, n)
    ex = _unsplit([r[3] for r in res], idx, n, (2,)); ok = _unsplit([r[4] for r in res], idx, n)
    out = {k: np.array(v) for k, v in rec.items()}
    out.update(W=W, W_hi=hi, W_lo=lo, audit_expl=ex, lp_ok=ok)
    np.savez_compressed(GD / "points.npz", **out)
    viol = out["R"] - W
    summ = {"n_points": n, "n_width_lps": 2 * n, "lp_failures": int((~ok).sum()), "audit_max_expl_minus_eps": float(ex.max() - EPS),
            "audit_n_violations_1e-7": int((ex - EPS > 1e-7).sum()), "R_minus_W_max": float(viol.max()), "n_R_gt_W_1e-6": int((viol > 1e-6).sum()),
            "R_min": float(out["R"].min()), "runtime_s": time.time() - t0}
    save_json(summ, GD / "score.json"); print(json.dumps(summ, indent=1))


# ------------------------------------------------------------------ stage 3: analysis
METRICS = ["l2", "M_support", "M_resp", "M_rand", "M_gvar", "W"]


def analyze():
    from scipy.stats import spearmanr, kendalltau
    P = dict(np.load(GD / "points.npz")); B = load_json(GD / "build.json")
    meth = P["method"]; N = P["N"]; R = P["R"]; n = len(R)
    res = {"n_points": int(n)}
    sp = lambda a, b: float(spearmanr(a, b)[0])
    # A. Spearman pooled, per method, per N, per cell
    res["spearman_pooled"] = {m: sp(R, P[m]) for m in METRICS}
    res["spearman_per_N"] = {int(v): {m: sp(R[N == v], P[m][N == v]) for m in METRICS} for v in N_SEL}
    res["spearman_per_method"] = {mm: {m: sp(R[meth == mm], P[m][meth == mm]) for m in METRICS} for mm in METHODS}
    cells = {}
    for mm in METHODS:
        for v in N_SEL:
            sel = (meth == mm) & (N == v)
            cells[f"{mm}|{v}"] = {m: sp(R[sel], P[m][sel]) for m in METRICS}
    res["spearman_cells"] = cells
    res["spearman_within_cell_mean"] = {m: float(np.mean([c[m] for c in cells.values()])) for m in METRICS}
    res["spearman_within_cell_frac_beats_l2"] = {m: float(np.mean([c[m] > c["l2"] for c in cells.values()])) for m in METRICS}
    # per-N-standardized pooled (removes the N trend): rank within N then pool
    from scipy.stats import rankdata
    def within_N_ranks(a):
        out = np.zeros_like(a, dtype=float)
        for v in N_SEL:
            sel = N == v; out[sel] = rankdata(a[sel]) / sel.sum()
        return out
    rR = within_N_ranks(R)
    res["spearman_pooled_withinN_ranks"] = {m: float(np.corrcoef(rR, within_N_ranks(P[m]))[0, 1]) for m in METRICS}
    # B. method level
    means = {mm: {int(v): {**{m: float(P[m][(meth == mm) & (N == v)].mean()) for m in METRICS}, "R": float(R[(meth == mm) & (N == v)].mean())}
                  for v in N_SEL} for mm in METHODS}
    res["method_means"] = means
    res["kendall_method_level"] = {int(v): {m: float(kendalltau([means[mm][v]["R"] for mm in METHODS], [means[mm][v][m] for mm in METHODS])[0]) for m in METRICS} for v in N_SEL}
    def pair(a, b):
        out = {}
        for v in N_SEL:
            ra, rb = means[a][v], means[b][v]
            out[int(v)] = {"R": [ra["R"], rb["R"]], **{m: [ra[m], rb[m]] for m in METRICS},
                           "same_order_as_R": {m: bool(np.sign(ra[m] - rb[m]) == np.sign(ra["R"] - rb["R"])) for m in METRICS}}
        return out
    res["paradox_pairs"] = {"DEC-133k vs RECON-889k": pair("DEC-133k", "RECON-889k"), "DEC-133k vs RECON-131k": pair("DEC-133k", "RECON-131k"),
                            "DEC-889k vs RECON-889k": pair("DEC-889k", "RECON-889k"), "RECON-JAC-889k vs DEC-889k": pair("RECON-JAC-889k", "DEC-889k")}
    # paired bootstrap of the per-opponent metric difference for the paradox pairs (sign stability)
    rng = np.random.default_rng(0)
    def boot_pair(a, b, v, m):
        xa = (R if m == "R" else P[m])[(meth == a) & (N == v)]; xb = (R if m == "R" else P[m])[(meth == b) & (N == v)]
        d = xa - xb; bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        return [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    res["paradox_pairs_bootstrap"] = {f"{a} - {b}": {int(v): {m: boot_pair(a, b, v, m) for m in ["R"] + METRICS} for v in N_SEL}
                                      for a, b in [("DEC-133k", "RECON-889k"), ("DEC-133k", "RECON-131k")]}
    # C. bounds
    s = float(B["inflation_s_fresh_centre"])
    ell = 2 * np.sqrt(s) * P["M_support"]
    res["bounds"] = {"R_minus_W_max": float((R - P["W"]).max()), "n_R_gt_W_1e-6": int((R - P["W"] > 1e-6).sum()),
                     "inflation_s": s, "ellipsoid_bound_violation_rate": float((R > ell + 1e-9).mean()),
                     "ellipsoid_uninflated_violation_rate": float((R > 2 * P["M_support"] + 1e-9).mean()),
                     "W_over_ellipsoid_bound_median": float(np.median(P["W"] / np.maximum(ell, 1e-12))),
                     "W_gt_ellipsoid_rate": float((P["W"] > ell + 1e-9).mean())}
    pos = R > 1e-9
    q = lambda a: {k: float(np.percentile(a, p)) for k, p in [("p10", 10), ("p25", 25), ("median", 50), ("p75", 75), ("p90", 90), ("max", 100)]}
    res["tightness"] = {"R_over_W": q(R[pos] / P["W"][pos]), "R_over_2Msupport_uninflated": q(R[pos] / (2 * P["M_support"][pos])),
                        "R_over_ellipsoid_inflated": q(R[pos] / ell[pos]), "W_over_2l2": q(P["W"] / (2 * P["l2"])),
                        "frac_R_zero": float((~pos).mean()),
                        "R_over_W_by_N": {int(v): float(np.median(R[pos & (N == v)] / P["W"][pos & (N == v)])) for v in N_SEL}}
    # audits
    res["audit"] = {"width_lps": int(2 * n), "max_expl_minus_eps": float(P["audit_expl"].max() - EPS), "n_violations": int((P["audit_expl"] - EPS > 1e-7).sum()),
                    "lp_failures": int((~P["lp_ok"]).sum())}
    # verdict inputs
    sp0 = res["spearman_pooled"]; best_S = max(sp0["M_support"], sp0["M_resp"], sp0["W"])
    pp = res["paradox_pairs"]["DEC-133k vs RECON-889k"]
    P2 = all(pp[v]["same_order_as_R"]["M_support"] and not pp[v]["same_order_as_R"]["l2"] for v in [100, 500])
    res["verdict_inputs"] = {"P1_diff": sp0["M_support"] - sp0["l2"], "P1": sp0["M_support"] - sp0["l2"] >= 0.15, "P2": bool(P2),
                             "P3": sp0["W"] >= sp0["M_support"], "P4": bool(sp0["M_rand"] <= sp0["l2"] and sp0["M_gvar"] <= sp0["l2"]),
                             "best_S_metric_minus_l2": best_S - sp0["l2"]}
    v = res["verdict_inputs"]
    res["verdict"] = ("NO-GO" if (v["best_S_metric_minus_l2"] < 0.05 and not v["P2"]) else
                      "GO" if (v["P1"] or (v["best_S_metric_minus_l2"] >= 0.05 and v["P2"])) else "WEAK/INCONCLUSIVE")
    save_json(res, GD / "analysis.json")
    print(json.dumps({k: res[k] for k in ["spearman_pooled", "spearman_within_cell_mean", "spearman_pooled_withinN_ranks", "verdict_inputs", "verdict", "bounds", "audit"]}, indent=1))


# ------------------------------------------------------------------ stretch
def _proj_init():
    _init()
    import scipy.sparse as sp, clarabel
    S = _W["S"]; A = sp.csc_matrix(S.A); F = sp.csc_matrix(S.F); n1 = A.shape[1]
    P = (2 * (A.T @ A)).tocsc()
    Acone = sp.vstack([F, -sp.identity(n1, format="csc")]).tocsc()
    b = np.concatenate([np.asarray(S.f, float), np.zeros(n1)])
    _W.update(A=A, P=P, Acone=Acone, b=b, cones=[clarabel.ZeroConeT(F.shape[0]), clarabel.NonnegativeConeT(n1)], clarabel=clarabel)


def _project(g_hat):
    """min_y ||A y - g_hat||^2 s.t. F y = f, y >= 0  (Euclidean projection onto the realizable set {A y})."""
    clarabel = _W["clarabel"]; A = _W["A"]
    q = -2 * (A.T @ g_hat)
    st = clarabel.DefaultSettings(); st.verbose = False; st.tol_gap_abs = 1e-10; st.tol_gap_rel = 1e-10; st.tol_feas = 1e-10
    sol = clarabel.DefaultSolver(_W["P"], q, _W["Acone"], _W["b"], _W["cones"], st).solve()
    y = np.asarray(sol.x); return str(sol.status), A @ y


def _proj_batch(args):
    G_hat, G_true, V = args
    out = []
    for gh, gt, v in zip(G_hat, G_true, V):
        status, gp = _project(gh)
        ok, x, e_os = _support(gp)
        L, S = _W["L"], _W["S"]
        pol = S.realization_to_behavioral(0, x); x_dep = S.behavioral_to_realization(0, pol)
        out.append((status, float(v - gt @ x_dep), float(e_os - EPS), float(np.linalg.norm(gp - gh)), float(np.linalg.norm(gp - gt)), float(np.linalg.norm(gh - gt))))
    return out


def _gain_batch(args):
    Gs, gbar, eps_list = args
    out = []
    for g in Gs:
        row = []
        for e in eps_list:
            ok1, x1, e1 = _support(g, e); ok2, x2, e2 = _support(gbar, e)
            row.append((float(g @ x1 - g @ x2), e1 - e, e2 - e, ok1 and ok2))
        out.append(row)
    return out


def stretch(workers=4):
    from .data.datasets import load_population
    t0 = time.time(); pop = load_population(); res = {}
    meta = load_json(EVAL / "predict_meta.json"); ns = meta["n_streams"]
    hist_opp = np.load(EVAL / "hist_opp.npy"); rows = np.arange(0, len(hist_opp), ns); opp = hist_opp[rows]
    Gt = pop["G"][opp].astype(np.float64); V = pop["V_oracle"][opp, EPS_IDX]
    # (1) realizability: project DEC-889k g_hat onto {A y}, re-solve, compare regret (first stream, N = 20 and 500)
    key = METHODS["DEC-889k"]; gh_all = np.load(EVAL / f"ghat_{key}.npy", mmap_mode="r"); u = np.load(EVAL / f"solve_{key}.npz")["u"]
    res["realizability"] = {}
    for N in [20, 500]:
        j = N_BUDGETS.index(N); gh = np.asarray(gh_all[rows, j], dtype=np.float64); R0 = V - u[rows, j, EPS_IDX]
        chunks = [(gh[i::workers], Gt[i::workers], V[i::workers]) for i in range(workers)]
        with mp.get_context("spawn").Pool(workers, initializer=_proj_init) as pool:
            parts = pool.map(_proj_batch, chunks)
        out = [None] * len(opp)
        for w, part in enumerate(parts):
            for k, o in enumerate(part):
                out[w + k * workers] = o
        R1 = np.array([o[1] for o in out]); ex = np.array([o[2] for o in out])
        d = R1 - R0; rng = np.random.default_rng(0); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        res["realizability"][N] = {"regret_original": float(R0.mean()), "regret_projected": float(R1.mean()), "diff_mean": float(d.mean()),
                                   "diff_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                   "frac_improved": float((d < -1e-9).mean()), "frac_worse": float((d > 1e-9).mean()),
                                   "proj_distance_mean": float(np.mean([o[3] for o in out])), "err_after_proj_mean": float(np.mean([o[4] for o in out])),
                                   "err_before_proj_mean": float(np.mean([o[5] for o in out])),
                                   "qp_status": sorted(set(o[0] for o in out)), "audit_max_expl_minus_eps": float(ex.max()), "audit_n_violations": int((ex > 1e-7).sum())}
        print("realizability", N, res["realizability"][N], flush=True)
    # (2) small eps: oracle safe gain over the mean-g response
    tr = np.flatnonzero(pop["split"] == 0); gbar = pop["G"][tr].mean(0)
    eps_list = [1e-4, 1e-3, 3e-3, 1e-2]
    chunks = [(Gt[i::workers], gbar, eps_list) for i in range(workers)]
    with mp.get_context("spawn").Pool(workers, initializer=_init) as pool:
        parts = pool.map(_gain_batch, chunks)
    rows_ = [None] * len(opp)
    for w, part in enumerate(parts):
        for k, o in enumerate(part):
            rows_[w + k * workers] = o
    gain = np.array([[r[0] for r in row] for row in rows_]); aud = np.array([[max(r[1], r[2]) for r in row] for row in rows_])
    t3 = load_json(OUT / "t3_eps_rank" / "t3_results.json")
    e_all = eps_list + [e for e in t3["eps_grid"] if e > 1e-2]
    g_all = gain.mean(0).tolist() + [g for e, g in zip(t3["eps_grid"], t3["component_split"]["gain_by_eps"]) if e > 1e-2]
    e_arr, g_arr = np.array(e_all), np.array(g_all)
    slopes = {f"{e_arr[i]:g}-{e_arr[i+1]:g}": float(np.log(g_arr[i + 1] / g_arr[i]) / np.log(e_arr[i + 1] / e_arr[i])) for i in range(len(e_arr) - 1)}
    res["small_eps"] = {"eps": e_all, "mean_gain": g_all, "local_loglog_slopes": slopes,
                        "gain_over_eps": (g_arr / e_arr).tolist(), "audit_max_expl_minus_eps": float(aud.max()), "audit_n_violations": int((aud > 1e-7).sum()),
                        "note": "eps <= 1e-2 computed here on the 300 test opponents; eps > 1e-2 taken from T3 (same opponents, same baseline)"}
    print("small eps", res["small_eps"], flush=True)
    res["runtime_s"] = time.time() - t0
    save_json(res, GD / "stretch.json")


if __name__ == "__main__":
    stage = sys.argv[1]
    {"build": build, "score": score, "analyze": analyze, "stretch": stretch}[stage]()
