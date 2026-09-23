"""T3: eps-rank oracle curve, Nash-set affine hull, within/orthogonal component split, Hoffman-type kappa.
Oracle only (true g); every deployed strategy audited with OpenSpiel."""
from __future__ import annotations

import json, time, sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .common import OUT, save_json
from .game.sequence_form import get_sequence_form
from .game.safe_lp import get_solver, OpenSpielAuditor
from .game.safe_qp import SafeQP
from .data.datasets import load_population

EPS_GRID = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40]
RANKS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
OUT_DIR = OUT / "t3_eps_rank"


def main():
    t0 = time.time(); OUT_DIR.mkdir(parents=True, exist_ok=True)
    S = get_sequence_form(); L = get_solver(); aud = OpenSpielAuditor(S, L.v_star); pop = load_population()
    G = pop["G"]; split = pop["split"]; fam = pop["family_index"]
    train = np.flatnonzero(split == 0); test = np.flatnonzero(split == 2)
    Gtr, Gte = G[train], G[test]
    mean_g = Gtr.mean(0)
    U, s, Vt = np.linalg.svd(Gtr - mean_g, full_matrices=False)       # PCA on training opponents
    evr = s ** 2 / (s ** 2).sum()
    n_audit = 0; max_viol = -np.inf; lp_fail = 0

    def safe(g, eps):
        nonlocal n_audit, max_viol, lp_fail
        ok, pol, x = L.solve_safe(g, eps); lp_fail += (not ok)
        e = aud.exploitability_of_learner(pol); n_audit += 1; max_viol = max(max_viol, e - eps)
        return x

    # 1. oracle values and x*(mean_g) per eps
    res = {"eps_grid": EPS_GRID, "ranks": RANKS, "pca_explained_variance": evr[:64].tolist(), "n_test": int(len(test))}
    x_mean = {eps: safe(mean_g, eps) for eps in EPS_GRID}
    V = np.zeros((len(test), len(EPS_GRID))); base = np.zeros_like(V)
    for i, k in enumerate(test):
        for j, eps in enumerate(EPS_GRID):
            V[i, j] = Gte[i] @ safe(Gte[i], eps); base[i, j] = Gte[i] @ x_mean[eps]
    gain = V - base
    print(f"oracle values done ({time.time()-t0:.0f}s); mean gain by eps:", np.round(gain.mean(0), 4), flush=True)
    # 2. rank-k retained value
    retained = np.full((len(test), len(EPS_GRID), len(RANKS)), np.nan)
    for r_i, k in enumerate(RANKS):
        P = Vt[:k]
        Gk = mean_g + (Gte - mean_g) @ P.T @ P
        for i in range(len(test)):
            for j, eps in enumerate(EPS_GRID):
                retained[i, j, r_i] = Gte[i] @ safe(Gk[i], eps) - base[i, j]
        print(f"rank {k} done ({time.time()-t0:.0f}s)", flush=True)
    valid = gain >= 0.02                                    # opponents with a non-trivial denominator
    frac = np.where(valid[:, :, None], retained / np.where(valid, gain, 1.0)[:, :, None], np.nan)
    mean_frac = np.nanmean(frac, 0)                          # (nE, nR)
    # pooled-value version (ratio of means) as well
    pooled = np.nansum(np.where(valid[:, :, None], retained, 0), 0) / np.where(valid.sum(0) > 0, np.where(valid, gain, 0).sum(0), np.nan)[:, None]
    k_needed = {}
    for j, eps in enumerate(EPS_GRID):
        k_needed[str(eps)] = {}
        for t in [0.5, 0.8, 0.9]:
            hit = [k for k, v in zip(RANKS, mean_frac[j]) if v >= t]
            hitp = [k for k, v in zip(RANKS, pooled[j]) if v >= t]
            k_needed[str(eps)][f"k{int(t*100)}_mean"] = int(hit[0]) if hit else None
            k_needed[str(eps)][f"k{int(t*100)}_pooled"] = int(hitp[0]) if hitp else None
        k_needed[str(eps)]["n_valid"] = int(valid[:, j].sum()); k_needed[str(eps)]["mean_gain"] = float(gain[:, j].mean())
    res["k_needed"] = k_needed; res["mean_frac"] = mean_frac.tolist(); res["pooled_frac"] = pooled.tolist()
    # per family
    res["k90_by_family"] = {}
    for f, name in enumerate(["NASH_LOGIT_PERTURB", "NASH_RANDOM_MIX", "STRUCTURED_CORRELATED", "UNSTRUCTURED_DIRICHLET"]):
        m = fam[test] == f
        mf = np.nanmean(frac[m], 0)
        res["k90_by_family"][name] = {str(eps): (int(next((k for k, v in zip(RANKS, mf[j]) if v >= 0.9), -1))) for j, eps in enumerate(EPS_GRID)}
    np.savez_compressed(OUT_DIR / "rank_curves.npz", retained=retained, gain=gain, V=V, base=base, frac=frac, test_ids=test)
    save_json(res, OUT_DIR / "t3_results_partial.json")
    # 3. Nash-set affine hull: eps=0 safe LP along random directions
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(400, S.n_seq[0]))
    pts = np.array([safe(d, 0.0) for d in dirs])
    pts_c = pts - pts.mean(0)
    sv = np.linalg.svd(pts_c, compute_uv=False)
    tol = 1e-6 * sv[0]
    nash_dim = int((sv > tol).sum())
    res["nash_hull"] = {"n_directions": 400, "singular_values_top20": sv[:20].tolist(), "dimension_tol1e-6": nash_dim,
                        "dimension_tol1e-4": int((sv > 1e-4 * sv[0]).sum()), "max_point_exploitability": float(max(L.exploitability(p) for p in pts[:50]))}
    # direction basis of the hull (in x-space); g components: g_N = projection of (g - mean) onto span, g_perp = rest
    Uh, sh, Vh = np.linalg.svd(pts_c, full_matrices=False)
    B = Vh[:nash_dim]                                        # (d, n0) orthonormal basis of the hull directions
    comp = np.zeros((len(test), len(EPS_GRID), 2))
    for i in range(len(test)):
        d = Gte[i] - mean_g
        gN = mean_g + d @ B.T @ B; gP = mean_g + (d - d @ B.T @ B)
        for j, eps in enumerate(EPS_GRID):
            comp[i, j, 0] = Gte[i] @ safe(gN, eps) - base[i, j]
            comp[i, j, 1] = Gte[i] @ safe(gP, eps) - base[i, j]
    cfrac = np.where(valid[:, :, None], comp / np.where(valid, gain, 1.0)[:, :, None], np.nan)
    res["component_split"] = {"within_nash_frac_by_eps": np.nanmean(cfrac[:, :, 0], 0).tolist(), "orthogonal_frac_by_eps": np.nanmean(cfrac[:, :, 1], 0).tolist(),
                              "within_nash_value_by_eps": comp[:, :, 0].mean(0).tolist(), "orthogonal_value_by_eps": comp[:, :, 1].mean(0).tolist(),
                              "gain_by_eps": gain.mean(0).tolist()}
    e = np.array(EPS_GRID[1:]); v = comp[:, 1:, 1].mean(0)
    slope = float((e @ v) / (e @ e)); r2 = float(1 - ((v - slope * e) ** 2).sum() / max(((v - v.mean()) ** 2).sum(), 1e-12))
    res["component_split"]["orthogonal_linear_fit_through_origin"] = {"slope": slope, "R2": r2}
    print(f"nash hull dim {nash_dim}; component split done ({time.time()-t0:.0f}s)", flush=True)
    # 4. kappa: dist(x, N) vs Expl(x), and the bound max_{S_eps} x^T g - max_N x^T g <= kappa eps ||g||
    qp = SafeQP(S, L.v_star)
    xs = []
    for i in rng.choice(len(test), 60, replace=False):
        for eps in [0.01, 0.02, 0.05, 0.1, 0.2, 0.4]:
            xs.append(safe(Gte[i], eps))
    for _ in range(60):
        pol = S.normalize_policy(0, rng.dirichlet(np.ones(3), size=S.n_infosets[0])); xs.append(S.behavioral_to_realization(0, pol))
    xs = np.array(xs)
    dist = np.zeros(len(xs)); expl = np.zeros(len(xs))
    for i, x in enumerate(xs):
        # distance to the Nash set: min ||x - n||^2 over n in S_0  (QP: maximize -1/2||n||^2 + x^T n)
        sol = qp.solve(x, 0.0, 1.0, tau_v=1e-6)                  # objective x^T n - 1/2 ||n||^2 over S_0
        dist[i] = np.linalg.norm(sol["x"] - x); expl[i] = L.exploitability(x)
    ok = expl > 1e-9
    kappa_dist = float(np.max(dist[ok] / expl[ok]))
    # value bound
    ratios = []
    for i in rng.choice(len(test), 100, replace=False):
        g = Gte[i]; vN = g @ safe(g, 0.0)
        for j, eps in enumerate(EPS_GRID[1:], 1):
            ratios.append(((V[i, j] - vN) / (eps * np.linalg.norm(g)), eps))
    ratios = np.array(ratios)
    kappa_val = float(ratios[:, 0].max())
    res["kappa"] = {"kappa_distance_max_ratio": kappa_dist, "kappa_distance_median_ratio": float(np.median(dist[ok] / expl[ok])),
                    "kappa_value_bound_max": kappa_val, "kappa_value_bound_by_eps": {str(e): float(ratios[ratios[:, 1] == e, 0].max()) for e in EPS_GRID[1:]},
                    "n_points_dist": int(ok.sum()), "violations_of_fitted_bound": 0}
    np.savez_compressed(OUT_DIR / "kappa_points.npz", dist=dist, expl=expl, ratios=ratios, comp=comp, nash_pts_sv=sv)
    res["audit"] = {"n_strategies_audited": n_audit, "max_violation": float(max_viol), "lp_failures": lp_fail, "runtime_s": time.time() - t0}
    save_json(res, OUT_DIR / "t3_results.json")
    print("T3 done", time.time() - t0, "s; audited", n_audit, "max viol", max_viol, "lp fails", lp_fail, flush=True)


if __name__ == "__main__":
    main()
