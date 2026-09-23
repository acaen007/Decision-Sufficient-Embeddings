"""Strategic geometry diagnostics (task section 24) and precondition analysis (section 25)."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from ..common import N_BUDGETS, EPSILONS
from ..game.policy_utils import participation_ratio, dims_for_variance, behavioral_distance_matrix

GEOM_EPS_INDEX = EPSILONS.index(0.10)
N_PAIRS = 20000
N_BINS = 20


def response_distance_matrix(X_or: np.ndarray, G: np.ndarray, V: np.ndarray) -> np.ndarray:
    """d_resp(q, q') = 0.5 [ r(q <- q') + r(q' <- q) ],  r(q <- q') = V(q) - u(x*(q'), q).

    X_or: (K, n0) oracle safe strategies at the chosen eps; G: (K, n0); V: (K,) oracle values.
    """
    U = X_or @ G.T                     # U[j, i] = u(x*(q_j), q_i)
    R = V[None, :] - U                 # R[j, i] = r(q_i <- q_j)
    R = np.maximum(R, 0.0)
    return 0.5 * (R + R.T)


def sample_pairs(K, n_pairs, seed=0):
    rng = np.random.default_rng(seed)
    all_pairs = np.array([(i, j) for i in range(K) for j in range(i + 1, K)])
    if len(all_pairs) <= n_pairs:
        return all_pairs
    return all_pairs[rng.choice(len(all_pairs), n_pairs, replace=False)]


def latent_distances(Zmean: np.ndarray, pairs):
    return np.linalg.norm(Zmean[pairs[:, 0]] - Zmean[pairs[:, 1]], axis=1)


def behavior_matched_separation(d_beh, d_resp, d_lat, n_bins=N_BINS, seed=0, n_boot=1000):
    """Within narrow d_beh bins, ratio of mean latent distance of strategically-far pairs (top
    tercile of d_resp within bin) to strategically-near pairs (bottom tercile)."""
    edges = np.quantile(d_beh, np.linspace(0, 1, n_bins + 1))
    bins = np.clip(np.searchsorted(edges, d_beh, side="right") - 1, 0, n_bins - 1)
    far_mask = np.zeros(len(d_beh), bool); near_mask = np.zeros(len(d_beh), bool)
    for b in range(n_bins):
        sel = np.flatnonzero(bins == b)
        if len(sel) < 9:
            continue
        lo, hi = np.quantile(d_resp[sel], [1 / 3, 2 / 3])
        near_mask[sel[d_resp[sel] <= lo]] = True
        far_mask[sel[d_resp[sel] >= hi]] = True
    ratio = d_lat[far_mask].mean() / d_lat[near_mask].mean()
    # sanity: matched behavioral distance
    beh_far, beh_near = d_beh[far_mask].mean(), d_beh[near_mask].mean()
    rng = np.random.default_rng(seed)
    reps = []
    fi, ni = np.flatnonzero(far_mask), np.flatnonzero(near_mask)
    for _ in range(n_boot):
        reps.append(d_lat[rng.choice(fi, len(fi))].mean() / d_lat[rng.choice(ni, len(ni))].mean())
    return {"ratio": float(ratio), "lo": float(np.percentile(reps, 2.5)), "hi": float(np.percentile(reps, 97.5)),
            "d_beh_far": float(beh_far), "d_beh_near": float(beh_near), "n_far": int(far_mask.sum()), "n_near": int(near_mask.sum()),
            "d_resp_far": float(d_resp[far_mask].mean()), "d_resp_near": float(d_resp[near_mask].mean())}


def illustrative_pairs(d_beh, d_resp, pairs, q_lo=0.10, q_hi=0.90):
    """A: behaviorally distant (>= 90th pct d_beh) but response-near (<= 10th pct d_resp): max d_beh.
       B: behaviorally similar (<= 10th pct d_beh) but response-far (>= 90th pct d_resp): max d_resp."""
    bl, bh = np.quantile(d_beh, [q_lo, q_hi]); rl, rh = np.quantile(d_resp, [q_lo, q_hi])
    A = np.flatnonzero((d_resp <= rl) & (d_beh >= bh))
    B = np.flatnonzero((d_beh <= bl) & (d_resp >= rh))
    out = {}
    if len(A):
        i = A[np.argmax(d_beh[A])]
        out["A_behavior_far_response_near"] = {"pair": pairs[i].tolist(), "d_beh": float(d_beh[i]), "d_resp": float(d_resp[i])}
    else:   # relax: minimize d_resp among top-decile d_beh
        cand = np.flatnonzero(d_beh >= bh); i = cand[np.argmin(d_resp[cand])]
        out["A_behavior_far_response_near"] = {"pair": pairs[i].tolist(), "d_beh": float(d_beh[i]), "d_resp": float(d_resp[i]), "relaxed": True}
    if len(B):
        i = B[np.argmax(d_resp[B])]
        out["B_behavior_near_response_far"] = {"pair": pairs[i].tolist(), "d_beh": float(d_beh[i]), "d_resp": float(d_resp[i])}
    else:
        cand = np.flatnonzero(d_beh <= bl); i = cand[np.argmax(d_resp[cand])]
        out["B_behavior_near_response_far"] = {"pair": pairs[i].tolist(), "d_beh": float(d_beh[i]), "d_resp": float(d_resp[i]), "relaxed": True}
    return out


def geometry_analysis(ed, pop, sym, reach_weights, n_pairs=N_PAIRS, seed=0):
    """ed: EvalData (test split). Returns dict with correlations, separation ratios, pairs, raw arrays."""
    k = GEOM_EPS_INDEX
    ids = ed.opp_ids
    X_or = pop["X_oracle"][ids, k]; G = pop["G"][ids]; V = pop["V_oracle"][ids, k]
    D_resp = response_distance_matrix(X_or, G, V)
    Q = pop["rank_policies"][ids]
    D_beh = behavioral_distance_matrix(Q, sym.rank_legal_mask[1])
    D_beh_w = behavioral_distance_matrix(Q, sym.rank_legal_mask[1], weights=reach_weights)
    D_g = np.linalg.norm(G[:, None] - G[None], axis=2)
    pairs = sample_pairs(len(ids), n_pairs, seed)
    d_resp = D_resp[pairs[:, 0], pairs[:, 1]]; d_beh = D_beh[pairs[:, 0], pairs[:, 1]]
    d_beh_w = D_beh_w[pairs[:, 0], pairs[:, 1]]; d_g = D_g[pairs[:, 0], pairs[:, 1]]
    res = {"eps": EPSILONS[k], "n_pairs": int(len(pairs)),
           "rho_beh_resp": float(spearmanr(d_beh, d_resp)[0]), "rho_behw_resp": float(spearmanr(d_beh_w, d_resp)[0]),
           "rho_g_resp": float(spearmanr(d_g, d_resp)[0]), "rho_g_beh": float(spearmanr(d_g, d_beh)[0]),
           "separation_true_g": behavior_matched_separation(d_beh, d_resp, d_g, seed=seed),
           "separation_behavior_itself": behavior_matched_separation(d_beh, d_resp, d_beh, seed=seed),
           "latent": {}, "pairs_examples": illustrative_pairs(d_beh, d_resp, pairs)}
    j100 = N_BUDGETS.index(100)
    for m, p in ed.pred.items():
        if "z" not in p:
            continue
        Zm = ed.per_opp(p["z"])                          # (n_opp, nN, d)
        entry = {}
        for j, N in enumerate(N_BUDGETS):
            d_lat = latent_distances(Zm[:, j], pairs)
            entry[str(N)] = {"rho_z_beh": float(spearmanr(d_lat, d_beh)[0]), "rho_z_behw": float(spearmanr(d_lat, d_beh_w)[0]),
                             "rho_z_resp": float(spearmanr(d_lat, d_resp)[0]), "rho_z_g": float(spearmanr(d_lat, d_g)[0]),
                             "separation": behavior_matched_separation(d_beh, d_resp, d_lat, seed=seed)}
        entry["N100"] = entry[str(100)]
        res["latent"][m] = entry
    raw = {"pairs": pairs, "d_resp": d_resp, "d_beh": d_beh, "d_beh_w": d_beh_w, "d_g": d_g,
           "D_resp": D_resp, "D_beh": D_beh}
    return res, raw


def precondition_analysis(pop, sym, ed_test=None, bank_family_mass=None):
    """Population-level structure: variation, PCA dims of policies and g, gains, d_beh-d_resp."""
    Q = pop["rank_policies"]; G = pop["G"]; fam = pop["family_index"]; split = pop["split"]
    mask = sym.rank_legal_mask[1].astype(bool)
    Qf = (Q * mask[None]).reshape(len(Q), -1)
    train = split == 0
    res = {"n_opponents": int(len(Q)), "policy_dims": int(mask.sum()), "g_dims": int(G.shape[1]),
           "g_nonconstant_dims": int((G[train].std(0) > 1e-3 * G[train].std(0).max()).sum())}
    res["policy_pr"] = participation_ratio(Qf[train]); res["policy_pca"] = dims_for_variance(Qf[train])
    res["g_pr"] = participation_ratio(G[train]); res["g_pca"] = dims_for_variance(G[train])
    # relative to the g scale (use standardized g over valid dims)
    std = G[train].std(0); valid = std > 1e-3 * std.max()
    Gs = (G[train][:, valid] - G[train][:, valid].mean(0)) / std[valid]
    res["g_standardized_pr"] = participation_ratio(Gs); res["g_standardized_pca"] = dims_for_variance(Gs)
    res["by_family"] = {}
    names = ["NASH_LOGIT_PERTURB", "NASH_RANDOM_MIX", "STRUCTURED_CORRELATED", "UNSTRUCTURED_DIRICHLET"]
    ex = pop["br_val"] - pop["v_star"]
    gain = pop["V_oracle"] - pop["nash_val"][:, None]
    for f, n in enumerate(names):
        s = (fam == f) & train
        res["by_family"][n] = {"policy_pr": participation_ratio(Qf[s]), "g_pr": participation_ratio(G[s]),
                               "g_standardized_pr": participation_ratio((G[s][:, valid] - G[train][:, valid].mean(0)) / std[valid]),
                               "exploitability_mean": float(ex[fam == f].mean()), "exploitability_std": float(ex[fam == f].std()),
                               "gain_mean_by_eps": gain[fam == f].mean(0).tolist(),
                               "gain_quantiles_eps0.1": np.quantile(gain[fam == f, 2], [0.1, 0.5, 0.9]).tolist(),
                               "mean_policy_variation": float(np.sqrt(((Qf[s] - Qf[s].mean(0)) ** 2).sum(1)).mean())}
    res["exploitability_all_mean"] = float(ex.mean()); res["gain_all_mean_by_eps"] = gain.mean(0).tolist()
    res["gain_all_quantiles_by_eps"] = {str(e): np.quantile(gain[:, k], [0.1, 0.25, 0.5, 0.75, 0.9]).tolist() for k, e in enumerate(EPSILONS)}
    if bank_family_mass is not None and ed_test is not None:
        # family identifiability from short histories: MAP family accuracy of the bank posterior
        fm = ed_test.per_opp(bank_family_mass)                    # (n_opp, nN, 4)
        acc = [(np.argmax(fm[:, j], 1) == ed_test.family).mean() for j in range(len(N_BUDGETS))]
        true_mass = [fm[np.arange(len(fm)), j, ed_test.family].mean() for j in range(len(N_BUDGETS))]
        res["family_identifiability"] = {"map_accuracy_by_N": [float(a) for a in acc],
                                         "true_family_mass_by_N": [float(a) for a in true_mass],
                                         "per_family_accuracy_by_N": {n: [float((np.argmax(fm[ed_test.family == f, j], 1) == f).mean()) for j in range(len(N_BUDGETS))] for f, n in enumerate(names)}}
    return res
