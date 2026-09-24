"""JAC-opp Step 1 (gate, no training): per-opponent decision-relevance weights of the reconstruction loss.

For every training opponent q (rank policy, 144 rank infosets x 3 action slots) and every opponent rank infoset I:
  J(q)      = dg/dq, g = A y_q (raw chips), built analytically: y_q(s) = prod of q along s's chain, so
              dy(s)/dq(j) = product of the other factors (leave-one-out) and J = A @ D (D sparse, <= 4 nnz per row)
  C(q)      = same with only the factors BELOW j kept (payoff-vector consequence c_{I,a}; J_I = r_I C_I)
  w_I(q)    = ||J_I(q) P_I||_F^2, P_I = I - 11^T/k on the k legal action slots          (projected, squared)
  u_I(q)    = ||J_I(q)||_F^2 on the legal slots                                           (unprojected, squared)
  reach_I   = r_I(q)^2, r_I = opponent's own reach to I (identical for all suit-isomorphic members of the class)
  cons_I    = ||C_I(q) P_I||_F^2                                                          (consequence only)
Checks: J vs central finite differences of the float64 map (coordinates and projected directions); J = r * C;
reproduction of V3's weight file (mean over training opponents of ||J_I||_F over all 3 slots).
Writes outputs/jacopp/{weights.npz, gate.json, gate_mass.png}.
"""
import json, time
import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr, pearsonr
from .common import OUT, save_json
from .game.leduc_tree import get_tree
from .game.sequence_form import get_sequence_form
from .game.symmetry import get_symmetry
from .game.policy_utils import RankPolicyToG
from .data.datasets import load_population
from .v3_analysis import FAMS

D_OUT = OUT / "jacopp"
ACT = "fcr"                                   # 0 fold, 1 call/check, 2 raise
RANKS = "JQK"


class Jac:
    def __init__(self):
        self.T, self.S, self.sym = get_tree(), get_sequence_form(), get_symmetry()
        self.r2g = RankPolicyToG(self.T, self.S, self.sym)
        self.A = sp.csr_matrix(self.S.A)                                   # (n0, n1): g = A y
        self.idx, self.valid = self.r2g.idx, self.r2g.valid              # (n1, 4)
        self.n1 = self.idx.shape[0]
        self.legal = np.asarray(self.sym.rank_legal_mask[1], dtype=bool)  # (144, 3)
        self.k = self.legal.sum(1)
        inf = self.T.infosets[1]
        self.members = self.sym.rank_members[1]
        self.par = np.asarray(inf.parent_seq)
        rows, cols = np.nonzero(self.valid)
        self.rows, self.cols, self.jj = rows, cols, self.idx[rows, cols]

    def factors(self, q):
        Q = q.reshape(-1)
        fac = np.where(self.valid, Q[self.idx], 1.0)                      # (n1, 4)
        pre = np.ones_like(fac); suf = np.ones_like(fac)
        for c in range(1, 4):
            pre[:, c] = pre[:, c - 1] * fac[:, c - 1]
        for c in range(2, -1, -1):
            suf[:, c] = suf[:, c + 1] * fac[:, c + 1]
        return pre, suf

    def jac(self, q):
        """Dense J (n0, 144, 3), C (n0, 144, 3) and own reach r (144,) at rank policy q (144, 3)."""
        pre, suf = self.factors(q)
        loo = (pre * suf)[self.rows, self.cols]; below = suf[self.rows, self.cols]
        D = sp.csr_matrix((loo, (self.rows, self.jj)), shape=(self.n1, 432))
        Dc = sp.csr_matrix((below, (self.rows, self.jj)), shape=(self.n1, 432))
        J = (self.A @ D).toarray().reshape(-1, 144, 3)
        C = (self.A @ Dc).toarray().reshape(-1, 144, 3)
        y = self.r2g.realization(q[None])[0]
        r = np.full(144, np.nan); spread = 0.0
        for I in range(144):
            v = y[self.par[self.members[I]]]
            r[I] = v[0]; spread = max(spread, float(np.ptp(v)))
        return J, C, r, spread

    def proj_norm2(self, M):
        """||M_I P_I||_F^2 per infoset for M (n0, 144, 3); illegal slots excluded."""
        Ml = np.where(self.legal[None], M, 0.0)
        tot = (Ml ** 2).sum(axis=(0, 2))
        s = Ml.sum(axis=2)                                                # M_I 1 (legal slots)
        return tot - (s ** 2).sum(0) / self.k, tot


def infoset_table(jc):
    T = jc.T; inf = T.infosets[1]; out = []
    for I in range(144):
        m = jc.members[I][0]
        pr, pub, hist = jc.sym.rank_keys[1][I]
        rnd = int(inf.round[m]); nthis = int(inf.n_actions_this_round[m])
        h = "".join(ACT[a] for a in hist)
        h1, h2 = (h, "") if rnd == 1 else (h[: len(h) - nthis], h[len(h) - nthis:])
        desc = f"{RANKS[pr]}" + (f"|{RANKS[pub]}" if pub >= 0 else "") + f" r{rnd} {h1}" + (f"/{h2}" if rnd == 2 else "")
        out.append({"I": I, "round": rnd, "depth": int(inf.depth[m]), "private": int(pr), "public": int(pub), "facing_raise": int(inf.facing_raise[m]),
                    "n_legal": int(jc.k[I]), "hist_len": len(hist), "desc": desc})
    return out


def fd_checks(jc, Qs, rng, n_opp=5, n_coord=40, n_dir=10, h=1e-6):
    rel_c, rel_d, jrc, spread = [], [], [], 0.0
    g = lambda q: jc.r2g.g(q[None])[0]
    for q in Qs[:n_opp]:
        J, C, r, sp_ = jc.jac(q); spread = max(spread, sp_)
        jrc.append(float(np.abs(np.where(jc.legal[None], J - r[None, :, None] * C, 0)).max() / (np.abs(J).max() + 1e-300)))
        legal_flat = np.flatnonzero(jc.legal.reshape(-1))
        for j in rng.choice(legal_flat, n_coord, replace=False):
            e = np.zeros(432); e[j] = h
            fd = (g(q + e.reshape(144, 3)) - g(q - e.reshape(144, 3))) / (2 * h)
            an = J.reshape(-1, 432)[:, j]
            if np.abs(an).max() > 0 or np.abs(fd).max() > 0:
                rel_c.append(float(np.linalg.norm(fd - an) / max(np.linalg.norm(an), 1e-12)))
        for _ in range(n_dir):                                            # tangent (sum-zero) direction at one infoset
            I = rng.integers(144); d = np.zeros((144, 3)); v = rng.normal(size=jc.k[I]); v -= v.mean()
            d[I, jc.legal[I]] = v
            fd = (g(q + h * d) - g(q - h * d)) / (2 * h)
            an = J[:, I, jc.legal[I]] @ v
            if np.linalg.norm(an) > 1e-12:
                rel_d.append(float(np.linalg.norm(fd - an) / np.linalg.norm(an)))
    return {"coord_rel_err_median": float(np.median(rel_c)), "coord_rel_err_max": float(np.max(rel_c)), "n_coord": len(rel_c),
            "tangent_rel_err_median": float(np.median(rel_d)), "tangent_rel_err_max": float(np.max(rel_d)), "n_dir": len(rel_d),
            "J_eq_r_times_C_max_rel": float(max(jrc)), "reach_spread_within_class_max": spread, "h": h, "n_opp": n_opp}


def summ(x):
    x = np.asarray(x, dtype=float)
    return {"median": float(np.median(x)), "q25": float(np.percentile(x, 25)), "q75": float(np.percentile(x, 75)),
            "p05": float(np.percentile(x, 5)), "p95": float(np.percentile(x, 95)), "mean": float(x.mean()), "min": float(x.min())}


def norm_rows(W):
    s = W.sum(1, keepdims=True)
    return W / np.where(s > 0, s, 1.0)


def cosine_to_mean(Wn):
    m = Wn.mean(0)
    return (Wn @ m) / (np.linalg.norm(Wn, axis=1) * np.linalg.norm(m)), m


def jaccard_by_family(Wn, fam, k):
    top = np.argsort(-Wn, axis=1, kind="stable")[:, :k]
    X = np.zeros_like(Wn, dtype=np.float32); np.put_along_axis(X, top, 1.0, axis=1)
    inter = X @ X.T; jac = inter / (2 * k - inter)
    n = len(Wn); iu = np.triu_indices(n, 1)
    out = {"all_pairs_mean": float(jac[iu].mean())}
    fp = {}
    for a in range(4):
        for b in range(a, 4):
            ia, ib = np.flatnonzero(fam == a), np.flatnonzero(fam == b)
            sub = jac[np.ix_(ia, ib)]
            v = sub[np.triu_indices(len(ia), 1)] if a == b else sub.reshape(-1)
            fp[f"{FAMS[a]}|{FAMS[b]}"] = float(v.mean())
    out["family_pairs_mean"] = fp
    return out


def behaviour_stats(Q, table, legal):
    """Simple style descriptors of each opponent (unweighted over rank infosets)."""
    r1_face = [t["I"] for t in table if t["round"] == 1 and t["facing_raise"] == 1]
    fold_r1 = Q[:, r1_face, 0].mean(1)
    aggr = Q[:, legal[:, 2], 2].mean(1)
    return fold_r1, aggr


def main():
    D_OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); rng = np.random.default_rng(0)
    jc = Jac(); pop = load_population()
    tr = np.flatnonzero(pop["split"] == 0); fam = np.asarray(pop["family_index"])[tr]
    Q = np.asarray(pop["rank_policies"], dtype=np.float64)
    table = infoset_table(jc)
    res = {"n_train": int(len(tr)), "n_rank_infosets": 144, "n_physical_infosets": 468,
           "n_legal_hist": np.bincount(jc.k).tolist()}

    # ---- checks
    res["checks"] = fd_checks(jc, Q[tr[rng.choice(len(tr), 5, replace=False)]], rng)
    print("checks", json.dumps(res["checks"]), flush=True)

    # ---- weights for every training opponent
    K = len(pop["split"])
    Wp = np.zeros((K, 144)); Wu = np.zeros((K, 144)); R2 = np.zeros((K, 144)); Cp = np.zeros((K, 144)); V3 = np.zeros((K, 144))
    illegal_max = 0.0
    for n, o in enumerate(tr):
        J, C, r, _ = jc.jac(Q[o])
        illegal_max = max(illegal_max, float(np.abs(J[:, ~jc.legal]).max()))
        Wp[o], Wu[o] = jc.proj_norm2(J)
        Cp[o], _ = jc.proj_norm2(C)
        R2[o] = r ** 2
        V3[o] = np.sqrt((J ** 2).sum(axis=(0, 2)))
        if (n + 1) % 300 == 0:
            print(f"{n+1}/{len(tr)} opponents ({time.time()-t0:.0f}s)", flush=True)
    v3_file = np.load(OUT / "weights_v3" / "recon_weights_jacobian.npy")
    v3_rep = V3[tr].mean(0)
    res["checks"]["illegal_slot_J_max"] = illegal_max
    res["checks"]["v3_weight_reproduction_max_rel_err"] = float(np.abs(v3_rep - v3_file).max() / np.abs(v3_file).max())
    np.savez_compressed(D_OUT / "weights.npz", w_proj=Wp, w_unproj=Wu, reach2=R2, cons_proj=Cp, v3_norm=V3, train_ids=tr)

    Wn = norm_rows(Wp[tr]); Un = norm_rows(Wu[tr]); Rn = norm_rows(R2[tr]); Cn = norm_rows(Cp[tr])
    zero_total = int((Wp[tr].sum(1) <= 0).sum())
    res["zero_total_opponents"] = zero_total

    # a. cosine to the population average
    cos, m = cosine_to_mean(Wn)
    res["a_cosine_to_mean"] = {"all": summ(cos), **{FAMS[f]: summ(cos[fam == f]) for f in range(4)}}
    # b. top-k overlap
    res["b_topk_jaccard"] = {f"top{k}": jaccard_by_family(Wn, fam, k) for k in (10, 20)}
    # c. coefficient of variation across opponents, per infoset
    mu = Wn.mean(0); cv = Wn.std(0) / np.where(mu > 0, mu, np.nan)
    order = np.argsort(-mu)
    res["c_cv_per_infoset"] = {"all": summ(cv[np.isfinite(cv)]), "mass_weighted_mean": float(np.nansum(cv * mu) / mu[np.isfinite(cv)].sum()),
                               "top20_by_mean_mass": summ(cv[order[:20]]),
                               "per_infoset": [{"I": int(I), "desc": table[I]["desc"], "mean_w": float(mu[I]), "cv": float(cv[I])} for I in order]}
    # d. effectively zero weights
    z = Wn < 1e-4
    res["d_frac_below_1e-4"] = {"all": float(z.mean()), **{FAMS[f]: float(z[fam == f].mean()) for f in range(4)},
                                "per_opponent": summ(z.mean(1))}
    # e. mass by round / depth / private rank, per family
    rnd = np.array([t["round"] for t in table]); dep = np.array([t["depth"] for t in table]); prv = np.array([t["private"] for t in table])
    groups = sorted({(int(a), int(b)) for a, b in zip(rnd, dep)})
    mass = {}
    for f in range(4):
        W_f = Wn[fam == f]
        mass[FAMS[f]] = {"round_depth": {f"r{a}d{b}": float(W_f[:, (rnd == a) & (dep == b)].sum(1).mean()) for a, b in groups},
                         "private_rank": {RANKS[p]: float(W_f[:, prv == p].sum(1).mean()) for p in range(3)},
                         "round2_mass": summ(W_f[:, rnd == 2].sum(1))}
    mass["ALL"] = {"round_depth": {f"r{a}d{b}": float(Wn[:, (rnd == a) & (dep == b)].sum(1).mean()) for a, b in groups},
                   "private_rank": {RANKS[p]: float(Wn[:, prv == p].sum(1).mean()) for p in range(3)}}
    # V3 global weights for comparison (normalized to sum 1)
    v3n = v3_file / v3_file.sum()
    mass["V3_GLOBAL"] = {"round_depth": {f"r{a}d{b}": float(v3n[(rnd == a) & (dep == b)].sum()) for a, b in groups},
                         "private_rank": {RANKS[p]: float(v3n[prv == p].sum()) for p in range(3)}}
    fold_r1, aggr = behaviour_stats(Q[tr], table, jc.legal)
    r2m = Wn[:, rnd == 2].sum(1); deep = Wn[:, (rnd == 2) & (dep >= 2)].sum(1)
    mass["style_correlations"] = {"spearman_round2_mass_vs_fold_r1": float(spearmanr(r2m, fold_r1)[0]),
                                  "spearman_round2_mass_vs_raise_freq": float(spearmanr(r2m, aggr)[0]),
                                  "spearman_deep_round2_mass_vs_fold_r1": float(spearmanr(deep, fold_r1)[0]),
                                  "spearman_deep_round2_mass_vs_raise_freq": float(spearmanr(deep, aggr)[0]),
                                  "definitions": "fold_r1 = mean fold prob over round-1 rank infosets facing a raise; raise_freq = mean raise prob over rank infosets where raising is legal (unweighted); deep = round 2, own decision depth >= 2"}
    res["e_mass"] = mass
    # f. reach alone
    sp_reach = np.array([spearmanr(Wn[i], Rn[i])[0] for i in range(len(tr))])
    sp_cons = np.array([spearmanr(Wn[i], Cn[i])[0] for i in range(len(tr))])
    cos_c, _ = cosine_to_mean(Cn); cos_r, _ = cosine_to_mean(Rn)
    # across opponents, per infoset: how much of the variation of log w~_I tracks log reach~_I
    with np.errstate(divide="ignore"):
        lw, lr, lc = np.log(Wn), np.log(Rn), np.log(Cn)
    fin = np.isfinite(lw) & np.isfinite(lr) & np.isfinite(lc)
    share = []
    for I in range(144):
        ok = fin[:, I]
        if ok.sum() > 50 and lw[ok, I].var() > 0:
            share.append(np.corrcoef(lw[ok, I], lr[ok, I])[0, 1] ** 2)
    res["f_reach"] = {"per_opp_spearman_w_vs_reach2": {"all": summ(sp_reach), **{FAMS[f]: summ(sp_reach[fam == f]) for f in range(4)}},
                      "per_opp_spearman_w_vs_consequence": {"all": summ(sp_cons), **{FAMS[f]: summ(sp_cons[fam == f]) for f in range(4)}},
                      "cosine_to_mean_reach2_only": summ(cos_r), "cosine_to_mean_consequence_only": summ(cos_c),
                      "per_infoset_R2_logw_on_logreach_across_opponents": summ(share), "n_infosets_R2": len(share)}
    # g. projected vs unprojected
    sp_pu = np.array([spearmanr(Wn[i], Un[i])[0] for i in range(len(tr))])
    pe_pu = np.array([pearsonr(Wn[i], Un[i])[0] for i in range(len(tr))])
    ratio = Wp[tr].sum(0) / np.where(Wu[tr].sum(0) > 0, Wu[tr].sum(0), np.nan)       # population share kept by P
    mu_u = Un.mean(0); change = mu / mu_u
    ch_order = np.argsort(change)
    res["g_projection"] = {"per_opp_spearman": summ(sp_pu), "per_opp_pearson": summ(pe_pu),
                           "kept_fraction_per_infoset": summ(ratio[np.isfinite(ratio)]),
                           "kept_fraction_by_n_legal": {str(kk): float(np.nanmean(ratio[jc.k == kk])) for kk in (2, 3)},
                           "largest_down": [{"desc": table[I]["desc"], "n_legal": table[I]["n_legal"], "mean_w_proj": float(mu[I]), "mean_w_unproj": float(mu_u[I]),
                                             "ratio_of_normalized_means": float(change[I]), "kept_fraction": float(ratio[I])} for I in ch_order[:10]],
                           "largest_up": [{"desc": table[I]["desc"], "n_legal": table[I]["n_legal"], "mean_w_proj": float(mu[I]), "mean_w_unproj": float(mu_u[I]),
                                           "ratio_of_normalized_means": float(change[I]), "kept_fraction": float(ratio[I])} for I in ch_order[::-1][:10]]}
    # arm-1 global weights vs V3 global weights
    res["arm1_vs_v3"] = {"cosine": float(m @ v3n / np.linalg.norm(m) / np.linalg.norm(v3n)), "spearman": float(spearmanr(m, v3n)[0]),
                         "max_over_min_ratio_arm1": float(m.max() / m[m > 0].min()), "max_over_min_ratio_v3": float(v3n.max() / v3n.min()),
                         "cv_arm1": float(m.std() / m.mean()), "cv_v3": float(v3n.std() / v3n.mean())}
    # gate
    med_cos = res["a_cosine_to_mean"]["all"]["median"]; top20 = res["b_topk_jaccard"]["top20"]["all_pairs_mean"]
    res["gate"] = {"median_cosine": med_cos, "mean_top20_jaccard": top20, "rule": "little to reallocate iff median cosine >= 0.95 AND mean top-20 overlap >= 0.9",
                   "little_to_reallocate": bool(med_cos >= 0.95 and top20 >= 0.9),
                   "decision": "run arms 0-1 only" if (med_cos >= 0.95 and top20 >= 0.9) else "run all arms"}
    res["infosets"] = table
    res["runtime_s"] = time.time() - t0
    save_json(res, D_OUT / "gate.json")
    figure(Wn, fam, table, rnd, dep, prv, groups, mass, fold_r1, r2m, cos, sp_reach)
    print(json.dumps({k: res[k] for k in ["gate", "arm1_vs_v3"]}, indent=1))
    print(f"done in {time.time()-t0:.0f}s")


def figure(Wn, fam, table, rnd, dep, prv, groups, mass, fold_r1, r2m, cos, sp_reach):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(19, 5.2))
    cols = plt.cm.viridis(np.linspace(0.05, 0.95, len(groups)))
    names = ["NASH\nLOGIT", "NASH\nMIX", "STRUCT\nCORR", "DIRICH-\nLET", "V3\nglobal"]
    keys = FAMS + ["V3_GLOBAL"]
    bottom = np.zeros(len(keys))
    for gi, (a, b) in enumerate(groups):
        v = np.array([mass[k]["round_depth"][f"r{a}d{b}"] for k in keys])
        ax[0].bar(range(len(keys)), v, bottom=bottom, color=cols[gi], label=f"round {a}, own depth {b}"); bottom += v
    ax[0].set_xticks(range(len(keys))); ax[0].set_xticklabels(names, fontsize=7); ax[0].set_ylabel("mean share of w~ (per opponent)")
    ax[0].set_title("(e) weight mass by round and opponent decision depth", fontsize=9); ax[0].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    bottom = np.zeros(len(keys)); pc = ["#4c72b0", "#dd8452", "#55a868"]
    for p in range(3):
        v = np.array([mass[k]["private_rank"][RANKS[p]] for k in keys])
        ax[1].bar(range(len(keys)), v, bottom=bottom, color=pc[p], label=f"private {RANKS[p]}"); bottom += v
    ax[1].set_xticks(range(len(keys))); ax[1].set_xticklabels(names, fontsize=7); ax[1].set_title("(e) weight mass by private card", fontsize=9)
    ax[1].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    fc = ["C0", "C1", "C2", "C3"]
    for f in range(4):
        s = fam == f
        ax[2].scatter(fold_r1[s], r2m[s], s=5, alpha=0.5, color=fc[f], label=FAMS[f])
    ax[2].set_xlabel("round-1 fold probability facing a raise (mean over rank infosets)"); ax[2].set_ylabel("round-2 share of w~")
    ax[2].set_title("(e) tightness vs round-2 mass, per opponent", fontsize=9); ax[2].legend(fontsize=6, markerscale=3)
    for f in range(4):
        s = fam == f
        ax[3].scatter(cos[s], sp_reach[s], s=5, alpha=0.5, color=fc[f])
    ax[3].set_xlabel("(a) cosine of w~ to population mean"); ax[3].set_ylabel("(f) Spearman(w~, reach^2)")
    ax[3].set_title("opponent-specificity vs reach explanation", fontsize=9)
    fig.tight_layout(); fig.savefig(D_OUT / "gate_mass.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
