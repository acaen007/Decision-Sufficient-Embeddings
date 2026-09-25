"""Non-stationary opponents: analysis per REPORT_LEDUC_NONSTATIONARY.md section 0.  Writes outputs/nonstat/analysis.json."""
import json
import numpy as np
from .common import OUT, save_json
from .ns_eval import CKPT, SWITCH_AT, WINS, GAMMAS

D = OUT / "nonstat"; EPS = 0.10; B = 2000
POST_T = [205, 210, 220, 250, 300]; POSTALL_T = [205, 210, 220, 250, 300, 400, 500]
DEFAULT = ["FIXED-NE", "BANK", "TAB-EM", "WIN-EM W=50", "DISC-EM g=0.98", "JAC-opp", "JAC-opp-W50", "DEC-889k", "PRIOR-EM", "PRIOR-EM-WIN"]
GRID = [f"WIN-EM W={w}" for w in WINS] + [f"DISC-EM g={g}" for g in GAMMAS]


def per_process(kind):
    P = np.load(D / "processes.npz", allow_pickle=True); S = np.load(D / "solve.npz", allow_pickle=True)
    pidx = np.flatnonzero(P["kind"] == kind); nT = len(CKPT[kind])
    ms = ["FIXED-NE"] + [str(m) for m in S[f"{kind}_methods"]]
    u = S[f"{kind}_u"]; u = u.reshape(u.shape[0], -1, 2, nT).mean(2)                         # (M-1, P, nT) streams averaged
    U = np.concatenate([S[f"{kind}_u_fixed_ne"][None], u], 0)
    V0, Ve = P["V0"][pidx][:, :nT], P["Veps"][pidx][:, :nT]
    return P, S, pidx, ms, U, V0, Ve


def curves(U, V0, Ve, idx):
    """Pooled (ratio of means) fraction and mean regret per method and checkpoint over processes idx; exclusions by headroom."""
    hd = Ve[idx] - V0[idx]; keep = hd >= 0.01
    num = np.where(keep[None], U[:, idx] - V0[idx][None], 0.0).sum(1); den = np.where(keep, hd, 0.0).sum(0)
    frac = num / np.maximum(den, 1e-12); reg = (Ve[idx][None] - U[:, idx]).mean(1)
    return frac, reg


def recovery_time(fr, pre, level, Ts):
    """Hands after the switch until the pooled fraction first reaches level*pre (linear interpolation between post checkpoints)."""
    target = level * pre; post = [(t - SWITCH_AT, fr[Ts.index(t)]) for t in Ts if t > SWITCH_AT]
    if post[0][1] >= target:
        return float(post[0][0])
    for (h0, f0), (h1, f1) in zip(post[:-1], post[1:]):
        if f1 >= target:
            return float(h0 + (target - f0) / (f1 - f0) * (h1 - h0))
    return float("inf")


def summarize_switch(ms, U, V0, Ve, idx, rng):
    Ts = CKPT["SWITCH"]; j200 = Ts.index(200); jp = [Ts.index(t) for t in POST_T]; ja = [Ts.index(t) for t in POSTALL_T]
    def stats(ix):
        fr, rg = curves(U, V0, Ve, ix); pre = fr[:, j200]; post = fr[:, jp].mean(1); postall = fr[:, ja].mean(1)
        rec = {p: np.array([recovery_time(fr[m], pre[m], p, Ts) if m > 0 else np.nan for m in range(len(ms))]) for p in (0.5, 0.8)}
        return fr, rg, pre, post, postall, rec
    fr, rg, pre, post, postall, rec = stats(idx)
    bs = [stats(rng.choice(idx, len(idx), replace=True)) for _ in range(B)]
    q = lambda arr: (np.percentile(arr, 2.5, 0).tolist(), np.percentile(arr, 97.5, 0).tolist())
    out = {"n_processes": int(len(idx)), "n_excluded_points": {str(t): int(((Ve[idx, j] - V0[idx, j]) < 0.01).sum()) for j, t in enumerate(Ts)},
           "fraction": {m: {"mean": fr[i].tolist(), "ci": q(np.array([b[0][i] for b in bs]))} for i, m in enumerate(ms)},
           "regret": {m: {"mean": rg[i].tolist(), "ci": q(np.array([b[1][i] for b in bs]))} for i, m in enumerate(ms)},
           "pre": {m: float(pre[i]) for i, m in enumerate(ms)}, "pre_ci": {m: q(np.array([b[2][i] for b in bs])) for i, m in enumerate(ms)},
           "post_mean": {m: float(post[i]) for i, m in enumerate(ms)}, "post_mean_ci": {m: q(np.array([b[3][i] for b in bs])) for i, m in enumerate(ms)},
           "post_all_mean": {m: float(postall[i]) for i, m in enumerate(ms)}, "post_all_mean_ci": {m: q(np.array([b[4][i] for b in bs])) for i, m in enumerate(ms)},
           "relative_recovery": {m: float(post[i] / pre[i]) if pre[i] > 0 else None for i, m in enumerate(ms)}}
    out["recovery"] = {}
    for p in (0.5, 0.8):
        R = np.array([b[5][p] for b in bs])
        out["recovery"][str(p)] = {m: {"hands": float(rec[p][i]), "ci": [float(np.percentile(R[:, i], 2.5)), float(np.percentile(R[:, i], 97.5))],
                                       "p_not_recovered": float(np.mean(~np.isfinite(R[:, i])))} for i, m in enumerate(ms) if i > 0}
    ix = {m: i for i, m in enumerate(ms)}
    def paired(a, b, which):     # difference of a pooled statistic between methods a and b over the same resamples
        k = {"post": 3, "pre": 2, "postall": 4}[which]; d = np.array([bb[k][ix[a]] - bb[k][ix[b]] for bb in bs])
        base = {"post": post, "pre": pre, "postall": postall}[which]
        return {"diff": float(base[ix[a]] - base[ix[b]]), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    def rel(a, b):
        d = np.array([bb[3][ix[a]] / bb[2][ix[a]] - bb[3][ix[b]] / bb[2][ix[b]] for bb in bs])
        return {"diff": float(post[ix[a]] / pre[ix[a]] - post[ix[b]] / pre[ix[b]]), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    return out, paired, rel


def main():
    rng = np.random.default_rng(0); res = {}
    # ---------------- SWITCH
    P, S, pidx, ms, U, V0, Ve = per_process("SWITCH"); sel = P["selection"][pidx]
    res["SWITCH"] = {"methods": ms, "checkpoints": CKPT["SWITCH"], "pair_dist": {g: {"median": float(np.median(P["pair_dist"][pidx][sel == g])), "min": float(P["pair_dist"][pidx][sel == g].min()),
                                                                                     "max": float(P["pair_dist"][pidx][sel == g].max())} for g in ("contrasting", "random")},
                     "median_pool_dist": float(P["median_pool_dist"])}
    helpers = {}
    for g in ("contrasting", "random"):
        idx = np.flatnonzero(sel == g); out, paired, rel = summarize_switch(ms, U, V0, Ve, idx, rng); res["SWITCH"][g] = out; helpers[g] = (paired, rel)
    # harm rate vs t (streams averaged; u < u_FIXED-NE)
    res["SWITCH"]["harm_rate"] = {m: (U[i] < U[0] - 1e-9).mean(0).tolist() for i, m in enumerate(ms)}
    # ---------------- DRIFT
    Pd, Sd, pidd, msd, Ud, V0d, Ved = per_process("DRIFT"); seld = Pd["selection"][pidd]
    res["DRIFT"] = {"methods": msd, "checkpoints": CKPT["DRIFT"], "pair_dist": {g: float(np.median(Pd["pair_dist"][pidd][seld == g])) for g in ("contrasting", "random")}}
    for g in ("contrasting", "random", "all"):
        idx = np.arange(len(pidd)) if g == "all" else np.flatnonzero(seld == g)
        fr, rg = curves(Ud, V0d, Ved, idx); bs = [curves(Ud, V0d, Ved, rng.choice(idx, len(idx), replace=True)) for _ in range(B)]
        q = lambda arr: (np.percentile(arr, 2.5, 0).tolist(), np.percentile(arr, 97.5, 0).tolist())
        res["DRIFT"][g] = {"n_processes": int(len(idx)), "n_excluded_points": {str(t): int(((Ved[idx, j] - V0d[idx, j]) < 0.01).sum()) for j, t in enumerate(CKPT["DRIFT"])},
                           "fraction": {m: {"mean": fr[i].tolist(), "ci": q(np.array([b[0][i] for b in bs]))} for i, m in enumerate(msd)},
                           "regret": {m: {"mean": rg[i].tolist(), "ci": q(np.array([b[1][i] for b in bs]))} for i, m in enumerate(msd)},
                           "mean_fraction_t100_500": {m: float(fr[i, 2:].mean()) for i, m in enumerate(msd)}}
    res["DRIFT"]["harm_rate"] = {m: (Ud[i] < Ud[0] - 1e-9).mean(0).tolist() for i, m in enumerate(msd)}
    # ---------------- audits
    aud = {}
    for kind in ("SWITCH", "DRIFT"):
        for i, m in enumerate(S[f"{kind}_methods"]):
            ex = S[f"{kind}_expl"][i] - EPS; ok = S[f"{kind}_ok"][i]
            a = aud.setdefault(str(m), {"n": 0, "max_expl_minus_eps": -1.0, "n_viol": 0, "lp_failures": 0, "min_expl": 1.0})
            a["n"] += int(ex.size); a["max_expl_minus_eps"] = max(a["max_expl_minus_eps"], float(ex.max())); a["n_viol"] += int((ex > 1e-7).sum())
            a["lp_failures"] += int((~ok).sum()); a["min_expl"] = min(a["min_expl"], float((ex + EPS).min()))
    res["audit"] = aud; res["audit_total"] = {"n": sum(a["n"] for a in aud.values()), "max_expl_minus_eps": max(a["max_expl_minus_eps"] for a in aud.values()),
                                              "violations": sum(a["n_viol"] for a in aud.values()), "lp_failures": sum(a["lp_failures"] for a in aud.values())}
    # ---------------- pre-registered verdicts (contrasting pairs)
    C = res["SWITCH"]["contrasting"]; paired, rel = helpers["contrasting"]; Ts = CKPT["SWITCH"]; j300 = Ts.index(300)
    v = {}
    s1 = {m: {"frac_300": C["fraction"][m]["mean"][j300], "pre": C["pre"][m], "ratio": C["fraction"][m]["mean"][j300] / C["pre"][m]} for m in ("TAB-EM", "PRIOR-EM")}
    v["S1"] = {"holds": bool(all(x["ratio"] < 0.8 for x in s1.values())), "detail": s1}
    s2 = {}
    for f, full in (("WIN-EM W=50", "TAB-EM"), ("DISC-EM g=0.98", "TAB-EM"), ("PRIOR-EM-WIN", "PRIOR-EM"), *[(gm, "TAB-EM") for gm in GRID if gm not in ("WIN-EM W=50", "DISC-EM g=0.98")]):
        pp, pr = paired(f, full, "post"), paired(f, full, "pre")
        s2[f] = {"vs": full, "post_mean_diff": pp, "pre_diff": pr, "faster": bool(pp["ci"][0] > 0), "stationary_cost": bool(pr["ci"][1] < 0)}
    v["S2"] = {"holds": bool(all(s2[m]["faster"] and s2[m]["stationary_cost"] for m in ("WIN-EM W=50", "DISC-EM g=0.98", "PRIOR-EM-WIN"))), "detail": s2}
    s3 = {f"{n} vs {e}": rel(n, e) for n in ("JAC-opp", "DEC-889k") for e in ("TAB-EM", "PRIOR-EM")}
    v["S3"] = {"holds": bool(all(x["ci"][0] <= 0 for x in s3.values())), "relative_recovery_diff": s3,
               "relative_recovery": {m: C["relative_recovery"][m] for m in ("TAB-EM", "PRIOR-EM", "JAC-opp", "DEC-889k")}}
    nonor = [m for m in DEFAULT if m != "FIXED-NE"]; ranking = sorted(nonor, key=lambda m: -C["post_all_mean"][m])
    v["S4"] = {"holds": bool(ranking[0] == "PRIOR-EM-WIN"), "ranking_post_all_mean": [(m, C["post_all_mean"][m]) for m in ranking],
               "best_minus_runner_up": paired(ranking[0], ranking[1], "postall"),
               "grid_post_all_mean": {m: C["post_all_mean"][m] for m in GRID}}
    v["S5"] = {"holds": bool(res["audit_total"]["violations"] == 0), **res["audit_total"]}
    R8 = {m: C["recovery"]["0.8"][m]["hands"] for m in nonor}; fastest = min(R8, key=lambda m: R8[m])
    oracle = C["recovery"]["0.8"]["POST-PRIOR-EM"]["hands"]; ratio = R8[fastest] / oracle if np.isfinite(R8[fastest]) and oracle > 0 else float("inf")
    gridR = {m: C["recovery"]["0.8"][m]["hands"] for m in GRID}; hind = min(gridR, key=lambda m: gridR[m])
    v["DECISION"] = {"fastest_nonoracle_predeclared": fastest, "R08_fastest": R8[fastest], "R08_POST_PRIOR_EM": oracle, "R08_POST_EM": C["recovery"]["0.8"]["POST-EM"]["hands"],
                     "ratio": ratio, "R08_all": R8, "hindsight_tuned_grid_point": hind, "R08_hindsight": gridR[hind],
                     "verdict": ("forgetting suffices: case for training sequence models on switching opponents is weak" if ratio <= 1.5
                                 else "gap for learned change detection: recommend the training experiment")}
    res["verdicts"] = v
    save_json(res, D / "analysis.json")
    print(json.dumps({k: (x.get("holds", x.get("verdict"))) for k, x in v.items()}, indent=1))
    print(json.dumps(v["DECISION"], indent=1))


if __name__ == "__main__":
    main()
