"""Generalization study analysis: fraction of attainable safe gain, raw regret, NE loss, harm rate, audits,
hybrid-vs-tabular-EM differences and the pre-registered E1-E5 checks.  Writes outputs/gen/analysis.json."""
import json
import numpy as np
from .common import OUT, N_BUDGETS, save_json
from .gen_families import FAMS

D = OUT / "gen"; EPS = 0.10; B = 2000; NS = list(N_BUDGETS)
NEURAL = ["DEC-889k", "RECON-889k", "RECON-JAC-global", "JAC-opp"]
HYB = ["PRIOR-EM (RECON-131k)", "PRIOR-EM (JAC-opp)"]
FAR = ["FAR-ARCH", "FAR-CFR", "FAR-EXPL"]


def load():
    F = np.load(D / "families.npz", allow_pickle=True); S = np.load(D / "solve.npz", allow_pickle=True)
    methods = ["FIXED-NE"] + [str(m) for m in S["methods"]]
    H = F["hist_opp"].shape[0]; nN = len(NS)
    U = np.concatenate([np.repeat(S["u_fixed_ne"][None, :, None], nN, 2), S["u"]], 0)          # (M, H, nN)
    X = np.concatenate([np.full((1, H, nN), float(S["expl_fixed_ne"])), S["expl"]], 0)
    OK = np.concatenate([np.ones((1, H, nN), bool), S["ok"]], 0)
    n_opp = len(F["family"]); cnt = np.bincount(F["hist_opp"], minlength=n_opp)
    Uo = np.stack([np.stack([np.bincount(F["hist_opp"], weights=U[m, :, j], minlength=n_opp) / cnt for j in range(nN)], 1) for m in range(len(methods))])
    return F, S, methods, U, X, OK, Uo


def boot_mean(a, rng):
    """a: (n, ...) -> mean, lo, hi over resamples of axis 0."""
    idx = rng.integers(0, len(a), (B, len(a))); bs = a[idx].mean(1)
    return a.mean(0), np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)


def ratio_boot(num, den, rng):
    idx = rng.integers(0, len(num), (B, len(num))); r = num[idx].mean(1) / den[idx].mean(1)[:, None] if num.ndim == 2 else None
    return num.mean(0) / den.mean(), np.percentile(r, 2.5, 0), np.percentile(r, 97.5, 0)


def main():
    F, S, methods, U, X, OK, Uo = load(); rng = np.random.default_rng(0)
    fam = F["family"]; V0, Ve = F["V0"], F["Veps"]; opp_expl = F["opp_expl"]; M = len(methods); nN = len(NS)
    v_star = None
    from .game.safe_lp import get_solver
    v_star = get_solver().v_star
    res = {"methods": methods, "families": FAMS, "N": NS, "per_family": {}}
    for f in FAMS:
        o = np.flatnonzero(fam == f); e = {"n_opponents": int(len(o))}
        R = Ve[o][None, :, None] - Uo[:, o]                                          # (M, n, nN) regret
        e["regret"] = {methods[m]: dict(zip(("mean", "lo", "hi"), [v.tolist() for v in boot_mean(R[m], rng)])) for m in range(M)}
        if f != "NE":
            keep = (Ve[o] - V0[o]) >= 0.01; oi = o[keep]; e["n_excluded_headroom"] = int((~keep).sum())
            den = Ve[oi] - V0[oi]; e["fraction"] = {}
            for m in range(M):
                num = Uo[m, oi] - V0[oi][:, None]
                mean, lo, hi = ratio_boot(num, den, rng)
                e["fraction"][methods[m]] = {"ratio_of_means": mean.tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
                                            "median_per_opponent": np.median(num / den[:, None], 0).tolist()}
        else:
            h = np.flatnonzero(np.isin(F["hist_opp"], o)); loss = v_star - U[:, h]                     # (M, 100, nN)
            e["ne_loss"] = {methods[m]: {**dict(zip(("mean", "lo", "hi"), [v.tolist() for v in boot_mean(loss[m], rng)])),
                                         "min": loss[m].min(0).tolist(), "max": loss[m].max(0).tolist()} for m in range(M)}
            tol = opp_expl[F["hist_opp"][h]]                                                         # allowed negative loss per history
            e["ne_loss_in_bounds"] = {methods[m]: bool(np.all(loss[m] >= -tol[:, None] - 1e-9) and np.all(loss[m] <= EPS + 1e-7)) for m in range(M)}
            e["ne_opp_exploitability"] = opp_expl[o].tolist()
        harm = Uo[:, o] < (Uo[0, o] - 1e-9)[None]
        e["harm_rate"] = {methods[m]: harm[m].mean(0).tolist() for m in range(M)}
        res["per_family"][f] = e
    # pooled harm rate over the 5 non-NE families (opponent-weighted)
    o = np.flatnonzero(fam != "NE"); harm = Uo[:, o] < (Uo[0, o] - 1e-9)[None]
    res["harm_rate_pooled_nonNE"] = {methods[m]: harm[m].mean(0).tolist() for m in range(M)}
    # hybrid - tabular EM regret (paired over opponents; NE: loss, paired over histories)
    res["hybrid_minus_tabem"] = {}
    ti = methods.index("TAB-EM")
    for hname in HYB:
        hi_ = methods.index(hname); res["hybrid_minus_tabem"][hname] = {}
        for f in FAMS:
            o = np.flatnonzero(fam == f)
            d = -(Uo[hi_, o] - Uo[ti, o]) if f != "NE" else -(U[hi_][np.isin(F["hist_opp"], o)] - U[ti][np.isin(F["hist_opp"], o)])
            mean, lo, hi = boot_mean(d, rng)
            cross = None
            for j in range(nN):
                if np.all(mean[j:] <= 0):
                    cross = NS[j]; break
            res["hybrid_minus_tabem"][hname][f] = {"mean": mean.tolist(), "lo": lo.tolist(), "hi": hi.tolist(), "stays_nonpositive_from_N": cross}
    # audits
    res["audit"] = {}
    for m in range(M):
        ex = X[m] - EPS
        res["audit"][methods[m]] = {"n": int(ex.size) if m else 1, "max_expl_minus_eps": float(ex.max()), "n_viol": int((ex > 1e-7).sum()),
                                   "lp_failures": int((~OK[m]).sum()), "median_expl": float(np.median(X[m])), "p99_expl": float(np.percentile(X[m], 99))}
    res["audit_total"] = {"n_deployed": int(sum(v["n"] for v in res["audit"].values())), "max_expl_minus_eps": max(v["max_expl_minus_eps"] for v in res["audit"].values()),
                          "violations": sum(v["n_viol"] for v in res["audit"].values()), "lp_failures": sum(v["lp_failures"] for v in res["audit"].values())}
    # ---------------- pre-registered checks
    j20, j500 = NS.index(20), NS.index(500)
    frac = lambda m, f, j: res["per_family"][f]["fraction"][m]["ratio_of_means"][j]
    mods = [m for m in methods if m != "FIXED-NE"]
    drop = {m: {str(N): frac(m, "ID-REF", j) - np.mean([frac(m, f, j) for f in FAR]) for N, j in ((20, j20), (500, j500))} for m in mods}
    e1 = drop["BANK"]["500"] == max(drop[m]["500"] for m in mods) and drop["BANK"]["500"] > drop["BANK"]["20"]
    v = {"E1": {"holds": bool(e1), "drop_IDREF_minus_FARmean": drop}}
    e2 = {}
    for m in NEURAL:
        ok_frac = all(frac(m, "ID-REF", j) > frac(m, "NEAR", j) > np.mean([frac(m, f, j) for f in FAR]) for j in (j20, j500))
        reg = lambda f: res["per_family"][f]["regret"][m]["mean"][j500]
        ok_floor = all(reg(f) > reg("ID-REF") for f in FAR)
        e2[m] = {"holds": bool(ok_frac and ok_floor), "fraction_order": bool(ok_frac), "floor_rises_on_all_FAR": bool(ok_floor),
                 "regret_N500": {f: reg(f) for f in FAMS}}
    v["E2"] = {"holds": bool(sum(x["holds"] for x in e2.values()) >= 3), "per_model": e2}
    e3 = {}
    for hname in HYB:
        per = {}
        for f in FAMS:
            d = res["hybrid_minus_tabem"][hname][f]
            close = abs(d["mean"][j500]) < 0.005 or (d["lo"][j500] <= 0 <= d["hi"][j500])
            shrink = abs(d["mean"][j500]) <= abs(d["mean"][j20])
            per[f] = {"close_at_500": bool(close), "shrinks_20_to_500": bool(shrink), "diff_500": d["mean"][j500], "diff_20": d["mean"][j20]}
        e3[hname] = {"holds": bool(all(x["close_at_500"] and x["shrinks_20_to_500"] for x in per.values())), "per_family": per}
    v["E3"] = e3
    e4 = {}
    for hname in HYB:
        e4[hname] = {f: {"hurts_at_N<=20": [NS[j] for j in range(NS.index(20) + 1) if res["hybrid_minus_tabem"][hname][f]["lo"][j] > 0],
                         "stays_nonpositive_from_N": res["hybrid_minus_tabem"][hname][f]["stays_nonpositive_from_N"]} for f in FAMS}
    v["E4"] = {"holds": bool(any(len(e4[h][f]["hurts_at_N<=20"]) > 0 for h in HYB for f in FAR)), "per_hybrid": e4}
    ne = res["per_family"]["NE"]
    e5_bounds = all(ne["ne_loss_in_bounds"].values())
    e5_mean = all(max(ne["ne_loss"][m]["mean"]) <= EPS / 2 for m in methods)
    v["E5"] = {"holds": bool(e5_bounds and e5_mean), "all_losses_in_bounds": bool(e5_bounds), "mean_loss_le_eps_over_2": bool(e5_mean),
               "max_mean_loss": {m: max(ne["ne_loss"][m]["mean"]) for m in methods}, "max_loss": {m: max(ne["ne_loss"][m]["max"]) for m in methods}}
    res["verdicts"] = v
    save_json(res, D / "analysis.json")
    print(json.dumps({k: (x["holds"] if isinstance(x, dict) and "holds" in x else {h: y["holds"] for h, y in x.items()}) for k, x in v.items()}, indent=1))
    print(json.dumps(res["audit_total"]))


if __name__ == "__main__":
    main()
