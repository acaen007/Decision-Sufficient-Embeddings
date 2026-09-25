"""Analysis of the change-detection study (REPORT_LEDUC_CHANGEDETECT.md section 0).  Writes outputs/cpd/analysis.json."""
import json
import numpy as np
from .common import OUT, save_json
from .ns_eval import SW_T, DR_T, SWITCH_AT, EPS
from .cpd_eval import D, GRID, CKPTS, SW_FINE, cfg_name, cur_values

B = 2000
MAIN = ["PRIOR-EM", "PRIOR-EM-WIN", "DISC-PRIOR-EM g=0.98", "DISC-PRIOR-EM g=0.95", "CPD-PRIOR-EM", "KNOWN-CUSUM-RESET", "POST-PRIOR-EM"]
POST_T = [205, 210, 220, 250, 300]; POSTALL_T = [205, 210, 220, 250, 300, 400, 500]


def recovery_time(fr, pre, level, Ts):
    target = level * pre; post = [(t - SWITCH_AT, fr[i]) for i, t in enumerate(Ts) if t > SWITCH_AT]
    if post[0][1] >= target:
        return float(post[0][0])
    for (h0, f0), (h1, f1) in zip(post[:-1], post[1:]):
        if f1 >= target:
            return float(h0 + (target - f0) / (f1 - f0) * (h1 - h0))
    return float("inf")


def pooled(U, V0, Ve, idx):
    """U (M, P, nT) -> pooled fraction (M, nT) over processes idx (headroom exclusion < 0.01)."""
    hd = Ve[idx] - V0[idx]; keep = hd >= 0.01
    return np.where(keep[None], U[:, idx] - V0[idx][None], 0).sum(1) / np.maximum(np.where(keep, hd, 0).sum(0), 1e-12)


def stats(U, V0, Ve, idx, Ts):
    fr = pooled(U, V0, Ve, idx); j200 = Ts.index(200)
    jp = [Ts.index(t) for t in POST_T]; ja = [Ts.index(t) for t in POSTALL_T]
    pre = fr[:, j200]; rec = {p: np.array([recovery_time(fr[m], pre[m], p, Ts) for m in range(len(fr))]) for p in (0.5, 0.8)}
    return {"fr": fr, "pre": pre, "post": fr[:, jp].mean(1), "postall": fr[:, ja].mean(1), "R50": rec[0.5], "R80": rec[0.8]}


def ci(x):
    x = np.asarray(x, float); xs = np.where(np.isfinite(x), x, 1e9)
    f = lambda v: float("inf") if v >= 1e9 else float(v)
    return [f(np.percentile(xs, 2.5)), f(np.percentile(xs, 97.5))]


def summarize(ms, U, V0, Ve, idx, Ts, rng):
    st = stats(U, V0, Ve, idx, Ts); bs = [stats(U, V0, Ve, rng.choice(idx, len(idx), replace=True), Ts) for _ in range(B)]
    out = {"n_processes": int(len(idx)), "checkpoints": Ts, "methods": {}}
    for i, m in enumerate(ms):
        out["methods"][m] = {"fraction": st["fr"][i].tolist(), "fraction_ci": [np.percentile([b["fr"][i] for b in bs], q, 0).tolist() for q in (2.5, 97.5)],
                             **{k: float(st[k][i]) for k in ("pre", "post", "postall", "R50", "R80")},
                             **{f"{k}_ci": ci([b[k][i] for b in bs]) for k in ("pre", "post", "postall", "R50", "R80")},
                             "p_not_recovered_R80": float(np.mean([not np.isfinite(b["R80"][i]) for b in bs]))}
    ix = {m: i for i, m in enumerate(ms)}
    def paired(a, b, k):
        d = [bb[k][ix[a]] - bb[k][ix[b]] for bb in bs]
        return {"diff": float(st[k][ix[a]] - st[k][ix[b]]), "ci": ci(d)}
    def ratio(a, b):
        r = [bb["R80"][ix[a]] / bb["R80"][ix[b]] if np.isfinite(bb["R80"][ix[a]]) else np.inf for bb in bs]
        return {"ratio": float(st["R80"][ix[a]] / st["R80"][ix[b]]), "ci": ci(r)}
    return out, paired, ratio


def detection(alarms, n, pre_hands_only):
    """Per-history detection statistics.  pre_hands_only: stationary histories (every alarm is false)."""
    if pre_hands_only:
        return {"false_alarms": int(len(alarms)), "per_1000_hands": 1000 * len(alarms) / (n * 500), "histories_with_alarm": int(len(np.unique(alarms[:, 1])))}
    pre = alarms[alarms[:, 2] < SWITCH_AT]; post = alarms[alarms[:, 2] >= SWITCH_AT]; delay = np.full(n, np.inf); cp = np.full(n, np.nan)
    for h in range(n):
        a = post[post[:, 1] == h]
        if len(a):
            a = a[np.argsort(a[:, 2])][0]; delay[h] = a[2] + 1 - SWITCH_AT; cp[h] = a[3] - SWITCH_AT
    return {"delay": delay, "cp_err": cp, "pre_false_alarms": int(len(pre)), "pre_per_1000_hands": 1000 * len(pre) / (n * SWITCH_AT),
            "post_alarms_total": int(len(post))}


def dsum(d, sel_h):
    x = d["delay"][sel_h]; c = d["cp_err"][sel_h]; fin = np.isfinite(x)
    return {"n_histories": int(len(x)), "median_delay": float(np.median(x)), "p25": float(np.percentile(x, 25)), "p75": float(np.percentile(x, 75)),
            "within_10": float(np.mean(x <= 10)), "within_25": float(np.mean(x <= 25)), "within_50": float(np.mean(x <= 50)), "within_100": float(np.mean(x <= 100)),
            "undetected": float(np.mean(~fin)), "median_cp_err": float(np.nanmedian(c)) if fin.any() else None,
            "median_abs_cp_err": float(np.nanmedian(np.abs(c))) if fin.any() else None, "frac_cp_before_switch": float(np.mean(c[fin] < 0)) if fin.any() else None}


def main():
    rng = np.random.default_rng(0); P = np.load(D / "processes.npz", allow_pickle=True); T = np.load(D / "test.npz", allow_pickle=True)
    meta = json.loads((D / "test_meta.json").read_text()); cal = json.loads((D / "calib.json").read_text()); res = {"meta": meta, "calibration": cal}
    sel = meta["selected_name"]; grid = [cfg_name(c) for c in GRID]
    def U_of(s, ms):
        return np.stack([T[f"{s}::{m}::u"].reshape(-1, 2, len(CKPTS[s])).mean(1) for m in ms])
    # ---------------- SWITCH
    _, V0f, Vef = cur_values(P, "SW", SW_FINE); jo = [SW_FINE.index(t) for t in SW_T]; group = P["SW_sel"]
    Uf = U_of("SW", MAIN); Ug = U_of("SW", grid)
    res["SWITCH"] = {}; helpers = {}
    for g in ("contrasting", "random"):
        idx = np.flatnonzero(group == g); res["SWITCH"][g] = {}
        for gname, Ts, U, V0, Ve in (("original_grid", SW_T, Uf[:, :, jo], V0f[:, jo], Vef[:, jo]), ("fine_grid", SW_FINE, Uf, V0f, Vef)):
            out, paired, ratio = summarize(MAIN, U, V0, Ve, idx, Ts, rng); res["SWITCH"][g][gname] = out; helpers[(g, gname)] = (paired, ratio)
        # hindsight grid (point estimates, original grid)
        stg = stats(Ug[:, :, jo], V0f[:, jo], Vef[:, jo], idx, SW_T); ro = res["SWITCH"][g]["original_grid"]["methods"]["POST-PRIOR-EM"]["R80"]
        res["SWITCH"][g]["hindsight_grid"] = {m: {"pre": float(stg["pre"][i]), "post": float(stg["post"][i]), "postall": float(stg["postall"][i]),
                                                  "R80": float(stg["R80"][i]), "ratio": float(stg["R80"][i] / ro)} for i, m in enumerate(grid)}
    # ---------------- TEST-STAT and DRIFT
    _, V0s, Ves = cur_values(P, "ST", CKPTS["ST"]); ms_st = [m for m in MAIN if f"ST::{m}::u" in T.files]
    Us = U_of("ST", ms_st); Usg = U_of("ST", grid); idx = np.arange(Us.shape[1])
    def fstat(U, ii):
        return pooled(U, V0s, Ves, ii).mean(1)
    fs = fstat(Us, idx); bs = [fstat(Us, rng.choice(idx, len(idx), replace=True)) for _ in range(B)]; ix = {m: i for i, m in enumerate(ms_st)}
    res["STAT"] = {"fraction_by_ckpt": {m: pooled(Us, V0s, Ves, idx)[i].tolist() for i, m in enumerate(ms_st)},
                   "mean_fraction": {m: float(fs[i]) for i, m in enumerate(ms_st)}, "mean_fraction_ci": {m: ci([b[i] for b in bs]) for i, m in enumerate(ms_st)},
                   "cost_vs_PRIOR-EM": {m: {"diff": float(fs[ix["PRIOR-EM"]] - fs[i]), "ci": ci([b[ix["PRIOR-EM"]] - b[i] for b in bs])} for i, m in enumerate(ms_st)}}
    fg = fstat(Usg, idx); res["STAT"]["hindsight_grid_cost"] = {m: float(fs[ix["PRIOR-EM"]] - fg[i]) for i, m in enumerate(grid)}
    _, V0d, Ved = cur_values(P, "DR", CKPTS["DR"]); ms_dr = [m for m in MAIN if f"DR::{m}::u" in T.files]; Ud = U_of("DR", ms_dr); idd = np.arange(Ud.shape[1])
    def fdr(U, ii):
        return pooled(U, V0d, Ved, ii)
    frd = fdr(Ud, idd); bsd = [fdr(Ud, rng.choice(idd, len(idd), replace=True)) for _ in range(B)]; ixd = {m: i for i, m in enumerate(ms_dr)}
    res["DRIFT"] = {"fraction": {m: frd[i].tolist() for i, m in enumerate(ms_dr)},
                    "fraction_ci": {m: [np.percentile([b[i] for b in bsd], q, 0).tolist() for q in (2.5, 97.5)] for i, m in enumerate(ms_dr)},
                    "mean_t100_500": {m: float(frd[i, 2:].mean()) for i, m in enumerate(ms_dr)},
                    "CPD_minus_WIN_t100_500": {"diff": float(frd[ixd["CPD-PRIOR-EM"], 2:].mean() - frd[ixd["PRIOR-EM-WIN"], 2:].mean()),
                                               "ci": ci([b[ixd["CPD-PRIOR-EM"], 2:].mean() - b[ixd["PRIOR-EM-WIN"], 2:].mean() for b in bsd])}}
    # ---------------- detection statistics
    al = np.load(D / "test_alarms.npy"); nsw = 2 * len(group); si = [cfg_name(c) for c in GRID].index(sel)
    a_sw = al[(al[:, 0] == si) & (al[:, 1] < nsw)]; a_st = al[(al[:, 0] == si) & (al[:, 1] >= nsw)]
    ak = np.load(D / "test_alarms_known.npy"); adr = np.load(D / "test_alarms_drift.npy")
    hsel = np.repeat(group, 2); det = {}
    for name, a in (("CPD", a_sw), ("KNOWN-CUSUM", ak)):
        d = detection(a, nsw, False); det[name] = {"pre_false_alarms": d["pre_false_alarms"], "pre_per_1000_hands": d["pre_per_1000_hands"],
                                                   **{g: dsum(d, hsel == g) for g in ("contrasting", "random")}}
        det[name]["_delay"] = d["delay"].tolist()
    det["CPD"]["TEST-STAT"] = detection(a_st, 2 * 40, True); det["CPD"]["DRIFT_alarms_per_history"] = float(len(adr) / (2 * 40))
    det["CPD"]["DRIFT_alarm_hist"] = np.histogram(adr[:, 2], bins=[0, 100, 200, 300, 400, 500])[0].tolist()
    res["detection"] = det
    # hindsight: false alarms per 1000 stationary hands for each grid configuration
    res["hindsight_false_alarms_per_1000"] = {m: 1000 * float(((al[:, 0] == i) & (al[:, 1] >= nsw)).sum()) / (80 * 500) for i, m in enumerate(grid)}
    # ---------------- information floor
    if (D / "floor.npz").exists():
        F = np.load(D / "floor.npz"); kl = F["KL_BA"]; fl = {"check_max_abs": float(F["check_max_abs"])}
        for g in ("contrasting", "random"):
            m = group == g
            fl[g] = {"median_KL_BA": float(np.median(kl[m])), "p10_KL_BA": float(np.percentile(kl[m], 10)), "p90_KL_BA": float(np.percentile(kl[m], 90)),
                     "median_KL_AB": float(np.median(F["KL_AB"][m])), "median_lorden_delay_log1000": float(np.median(np.log(1000) / kl[m])),
                     "mc_llr_vs_KL_median_rel_diff": float(np.median(np.abs(F["mc_llr_post"][m] - kl[m]) / kl[m]))}
        fl["KL_BA"] = kl.tolist(); res["floor"] = fl
    # ---------------- audit (every unique deployed test strategy)
    ex = T["expl_all"] - EPS; ok = T["ok_all"]
    res["audit"] = {"n": int(len(ex)), "max_expl_minus_eps": float(np.nanmax(ex)), "violations": int((ex > 1e-7).sum()), "lp_failures": int((~ok).sum()),
                    "n_method_level": int(sum(np.isfinite(T[k]).sum() for k in T.files if k.endswith("::u")))}
    # ---------------- expectations and decision
    C = res["SWITCH"]["contrasting"]["original_grid"]["methods"]; paired, ratio = helpers[("contrasting", "original_grid")]
    v = {}
    v["C1"] = {"holds": bool(C["CPD-PRIOR-EM"]["R80"] < C["PRIOR-EM-WIN"]["R80"] and paired("CPD-PRIOR-EM", "PRIOR-EM-WIN", "post")["ci"][0] > 0),
               "R80_CPD": C["CPD-PRIOR-EM"]["R80"], "R80_WIN": C["PRIOR-EM-WIN"]["R80"], "post_diff": paired("CPD-PRIOR-EM", "PRIOR-EM-WIN", "post")}
    pw = paired("CPD-PRIOR-EM", "PRIOR-EM-WIN", "pre"); pf = paired("CPD-PRIOR-EM", "PRIOR-EM", "pre")
    v["C2"] = {"holds": bool(pf["diff"] >= -0.02 and pw["ci"][0] > 0), "pre_minus_full": pf, "pre_minus_win": pw}
    sc = res["STAT"]["cost_vs_PRIOR-EM"]["CPD-PRIOR-EM"]
    v["C3"] = {"holds": bool(abs(sc["diff"]) <= 0.02), "stationary_cost": sc}
    v["C4"] = {"holds": bool(res["DRIFT"]["CPD_minus_WIN_t100_500"]["diff"] < 0), **res["DRIFT"]["CPD_minus_WIN_t100_500"]}
    v["C5"] = {"holds": bool(res["audit"]["violations"] == 0), **res["audit"]}
    r = ratio("CPD-PRIOR-EM", "POST-PRIOR-EM"); rk = ratio("KNOWN-CUSUM-RESET", "POST-PRIOR-EM"); rw = ratio("PRIOR-EM-WIN", "POST-PRIOR-EM")
    band = "no-go" if r["ratio"] <= 1.5 else ("go" if (r["ratio"] > 2.0 or not np.isfinite(r["ratio"])) else "inconclusive")
    v["DECISION"] = {"R80_CPD": C["CPD-PRIOR-EM"]["R80"], "R80_CPD_ci": C["CPD-PRIOR-EM"]["R80_ci"], "R80_oracle": C["POST-PRIOR-EM"]["R80"],
                     "R80_oracle_ci": C["POST-PRIOR-EM"]["R80_ci"], "ratio": r, "band": band, "qualified_by_stationary_cost": bool(sc["diff"] > 0.03),
                     "R80_known": C["KNOWN-CUSUM-RESET"]["R80"], "ratio_known": rk, "ratio_win": rw,
                     "R80_diff_CPD_minus_oracle": paired("CPD-PRIOR-EM", "POST-PRIOR-EM", "R80"),
                     "post_diff_CPD_minus_oracle": paired("CPD-PRIOR-EM", "POST-PRIOR-EM", "post"),
                     "postall_diff_CPD_minus_oracle": paired("CPD-PRIOR-EM", "POST-PRIOR-EM", "postall"),
                     "post_diff_KNOWN_minus_oracle": paired("KNOWN-CUSUM-RESET", "POST-PRIOR-EM", "post"),
                     "post_diff_CPD_minus_KNOWN": paired("CPD-PRIOR-EM", "KNOWN-CUSUM-RESET", "post"),
                     "fine_grid": {m: res["SWITCH"]["contrasting"]["fine_grid"]["methods"][m]["R80"] for m in MAIN}}
    pr, rr = helpers[("contrasting", "fine_grid")]; v["DECISION"]["fine_grid_ratio"] = rr("CPD-PRIOR-EM", "POST-PRIOR-EM")
    prr, rrr = helpers[("random", "original_grid")]; v["DECISION"]["random_pairs_ratio"] = rrr("CPD-PRIOR-EM", "POST-PRIOR-EM")
    res["verdicts"] = v
    save_json(res, D / "analysis.json")
    print(json.dumps({k: x.get("holds", x.get("band")) for k, x in v.items()}, indent=1))
    print(json.dumps({k: v["DECISION"][k] for k in ("R80_CPD", "R80_CPD_ci", "R80_oracle", "ratio", "band", "R80_known", "ratio_known", "fine_grid", "fine_grid_ratio", "random_pairs_ratio")}, indent=1, default=float))


if __name__ == "__main__":
    main()
