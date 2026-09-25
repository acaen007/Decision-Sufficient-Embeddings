"""Analysis of the decision-compression study (REPORT_LEDUC_DECISION_COMPRESSION.md): Part A verdicts from oracle.json,
Part B arm x d tables and H1-H4, Part C decision-equivalence structure.  Writes outputs/dcomp/analysis.json."""
import json
import numpy as np
from scipy.stats import spearmanr, rankdata
from .common import save_json
from .dc_common import D, EPS_LIST, OOD_FAMS, load_sets, HandDist, js_matrix, fraction
from .dc_ae import ARMS, POSTHOC_ARMS, DIMS, SEEDS, EVALS, run_name

B = 2000; KS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def part_a(orc):
    P = orc["partitions"]; v = {}
    f = lambda t, K, s="test", k="frac_0.1": P[f"{t}_{K}"][s][k]
    v["A1"] = {"holds": all(f("DEC", K) > max(f("BEH", K), f("GVAR", K)) for K in KS if K >= 4),
               "margins_vs_best_other": {K: f("DEC", K) - max(f("BEH", K), f("GVAR", K)) for K in KS}}
    k90 = {t: next((K for K in KS if f(t, K) >= 0.90), None) for t in ("DEC", "BEH", "GVAR")}
    v["A2"] = {"holds": bool(k90["DEC"] is not None and k90["DEC"] <= 32 and (k90["BEH"] is None or k90["BEH"] >= 4 * k90["DEC"])), "K90": k90,
               "all_1nn": {t: f(t, "all") for t in ("DEC", "BEH", "GVAR")}}
    if k90["DEC"] is not None:
        K = k90["DEC"]; v["A3"] = {"holds": bool(f("BEH", K, k="beh_fraction") - f("DEC", K, k="beh_fraction") >= 0.10), "K": K,
                                   "beh_BEH": f("BEH", K, k="beh_fraction"), "beh_DEC": f("DEC", K, k="beh_fraction")}
    else:
        v["A3"] = {"holds": None, "note": "K90(DEC) not reached on the grid"}
    adv_t = f("DEC", 16) - f("BEH", 16); adv_o = f("DEC", 16, "ood") - f("BEH", 16, "ood")
    v["A4"] = {"holds": bool(adv_o < adv_t), "adv_test": adv_t, "adv_ood": adv_o}
    return v


def load_eval(name):
    p = EVALS / f"{name}.npz"
    return dict(np.load(p)) if p.exists() else None


def main():
    S, pop = load_sets(); res = {}
    orc = json.loads((D / "oracle.json").read_text()); res["partA"] = part_a(orc)
    # ---------------- Part B: per (arm, d)
    tab = {}; rng = np.random.default_rng(0)
    for arm in ARMS + POSTHOC_ARMS:
        for d in DIMS:
            ev = [load_eval(run_name(arm, d, s)) for s in SEEDS]; ev = [e for e in ev if e is not None]
            if not ev:
                continue
            row = {"n_seeds": len(ev)}
            for sname, eps_list in (("test", EPS_LIST), ("val", [0.10]), ("ood", [0.10])):
                s = S[sname]
                for e in eps_list:
                    per = [fraction(x[f"{sname}_u_{e}"], s["V0"], s["V"][e]) for x in ev]; row[f"{sname}_frac_{e}"] = float(np.mean(per)); row[f"{sname}_frac_{e}_seeds"] = per
                    if sname == "test":
                        u = np.mean([x[f"{sname}_u_{e}"] for x in ev], 0); hd_ = s["V"][e] - s["V0"]; keep = np.flatnonzero(hd_ >= 0.01)
                        ii = rng.integers(0, len(keep), (B, len(keep))); bs = (u - s["V0"])[keep][ii].sum(1) / hd_[keep][ii].sum(1)
                        row[f"test_frac_{e}_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
                    row[f"{sname}_viol_{e}"] = int(sum(((x[f"{sname}_expl_{e}"] - e) > 1e-7).sum() for x in ev)); row[f"{sname}_maxexcess_{e}"] = float(max((x[f"{sname}_expl_{e}"] - e).max() for x in ev))
                    row[f"{sname}_lpfail_{e}"] = int(sum((~x[f"{sname}_ok_{e}"].astype(bool)).sum() for x in ev))
                row[f"{sname}_beh"] = float(np.mean([1 - x[f"{sname}_kl_beh"].sum() / x[f"{sname}_kl_beh0"].sum() for x in ev]))
                row[f"{sname}_pol"] = float(np.mean([1 - x[f"{sname}_kl_pol"].sum() / x[f"{sname}_kl_pol0"].sum() for x in ev]))
                row[f"{sname}_gR2"] = float(np.mean([1 - x[f"{sname}_gerr"].sum() / x[f"{sname}_gvar"].sum() for x in ev]))
                if sname == "ood":
                    for fam in OOD_FAMS:
                        m = s["family"] == fam
                        row[f"ood_frac_{fam}"] = float(np.mean([fraction(x["ood_u_0.1"][m], s["V0"][m], s["V"][0.10][m]) for x in ev]))
                        row[f"ood_beh_{fam}"] = float(np.mean([1 - x["ood_kl_beh"][m].sum() / x["ood_kl_beh0"][m].sum() for x in ev]))
            tab[f"{arm}_d{d}"] = row
    res["partB"] = tab
    T = lambda arm, d, k="test_frac_0.1": tab[f"{arm}_d{d}"][k]
    dec_arms = ["G-MSE", "REGRET", "JAC-OPP"]; v = {}
    if all(f"{a}_d{d}" in tab for a in ARMS for d in DIMS):
        h1 = []
        for d in [x for x in DIMS if x <= 8]:
            best = max(dec_arms, key=lambda a: T(a, d)); val = T(best, d); dr = next((x for x in DIMS if T("RECON", x) >= val), None)
            ok = val >= 0.90 and T("RECON", d) <= 0.80 and (dr is None or dr >= 4 * d)
            h1.append({"d": d, "best_arm": best, "best_frac": val, "recon_frac": T("RECON", d), "recon_d_to_match": dr, "holds": bool(ok),
                       "beh_best": T(best, d, "test_beh"), "beh_recon": T("RECON", d, "test_beh")})
        v["H1"] = {"holds": any(x["holds"] for x in h1), "detail": h1}
        hd = [x for x in h1 if x["holds"]]
        v["H2"] = {"holds": bool(hd and hd[0]["beh_recon"] - hd[0]["beh_best"] >= 0.10), "detail": hd[0] if hd else None}
        spread = {d: max(T(a, d) for a in ARMS) - min(T(a, d) for a in ARMS) for d in DIMS}
        v["F"] = {"holds": all(x <= 0.03 for x in spread.values()), "max_minus_min": spread}
        v["H3"] = {"holds": all(abs(T("JAC-OPP", d) - T("G-MSE", d)) <= 0.02 for d in DIMS if d >= 4), "jac_minus_gmse": {d: T("JAC-OPP", d) - T("G-MSE", d) for d in DIMS}}
        best8 = max(dec_arms, key=lambda a: T(a, 8)); at = T(best8, 8) - T("RECON", 8); ao = T(best8, 8, "ood_frac_0.1") - T("RECON", 8, "ood_frac_0.1")
        re = {e: T("REGRET", 8, f"test_frac_{e}") - T("RECON", 8, f"test_frac_{e}") for e in EPS_LIST}
        v["H4"] = {"a_holds": bool(ao < at), "adv_test": at, "adv_ood": ao, "best_arm_d8": best8,
                   "b_holds": bool(re[0.05] < re[0.10] and re[0.20] < re[0.10]), "regret_minus_recon_by_eps": re}
    # ---------------- Part C: decision-equivalence structure on test pairs
    te = S["test"]; hd_ = HandDist(pop); Pt = hd_(te["Q"]); Dbeh = js_matrix(Pt)
    X = te["X"][0.10]; Gt = te["G"]; V = te["V"][0.10]; C = V[:, None] - Gt @ X.T; Ddec = 0.5 * (C + C.T)
    iu = np.triu_indices(len(Gt), 1); db, dd = Dbeh[iu], Ddec[iu]; pc = {}
    for pct in (10, 20):
        deq = (dd <= np.percentile(dd, pct)) & (db >= np.median(db)); bcd = (db <= np.percentile(db, pct)) & (dd >= np.median(dd))
        pc[pct] = (deq, bcd)
        if deq.sum() >= 50 and bcd.sum() >= 50:
            break
    deq, bcd = pc[pct]; rho_db = spearmanr(dd, db)[0]
    partc = {"n_pairs": int(len(dd)), "pct_used": pct, "n_DEQ_BF": int(deq.sum()), "n_BC_DD": int(bcd.sum()), "spearman_Ddec_Dbeh": float(rho_db),
             "Ddec_quantiles": [float(np.percentile(dd, q)) for q in (10, 50, 90)], "Dbeh_quantiles": [float(np.percentile(db, q)) for q in (10, 50, 90)], "models": {}}
    rdd, rdb = rankdata(dd), rankdata(db)
    for arm in ARMS + POSTHOC_ARMS:
        for d in DIMS:
            vals = []
            for s in SEEDS:
                e = load_eval(run_name(arm, d, s))
                if e is None:
                    continue
                z = e["test_z"]; Dz = np.sqrt(np.maximum(((z[:, None] - z[None]) ** 2).sum(2), 0))[iu]; Dz = Dz / np.median(Dz); rz = rankdata(Dz)
                r_zd = np.corrcoef(rz, rdd)[0, 1]; r_zb = np.corrcoef(rz, rdb)[0, 1]; r_db = np.corrcoef(rdd, rdb)[0, 1]
                partial = (r_zd - r_zb * r_db) / np.sqrt((1 - r_zb ** 2) * (1 - r_db ** 2))
                vals.append((r_zd, r_zb, partial, Dz[deq].mean() / Dz[bcd].mean()))
            if vals:
                a = np.array(vals); partc["models"][f"{arm}_d{d}"] = {"rho_dec": float(a[:, 0].mean()), "rho_beh": float(a[:, 1].mean()), "partial_dec_given_beh": float(a[:, 2].mean()),
                                                                     "SR": float(a[:, 3].mean()), "SR_seeds": a[:, 3].tolist()}
    # oracle reference: the DEC and BEH partitions at K = 16 (same-cell = distance 0)
    asg = np.load(D / "oracle_assign.npz")
    for key in ("DEC_16", "BEH_16", "GVAR_16", "DEC_64", "BEH_64"):
        a = asg[key]; same = (a[:, None] == a[None])[iu]
        partc[f"oracle_{key}"] = {"share_DEQ_BF_same_cell": float(same[deq].mean()), "share_BC_DD_same_cell": float(same[bcd].mean()), "share_all_same_cell": float(same.mean())}
    res["partC"] = partc; m8 = partc["models"]
    if all(f"{a}_d8" in m8 for a in ("REGRET", "G-MSE", "RECON")):
        gap = {a: m8[f"{a}_d8"]["rho_dec"] - m8[f"{a}_d8"]["rho_beh"] for a in ARMS if f"{a}_d8" in m8}
        v["C1"] = {"holds": bool(gap["REGRET"] > gap["RECON"] and gap["G-MSE"] > gap["RECON"]), "rho_dec_minus_rho_beh_d8": gap}
        v["C2"] = {"holds": bool(m8["REGRET_d8"]["SR"] < 1 and m8["RECON_d8"]["SR"] > 1), "SR_d8": {a: m8[f"{a}_d8"]["SR"] for a in ARMS if f"{a}_d8" in m8}}
    res["verdicts_B_C"] = v
    # post-hoc: the behaviour-optimal autoencoder (OBS-RECON) as the behavioural reference (not a pre-registered test)
    if all(f"OBS-RECON_d{d}" in tab for d in DIMS):
        ph = []
        for d in DIMS:
            val = T("REGRET", d); dr = next((x for x in DIMS if T("OBS-RECON", x) >= val), None)
            ph.append({"d": d, "REGRET": val, "OBS-RECON": T("OBS-RECON", d), "obs_d_to_match_regret": dr,
                       "beh_REGRET": T("REGRET", d, "test_beh"), "beh_OBS": T("OBS-RECON", d, "test_beh"), "beh_RECON": T("RECON", d, "test_beh"),
                       "pol_REGRET": T("REGRET", d, "test_pol"), "pol_OBS": T("OBS-RECON", d, "test_pol"), "pol_RECON": T("RECON", d, "test_pol")})
        res["posthoc_obs_recon"] = ph
    save_json(res, D / "analysis.json")
    print(json.dumps({**{k: x.get("holds") for k, x in res["partA"].items()}, **{k: (x.get("holds"), x.get("a_holds"), x.get("b_holds")) if k == "H4" else x.get("holds") for k, x in v.items()}}, indent=1))


if __name__ == "__main__":
    main()
