"""Analysis for REPORT_LEDUC_FT: paired comparisons (seed-averaged and per seed), g-NMSE, decision hit rate,
eps transfer, per family, EM-prior check, audits, pre-registered verdicts."""
import json, sys
import numpy as np
from .common import OUT, N_BUDGETS, EPSILONS, save_json, load_json
from .data.datasets import load_population
from .analysis.metrics import EvalData, paired_bootstrap_diff, bootstrap_mean
from .v3_analysis import FAMS

NI = N_BUDGETS; NGE50 = [NI.index(n) for n in (50, 100, 200, 500)]


def members(ed, group):
    return sorted(ed.neural_groups().get(group, []))


def R_of(ed, m, k):
    return ed.regret(m)[:, :, k]                                            # (n_opp, nN)


def hit_of(ed, m, k, thr=1e-6):
    u = ed.solve[m]["u"][:, :, k]                                           # (H, nN)
    Rh = ed.V[np.repeat(np.arange(ed.n_opp), ed.n_streams), k][:, None] - u
    return ed.per_opp((Rh < thr).astype(float))                              # (n_opp, nN) fraction of streams


def nmse_of(ed, m):
    return ed.per_opp(ed.pred[m]["g_nmse"])


def paired(a, b, rng, n_boot=2000):
    m, lo, hi, _ = paired_bootstrap_diff(a, b, rng, n_boot=n_boot)
    return {"mean": m.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}


def arm_compare(ed, ga, gb, rng, ks=(1, 2, 3)):
    ma, mb = members(ed, ga), members(ed, gb)
    if not ma or not mb:
        return None
    out = {"a": ga, "b": gb, "seeds": [len(ma), len(mb)]}
    for k in ks:
        if not all(np.isfinite(ed.solve[m]["u"][:, :, k]).all() for m in ma + mb):
            continue
        Ra = np.mean([R_of(ed, m, k) for m in ma], 0); Rb = np.mean([R_of(ed, m, k) for m in mb], 0)
        e = {"regret_a": Ra.mean(0).tolist(), "regret_b": Rb.mean(0).tolist(), "diff": paired(Ra, Rb, rng)}
        e["per_seed"] = {}
        for sa, sb in zip(ma, mb):
            ra, rb = R_of(ed, sa, k), R_of(ed, sb, k); e["per_seed"][f"{sa} - {sb}"] = paired(ra, rb, rng, 1000)
        e["by_family"] = {fn: paired(Ra[ed.family == f], Rb[ed.family == f], rng, 1000) for f, fn in enumerate(FAMS)}
        Ha = np.mean([hit_of(ed, m, k) for m in ma], 0); Hb = np.mean([hit_of(ed, m, k) for m in mb], 0)
        e["hit_rate"] = {"a_by_N": Ha.mean(0).tolist(), "b_by_N": Hb.mean(0).tolist(), "a_pooled": float(Ha.mean()), "b_pooled": float(Hb.mean()),
                         "diff_pooled": paired(Ha.mean(1, keepdims=True), Hb.mean(1, keepdims=True), rng)}
        for thr in (1e-3, 1e-2):
            Ha2 = np.mean([hit_of(ed, m, k, thr) for m in ma], 0); Hb2 = np.mean([hit_of(ed, m, k, thr) for m in mb], 0)
            e[f"near_hit_{thr:g}"] = {"a_by_N": Ha2.mean(0).tolist(), "b_by_N": Hb2.mean(0).tolist(), "diff_pooled": paired(Ha2.mean(1, keepdims=True), Hb2.mean(1, keepdims=True), rng)}
        out[f"eps{EPSILONS[k]}"] = e
    Na = np.mean([nmse_of(ed, m) for m in ma], 0); Nb = np.mean([nmse_of(ed, m) for m in mb], 0)
    out["g_nmse"] = {"a": Na.mean(0).tolist(), "b": Nb.mean(0).tolist(), "diff": paired(Na, Nb, rng)}
    return out


def main():
    pop = load_population(); ed = EvalData(OUT / "eval" / "test", pop); rng = np.random.default_rng(0); res = {}
    res["groups"] = {g: members(ed, g) for g in ["NEURAL_FTJACCTRL", "NEURAL_FTJACSPO", "NEURAL_FTDECCTRL", "NEURAL_FTDECSPO", "NEURAL_RECJAC889K", "NEURAL_DECISION"]}
    res["phase2"] = arm_compare(ed, "NEURAL_FTJACSPO", "NEURAL_FTJACCTRL", rng)
    res["phase3"] = arm_compare(ed, "NEURAL_FTDECSPO", "NEURAL_FTDECCTRL", rng)
    res["jac_ctrl_vs_base"] = arm_compare(ed, "NEURAL_FTJACCTRL", "NEURAL_RECJAC889K", rng)
    res["jac_spo_vs_base"] = arm_compare(ed, "NEURAL_FTJACSPO", "NEURAL_RECJAC889K", rng)
    res["dec_ctrl_vs_base"] = arm_compare(ed, "NEURAL_FTDECCTRL", "NEURAL_DECISION", rng)
    res["dec_spo_vs_base"] = arm_compare(ed, "NEURAL_FTDECSPO", "NEURAL_DECISION", rng)
    res["jac_spo_vs_dec_spo"] = arm_compare(ed, "NEURAL_FTJACSPO", "NEURAL_FTDECSPO", rng)
    # EM-prior check
    em = {}
    for a, b in [("HYB_PRIOR_EM_FTSPO", "HYB_PRIOR_EM"), ("HYB_PRIOR_EM_FTSPO", "HYB_PRIOR_EM_FTCTRL"), ("HYB_PRIOR_EM_FTCTRL", "HYB_PRIOR_EM"),
                 ("HYB_PRIOR_EM_FTSPO", "HYB_PRIOR_EM_JAC")]:
        if a in ed.methods and b in ed.methods:
            em[f"{a} - {b}"] = {"a": ed.regret(a)[:, :, 2].mean(0).tolist(), "b": ed.regret(b)[:, :, 2].mean(0).tolist(), "diff": paired(ed.regret(a)[:, :, 2], ed.regret(b)[:, :, 2], rng)}
    res["em_prior"] = em
    # audits of every FT / hybrid deployed strategy
    aud = {}
    for m in ed.methods:
        if m.startswith(("NEURAL_FT", "HYB_PRIOR_EM_FT")):
            s = ed.solve[m]; e = s["e_os"]; ok = s["ok"] & np.isfinite(e)
            eps = np.array(EPSILONS)[None, None, :]; viol = np.where(ok, e - eps, -np.inf)
            aud[m] = {"n": int(ok.sum()), "max_expl_minus_eps": float(viol.max()), "n_viol": int((viol > 1e-7).sum()), "lp_failures": int(s["lp_failures"])}
    res["audit"] = aud
    # pre-registered verdicts
    v = {}
    p2 = res["phase2"]
    if p2 and "eps0.1" in p2:
        d = p2["eps0.1"]["diff"]; ok_each = all(d["mean"][j] <= -0.008 and d["hi"][j] < 0 for j in NGE50)
        seeds_ok = all(np.mean([ps["mean"][j] for j in NGE50]) < 0 for ps in p2["eps0.1"]["per_seed"].values())
        v["P1"] = {"holds": bool(ok_each and seeds_ok), "diff_at_N>=50": [d["mean"][j] for j in NGE50], "hi_at_N>=50": [d["hi"][j] for j in NGE50],
                   "per_seed_mean_over_N>=50": {k: float(np.mean([ps["mean"][j] for j in NGE50])) for k, ps in p2["eps0.1"]["per_seed"].items()}}
        g = p2["g_nmse"]["diff"]; v["P2"] = {"holds": bool(all(g["hi"][j] >= 0 for j in NGE50)), "nmse_diff_at_N>=50": [g["mean"][j] for j in NGE50],
                                             "hi": [g["hi"][j] for j in NGE50]}
        h = p2["eps0.1"]["hit_rate"]; v["P3"] = {"holds": bool(h["diff_pooled"]["lo"][0] > 0), "a": h["a_pooled"], "b": h["b_pooled"], "diff": h["diff_pooled"]}
    p3 = res["phase3"]
    if p3 and "eps0.1" in p3:
        d = p3["eps0.1"]["diff"]; avg = float(np.mean([d["mean"][j] for j in NGE50]))
        v["P4"] = {"holds": bool(avg < -0.008 and all(d["hi"][j] < 0 for j in NGE50)), "mean_diff_over_N>=50": avg, "hi_at_N>=50": [d["hi"][j] for j in NGE50]}
    v["decision"] = ("GO" if v.get("P1", {}).get("holds") else "NO-GO: instance-specific decision-awareness adds little beyond decision-relevance weighting") if "P1" in v else "pending"
    res["verdict"] = v
    save_json(res, OUT / "ft" / "analysis.json")
    print(json.dumps(res["verdict"], indent=1))


if __name__ == "__main__":
    main()
