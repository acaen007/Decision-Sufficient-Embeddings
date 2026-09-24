"""Analysis for REPORT_LEDUC_JACOPP: pre-registered P1-P4 and falsification, all arms vs arm 0, per seed, per family,
eps transfer, g-NMSE, validation selection summary, EM-prior check, audits.  Test data are only read here.

Usage: python -m leduc_decision_repr.jacopp_analysis [--dry A0=GROUP,A3=GROUP,...]  (--dry maps arms onto existing
groups to exercise the code; the output then goes to analysis_dry.json)."""
import argparse, json
import numpy as np
from .common import OUT, N_BUDGETS, EPSILONS, save_json
from .data.datasets import load_population
from .analysis.metrics import EvalData
from .v3_analysis import FAMS
from .ft_analysis import members, R_of, nmse_of, paired

NS = list(N_BUDGETS); GE50 = [NS.index(n) for n in (50, 100, 200, 500)]; GE20 = [NS.index(n) for n in (20, 50, 100, 200, 500)]
ARMS = {"A0": "NEURAL_JO_A0", "A1": "NEURAL_JO_A1", "A2": "NEURAL_JO_A2", "A3": "NEURAL_JO_A3", "A4": "NEURAL_JO_A4"}
NAMES = {"A0": "RECON-JAC-global", "A1": "JAC-global-proj", "A2": "OPP-REACH", "A3": "JAC-opp", "A4": "JAC-opp + g-term"}


def seed_avg(ed, g, k, seeds=None):
    ms = members(ed, g)
    if seeds is not None:
        ms = [m for m in ms if int(m.rsplit("_s", 1)[1]) in seeds]
    return np.mean([R_of(ed, m, k) for m in ms], 0), ms


def compare(ed, ga, gb, rng, k, common_seeds=True):
    """Paired over opponents; seeds restricted to those both arms have (so the per-seed pairing is exact)."""
    sa = {int(m.rsplit("_s", 1)[1]) for m in members(ed, ga)}; sb = {int(m.rsplit("_s", 1)[1]) for m in members(ed, gb)}
    seeds = sorted(sa & sb) if common_seeds else None
    if not seeds:
        return None
    Ra, _ = seed_avg(ed, ga, k, seeds); Rb, _ = seed_avg(ed, gb, k, seeds)
    out = {"seeds": seeds, "a": Ra.mean(0).tolist(), "b": Rb.mean(0).tolist(), "diff": paired(Ra, Rb, rng)}
    out["per_seed"] = {s: paired(seed_avg(ed, ga, k, [s])[0], seed_avg(ed, gb, k, [s])[0], rng, 1000) for s in seeds}
    out["by_family"] = {fn: paired(Ra[ed.family == f], Rb[ed.family == f], rng, 1000) for f, fn in enumerate(FAMS)}
    return out


def ratio_boot(ed, arms, rng, k=2, n_boot=2000):
    """P4: (A2 - A0) / (A3 - A0) per N >= 50 and pooled over N >= 50, with one set of opponent resamples."""
    seeds = sorted(set.intersection(*[{int(m.rsplit("_s", 1)[1]) for m in members(ed, arms[a])} for a in ("A0", "A2", "A3")]))
    R0, R2, R3 = (seed_avg(ed, arms[a], k, seeds)[0] for a in ("A0", "A2", "A3"))
    D2, D3 = R2 - R0, R3 - R0; n = len(D2)
    def stat(idx):
        num = D2[idx][:, GE50].mean(0); den = D3[idx][:, GE50].mean(0)
        return np.concatenate([num / den, [num.mean() / den.mean()]]), np.concatenate([num, [num.mean()]]), np.concatenate([den, [den.mean()]])
    r0, nu0, de0 = stat(np.arange(n))
    B = [stat(rng.integers(0, n, n)) for _ in range(n_boot)]
    rb = np.array([b[0] for b in B]); nb = np.array([b[1] for b in B]); db = np.array([b[2] for b in B])
    lab = [str(NS[j]) for j in GE50] + ["pooled_N>=50"]
    pct = lambda a: (np.percentile(a, 2.5, axis=0).tolist(), np.percentile(a, 97.5, axis=0).tolist())
    return {"seeds": seeds, "labels": lab, "ratio": r0.tolist(), "ratio_ci": pct(rb), "numerator_A2_minus_A0": nu0.tolist(), "numerator_ci": pct(nb),
            "denominator_A3_minus_A0": de0.tolist(), "denominator_ci": pct(db),
            "denominator_ci_excludes_0": [bool(lo > 0 or hi < 0) for lo, hi in zip(*pct(db))]}


def val_summary():
    R = OUT / "runs_jacopp"; out = {}
    for d in sorted(R.glob("*_s*")):
        if not (d / "result.json").exists():
            continue
        res = json.loads((d / "result.json").read_text()); curve, aud, viol, fails = [], 0.0, 0, 0
        for l in open(d / "log.jsonl"):
            r = json.loads(l)
            if "val_sel_regret" in r:
                curve.append([r["step"] + 1, r["val_sel_regret"], r["val_sel_g_nmse"]])
                aud = max(aud, r["val_sel_audit_max"]); viol += r["val_sel_audit_viol"]; fails += r["val_sel_lp_fail"]
        out[d.name] = {"best_step": res["best_step"] + 1, "best_val_regret": res["best_val_loss"], "runtime_h": res["runtime_s"] / 3600,
                       "curve": curve, "val_audit_max": aud, "val_audit_viol": viol, "val_lp_fail": fails, "n_val_strategies": 450 * len(curve)}
    return out


def main(dry=None):
    pop = load_population(); ed = EvalData(OUT / "eval" / "test", pop); rng = np.random.default_rng(0)
    arms = dict(ARMS); arms.update(dry or {})
    res = {"arms": {a: {"group": g, "name": NAMES[a], "members": members(ed, g)} for a, g in arms.items()}}
    res["three_seeds"] = {a: len(v["members"]) == 3 for a, v in res["arms"].items()}
    # regret tables and every arm vs arm 0, at each eps
    for k in (1, 2, 3):
        e = f"eps{EPSILONS[k]}"; res[e] = {"regret": {}, "vs_A0": {}, "vs_A3": {}}
        for a, g in arms.items():
            if members(ed, g) and all(np.isfinite(ed.solve[m]["u"][:, :, k]).all() for m in members(ed, g)):
                res[e]["regret"][a] = seed_avg(ed, g, k)[0].mean(0).tolist()
        for a in ("A1", "A2", "A3", "A4"):
            if a in res[e]["regret"] and "A0" in res[e]["regret"]:
                res[e]["vs_A0"][a] = compare(ed, arms[a], arms["A0"], rng, k)
        for a in ("A1", "A2", "A4"):
            if a in res[e]["regret"] and "A3" in res[e]["regret"]:
                res[e]["vs_A3"][a] = compare(ed, arms[a], arms["A3"], rng, k)
    # g-NMSE
    res["g_nmse"] = {a: np.mean([nmse_of(ed, m) for m in members(ed, g)], 0).mean(0).tolist() for a, g in arms.items() if members(ed, g)}
    res["g_nmse_vs_A0"] = {}
    for a in ("A1", "A2", "A3", "A4"):
        if a in res["g_nmse"] and "A0" in res["g_nmse"]:
            s = sorted({int(m.rsplit("_s", 1)[1]) for m in members(ed, arms[a])} & {int(m.rsplit("_s", 1)[1]) for m in members(ed, arms["A0"])})
            Na = np.mean([nmse_of(ed, f"{arms[a]}_s{x}") for x in s], 0); Nb = np.mean([nmse_of(ed, f"{arms['A0']}_s{x}") for x in s], 0)
            res["g_nmse_vs_A0"][a] = paired(Na, Nb, rng)
    # pre-registered verdicts (eps = 0.10)
    v = {}; e = res["eps0.1"]
    if "A3" in e["vs_A0"] and e["vs_A0"]["A3"]:
        d = e["vs_A0"]["A3"]["diff"]; gap = -np.array(d["mean"])
        slope = float(np.polyfit(np.log([NS[j] for j in GE50]), gap[GE50], 1)[0])
        each = all(d["mean"][j] <= -0.003 and d["hi"][j] < 0 for j in GE50); growing = bool(gap[GE50[-1]] > gap[GE50[0]] and slope > 0)
        # falsification: CI of the difference averaged over N >= 50
        seeds = e["vs_A0"]["A3"]["seeds"]
        D = seed_avg(ed, arms["A3"], 2, seeds)[0][:, GE50].mean(1, keepdims=True) - seed_avg(ed, arms["A0"], 2, seeds)[0][:, GE50].mean(1, keepdims=True)
        avg = paired(D, np.zeros_like(D), rng)
        v["P1"] = {"holds": bool(each and growing), "each_N_ge50_le_-0.003_and_CI_below_0": bool(each), "gap_growing": growing,
                   "diff_N>=50": [d["mean"][j] for j in GE50], "hi_N>=50": [d["hi"][j] for j in GE50], "slope_gap_on_logN": slope,
                   "mean_diff_over_N>=50": avg, "seeds": seeds}
        v["falsified"] = bool(avg["hi"][0] >= 0)
    if "A4" in e["vs_A3"] and e["vs_A3"]["A4"]:
        d = e["vs_A3"]["A4"]["diff"]
        v["P2"] = {"holds": bool(all(d["hi"][j] < 0.003 for j in GE20)), "diff_N>=20": [d["mean"][j] for j in GE20], "hi_N>=20": [d["hi"][j] for j in GE20]}
    if "A1" in e["vs_A0"] and e["vs_A0"]["A1"]:
        d = e["vs_A0"]["A1"]["diff"]
        v["P3"] = {"note": "V3 did not project (Step 0): reported as the effect of the correction (projection + squaring + per-opponent normalization)",
                   "diff": d["mean"], "lo": d["lo"], "hi": d["hi"]}
    if all(a in e["regret"] for a in ("A0", "A2", "A3")):
        v["P4"] = ratio_boot(ed, arms, rng)
    res["verdict"] = v
    # EM-prior check
    em = {}
    for a, b in [("HYB_PRIOR_EM_JOBEST", "HYB_PRIOR_EM"), ("HYB_PRIOR_EM_JOBEST", "HYB_PRIOR_EM_JAC")]:
        if a in ed.methods and b in ed.methods and np.isfinite(ed.solve[a]["u"][:, :, 2]).all():
            em[f"{a} - {b}"] = {"a": ed.regret(a)[:, :, 2].mean(0).tolist(), "b": ed.regret(b)[:, :, 2].mean(0).tolist(),
                                "diff": paired(ed.regret(a)[:, :, 2], ed.regret(b)[:, :, 2], rng)}
    for a, g in arms.items():
        if "HYB_PRIOR_EM_JOBEST" in ed.methods and members(ed, g):
            em[f"HYB_PRIOR_EM_JOBEST - {a}"] = {"diff": paired(ed.regret("HYB_PRIOR_EM_JOBEST")[:, :, 2], seed_avg(ed, g, 2)[0], rng)}
    res["em_prior"] = em
    if (OUT / "jacopp" / "emprior_choice.json").exists():
        res["em_prior_choice"] = json.loads((OUT / "jacopp" / "emprior_choice.json").read_text())
    # audits of every deployed test strategy of this study
    aud = {}
    for m in ed.methods:
        if m.startswith(("NEURAL_JO_", "HYB_PRIOR_EM_JOBEST")):
            s = ed.solve[m]; e_ = s["e_os"]; ok = s["ok"] & np.isfinite(e_)
            viol = np.where(np.isfinite(e_), e_ - np.array(EPSILONS)[None, None, :], -np.inf)
            aud[m] = {"n_solved": int(np.isfinite(e_).sum()), "n_ok": int(ok.sum()), "max_expl_minus_eps": float(viol.max()),
                      "n_viol": int((viol > 1e-7).sum()), "lp_failures": int(s["lp_failures"])}
    res["audit"] = aud
    if not dry:
        res["validation"] = val_summary()
        if (OUT / "jacopp" / "decisions.json").exists():
            res["decisions"] = json.loads((OUT / "jacopp" / "decisions.json").read_text())
    save_json(res, OUT / "jacopp" / ("analysis_dry.json" if dry else "analysis.json"))
    print(json.dumps(res["verdict"], indent=1)[:4000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", default="")
    a = ap.parse_args()
    main(dict(kv.split("=") for kv in a.dry.split(",")) if a.dry else None)
