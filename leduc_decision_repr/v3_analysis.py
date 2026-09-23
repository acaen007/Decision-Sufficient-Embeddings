"""V3 analysis helpers: matched-pair comparisons (T1), censoring toggle (T2), SPO+ arms (T4), hybrids (T5).
All numbers come from the saved evaluation arrays; paired bootstrap over the 300 held-out opponents."""
from __future__ import annotations

import json, sys, glob
from pathlib import Path

import numpy as np

from .common import OUT, N_BUDGETS, EPSILONS, save_json, load_json
from .data.datasets import load_population
from .analysis.metrics import EvalData, bootstrap_mean, paired_bootstrap_diff, auc_log_n

FAMS = ["NASH_LOGIT_PERTURB", "NASH_RANDOM_MIX", "STRUCTURED_CORRELATED", "UNSTRUCTURED_DIRICHLET"]


def group_regret(ed, group, eps_idx):
    """Mean over seeds of the per-opponent regret (n_opp, nN) for a group name; also per-seed list."""
    members = ed.neural_groups().get(group, [])
    if not members:
        return None, []
    per_seed = [ed.regret(m)[:, :, eps_idx] for m in members]
    return np.mean(per_seed, 0), per_seed


def compare(ed, a, b, eps_idx, rng, label=None):
    """Paired comparison a - b (negative = a better). Returns dict with means, CIs, per family, seed spread."""
    Ra, sa = group_regret(ed, a, eps_idx); Rb, sb = group_regret(ed, b, eps_idx)
    if Ra is None or Rb is None:
        return None
    out = {"a": a, "b": b, "eps": EPSILONS[eps_idx], "n_seeds": [len(sa), len(sb)],
           "regret_a": np.nanmean(Ra, 0).tolist(), "regret_b": np.nanmean(Rb, 0).tolist()}
    m, lo, hi, _ = paired_bootstrap_diff(Ra, Rb, rng)
    out["diff_mean"] = m.tolist(); out["diff_lo"] = lo.tolist(); out["diff_hi"] = hi.tolist()
    out["seed_spread_a"] = (np.max([np.nanmean(s, 0) for s in sa], 0) - np.min([np.nanmean(s, 0) for s in sa], 0)).tolist()
    out["seed_spread_b"] = (np.max([np.nanmean(s, 0) for s in sb], 0) - np.min([np.nanmean(s, 0) for s in sb], 0)).tolist()
    out["auc_a"] = auc_log_n(np.nanmean(Ra, 0)); out["auc_b"] = auc_log_n(np.nanmean(Rb, 0))
    out["by_family"] = {}
    for f, fn in enumerate(FAMS):
        sel = ed.family == f
        mf, lof, hif, _ = paired_bootstrap_diff(Ra[sel], Rb[sel], rng, n_boot=1000)
        out["by_family"][fn] = {"regret_a": np.nanmean(Ra[sel], 0).tolist(), "regret_b": np.nanmean(Rb[sel], 0).tolist(),
                                "diff_mean": mf.tolist(), "diff_lo": lof.tolist(), "diff_hi": hif.tolist()}
    return out


def run_summary(run_dir):
    r = load_json(Path(run_dir) / "result.json"); c = load_json(Path(run_dir) / "config.json")
    recs = [json.loads(l) for l in open(Path(run_dir) / "log.jsonl")]
    vals = [x for x in recs if "val_loss" in x]
    return {"n_params_head": c["n_params_head"], "n_params_encoder": c["n_params_encoder"], "best_step": r["best_step"] + 1,
            "best_val_loss": r["best_val_loss"], "steps_run": r["steps_run"], "stopped_early_at": r["stopped_early_at"],
            "plateau": r["plateau"], "val_curve": [(x["step"] + 1, x["val_loss"]) for x in vals],
            "val_g_nmse_N500_curve": [(x["step"] + 1, x["val_per_N"]["500"]["g_nmse"]) for x in vals], "runtime_h": r["runtime_s"] / 3600}


def t1(eval_dir=OUT / "eval" / "test"):
    pop = load_population(); ed = EvalData(eval_dir, pop); rng = np.random.default_rng(0)
    groups = ed.neural_groups()
    res = {"groups": {g: len(m) for g, m in groups.items()}, "comparisons": {}, "runs": {}}
    pairs = [("NEURAL_DEC133K", "NEURAL_RECON"), ("NEURAL_DECISION", "NEURAL_REC889K"), ("NEURAL_DEC133K", "NEURAL_DECISION"),
             ("NEURAL_REC889K", "NEURAL_RECON"), ("NEURAL_RECJAC889K", "NEURAL_REC889K"), ("NEURAL_RECJAC889K", "NEURAL_DECISION"),
             ("NEURAL_RECREACH889K", "NEURAL_REC889K"), ("NEURAL_DEC133K", "NEURAL_BANK_POSTERIOR")]
    for a, b in pairs:
        for k in [1, 2, 3]:
            c = compare(ed, a, b, k, rng)
            if c:
                res["comparisons"][f"{a}_minus_{b}_eps{EPSILONS[k]}"] = c
    # absolute curves for all groups + classical at eps 0.1
    res["curves_eps0.1"] = {}
    for g in groups:
        R, _ = group_regret(ed, g, 2)
        m, lo, hi, _ = bootstrap_mean(R, rng); res["curves_eps0.1"][g] = {"mean": m.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
    for m_ in ["BANK_POSTERIOR", "TABULAR_EM_UNIFORM", "TABULAR_EM_NASH"]:
        if m_ in ed.methods:
            R = ed.regret(m_)[:, :, 2]; m, lo, hi, _ = bootstrap_mean(R, rng); res["curves_eps0.1"][m_] = {"mean": m.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
    for d in sorted(glob.glob(str(OUT / "runs_v3" / "*"))):
        if (Path(d) / "result.json").exists():
            res["runs"][Path(d).name] = run_summary(d)
    # V1 reference runs (unmatched arms)
    for d in sorted(glob.glob(str(OUT / "runs" / "*"))):
        if (Path(d) / "result.json").exists() and (Path(d) / "log.jsonl").exists():
            try:
                res["runs"]["V1_" + Path(d).name] = run_summary(d)
            except Exception:
                pass
    res["safety"] = {m: ed.solve_summary(m) if hasattr(ed, "solve_summary") else None for m in []}
    save_json(res, OUT / "v3_t1_analysis.json")
    return res


def t2(eval_dir=OUT / "eval" / "test", rev_dir=OUT / "eval" / "test_revealed"):
    """Covariance gap vs decision advantage (matched 889k models), and the censoring toggle at 3000 steps."""
    from scipy.stats import spearmanr
    pop = load_population(); rng = np.random.default_rng(0)
    ed = EvalData(eval_dir, pop)
    res = {}
    gaps = np.load(OUT / "t2_covariance" / "gaps.npz")
    gap_g = ed.per_opp(gaps["gap_g"]); gap_reg = ed.per_opp(gaps["gap_reg"])         # (n_opp, nN)
    Rd, _ = group_regret(ed, "NEURAL_DECISION", 2); Rr, _ = group_regret(ed, "NEURAL_REC889K", 2)
    if Rr is None:
        Rr, _ = group_regret(ed, "NEURAL_RECON", 2); res["note"] = "REC889K not available; used RECON-131k"
    adv = Rr - Rd                                                                  # decision advantage (positive = decision better)
    res["gap_means_by_N"] = {"gap_g": gap_g.mean(0).tolist(), "gap_reg": gap_reg.mean(0).tolist(), "advantage": adv.mean(0).tolist()}
    res["spearman"] = {}
    for j, N in enumerate(N_BUDGETS):
        e = {"pooled_gap_g": float(spearmanr(gap_g[:, j], adv[:, j])[0]), "pooled_gap_reg": float(spearmanr(gap_reg[:, j], adv[:, j])[0])}
        for f, fn in enumerate(FAMS):
            sel = ed.family == f
            e[fn] = {"gap_g": float(spearmanr(gap_g[sel, j], adv[sel, j])[0]), "gap_reg": float(spearmanr(gap_reg[sel, j], adv[sel, j])[0]),
                     "mean_gap_g": float(gap_g[sel, j].mean()), "mean_gap_reg": float(gap_reg[sel, j].mean()), "mean_adv": float(adv[sel, j].mean())}
        res["spearman"][str(N)] = e
    # pooled over N as well
    res["spearman_pooled_allN"] = {"gap_g": float(spearmanr(gap_g.ravel(), adv.ravel())[0]), "gap_reg": float(spearmanr(gap_reg.ravel(), adv.ravel())[0])}
    # censoring toggle: revealed (censdec133k, censrec131k) vs uncensored at 3000 steps (dec133k step3000 ckpt = DEC133K3K, rec131k3k)
    if rev_dir.exists() and (rev_dir / "predict_meta.json").exists():
        er = EvalData(rev_dir, pop)
        c_rev = compare(er, "NEURAL_CENSDEC133K", "NEURAL_CENSREC131K", 2, rng)
        c_unc = compare(ed, "NEURAL_DEC133K3K", "NEURAL_REC131K3K", 2, rng)
        res["censoring"] = {"revealed_dec_minus_rec": c_rev, "censored_dec_minus_rec": c_unc}
        if c_rev and c_unc:
            # advantage difference (censored advantage - revealed advantage), paired over opponents
            Ra_r, _ = group_regret(er, "NEURAL_CENSDEC133K", 2); Rb_r, _ = group_regret(er, "NEURAL_CENSREC131K", 2)
            Ra_c, _ = group_regret(ed, "NEURAL_DEC133K3K", 2); Rb_c, _ = group_regret(ed, "NEURAL_REC131K3K", 2)
            adv_r = Rb_r - Ra_r; adv_c = Rb_c - Ra_c
            m, lo, hi, _ = paired_bootstrap_diff(adv_c, adv_r, rng)
            res["censoring"]["advantage_censored_minus_revealed"] = {"mean": m.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
            res["censoring"]["by_family"] = {}
            for f, fn in enumerate(FAMS):
                sel = ed.family == f
                mr, lor, hir, _ = bootstrap_mean(adv_r[sel], rng, n_boot=1000); mc, loc, hic, _ = bootstrap_mean(adv_c[sel], rng, n_boot=1000)
                res["censoring"]["by_family"][fn] = {"adv_revealed": mr.tolist(), "adv_revealed_lo": lor.tolist(), "adv_revealed_hi": hir.tolist(),
                                                     "adv_censored": mc.tolist(), "adv_censored_lo": loc.tolist(), "adv_censored_hi": hic.tolist()}
    save_json(res, OUT / "v3_t2_analysis.json")
    return res


def t4(eval_dir=OUT / "eval" / "test"):
    pop = load_population(); ed = EvalData(eval_dir, pop); rng = np.random.default_rng(0)
    res = {"comparisons": {}, "curves_eps0.1": {}, "g_nmse": {}}
    for a in ["NEURAL_SPO0_133K", "NEURAL_SPO1_133K", "NEURAL_SPO03_133K"]:
        c = compare(ed, a, "NEURAL_DEC133K", 2, rng)
        if c:
            res["comparisons"][f"{a}_minus_NEURAL_DEC133K"] = c
    for g, members in ed.neural_groups().items():
        if g in ("NEURAL_DEC133K", "NEURAL_SPO0_133K", "NEURAL_SPO1_133K", "NEURAL_SPO03_133K"):
            R, _ = group_regret(ed, g, 2); m, lo, hi, _ = bootstrap_mean(R, rng)
            res["curves_eps0.1"][g] = {"mean": m.tolist(), "lo": lo.tolist(), "hi": hi.tolist(), "n_seeds": len(members)}
            res["g_nmse"][g] = np.mean([ed.per_opp(ed.pred[m_]["g_nmse"]).mean(0) for m_ in members], 0).tolist()
    save_json(res, OUT / "v3_t4_analysis.json")
    return res


def t5(eval_dir=OUT / "eval" / "test"):
    pop = load_population(); ed = EvalData(eval_dir, pop); rng = np.random.default_rng(0)
    res = {"curves_eps0.1": {}, "envelope": {}}
    names = {}
    for g in ed.neural_groups():
        if g in ("NEURAL_DEC133K", "NEURAL_COUNT133K", "NEURAL_SPO0_133K", "NEURAL_SPO1_133K"):
            names[g] = group_regret(ed, g, 2)[0]
    for m_ in ["BANK_POSTERIOR", "TABULAR_EM_UNIFORM", "HYB_BLEND", "HYB_PRIOR_EM", "HYB_PRIOR_EM_S0", "HYB_ENSREC", "HYB_PRIOR_EM_889K"]:
        if m_ in ed.methods:
            names[m_] = ed.regret(m_)[:, :, 2]
    for k, R in names.items():
        m, lo, hi, _ = bootstrap_mean(R, rng); res["curves_eps0.1"][k] = {"mean": m.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
    base = [k for k in ["BANK_POSTERIOR", "NEURAL_DEC133K", "TABULAR_EM_UNIFORM"] if k in names]
    if base:
        env = np.min([np.nanmean(names[k], 0) for k in base], 0)
        res["envelope"]["lower_envelope_of_" + "+".join(base)] = env.tolist()
        for k, R in names.items():
            m = np.nanmean(R, 0)
            res["envelope"][k] = {"regret": m.tolist(), "excess_over_envelope": (m - env).tolist(), "on_envelope_within_0.005_all_N": bool(np.all(m - env <= 0.005))}
        # paired excess CI for hybrids vs the envelope's per-N best method
        for k in ["HYB_BLEND", "HYB_PRIOR_EM", "HYB_PRIOR_EM_S0", "HYB_ENSREC", "HYB_PRIOR_EM_889K", "NEURAL_COUNT133K"]:
            if k in names:
                best_per_N = [base[int(np.argmin([np.nanmean(names[b], 0)[j] for b in base]))] for j in range(len(N_BUDGETS))]
                Rbest = np.stack([names[best_per_N[j]][:, j] for j in range(len(N_BUDGETS))], 1)
                m, lo, hi, _ = paired_bootstrap_diff(names[k], Rbest, rng)
                res["envelope"][k]["excess_ci"] = {"mean": m.tolist(), "lo": lo.tolist(), "hi": hi.tolist(), "best_per_N": best_per_N}
    res["paired"] = {}
    for a, b in [("HYB_PRIOR_EM", "TABULAR_EM_UNIFORM"), ("HYB_PRIOR_EM", "BANK_POSTERIOR"), ("HYB_PRIOR_EM", "NEURAL_DEC133K"), ("HYB_PRIOR_EM", "HYB_PRIOR_EM_S0"),
                 ("HYB_ENSREC", "HYB_PRIOR_EM"), ("HYB_BLEND", "NEURAL_DEC133K"), ("HYB_BLEND", "TABULAR_EM_UNIFORM"), ("NEURAL_COUNT133K", "NEURAL_DEC133K"), ("HYB_PRIOR_EM_889K", "HYB_PRIOR_EM")]:
        if a in names and b in names:
            m, lo, hi, _ = paired_bootstrap_diff(names[a], names[b], rng)
            e = {"diff_mean": m.tolist(), "diff_lo": lo.tolist(), "diff_hi": hi.tolist(), "by_family": {}}
            for f, fn in enumerate(FAMS):
                sel = ed.family == f; mf, lof, hif, _ = paired_bootstrap_diff(names[a][sel], names[b][sel], rng, n_boot=1000)
                e["by_family"][fn] = {"diff_mean": mf.tolist(), "diff_lo": lof.tolist(), "diff_hi": hif.tolist()}
            res["paired"][f"{a}_minus_{b}"] = e
    save_json(res, OUT / "v3_t5_analysis.json")
    return res


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "t1"
    r = {"t1": t1, "t2": t2, "t4": t4, "t5": t5}[which]()
    for k, c in r.get("comparisons", {}).items():
        if c["eps"] == 0.1:
            print(k, "diff:", " ".join(f"{a:+.3f}[{lo:+.3f},{hi:+.3f}]" for a, lo, hi in zip(c["diff_mean"], c["diff_lo"], c["diff_hi"])))
