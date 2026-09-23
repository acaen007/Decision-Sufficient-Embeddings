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


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "t1"
    r = {"t1": t1}[which]()
    for k, c in r.get("comparisons", {}).items():
        if c["eps"] == 0.1:
            print(k, "diff:", " ".join(f"{a:+.3f}[{lo:+.3f},{hi:+.3f}]" for a, lo, hi in zip(c["diff_mean"], c["diff_lo"], c["diff_hi"])))
