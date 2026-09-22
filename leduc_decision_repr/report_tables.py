"""Print markdown tables for the report from summary.json / geometry.json / precondition.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .common import OUT, N_BUDGETS, EPSILONS, load_json

ORDER = ["NEURAL_DECISION", "NEURAL_RECON", "BANK_POSTERIOR", "TABULAR_EM_NASH", "TABULAR_EM_UNIFORM"]


def fmt(v, d=3):
    return f"{v:.{d}f}"


def main(eval_dir: Path):
    S = load_json(eval_dir / "summary.json"); G = load_json(eval_dir / "geometry.json"); P = load_json(eval_dir / "precondition.json")
    out = []
    out.append("### Table R1. Safe response regret (chips/hand, mean over 300 held-out opponents, 8 streams each; 95% paired-bootstrap CI)\n")
    for eps in EPSILONS:
        out.append(f"**ε = {eps}** (Nash regret = oracle gain = {S['oracle_gain'][str(eps)]['mean']:.3f} [{S['oracle_gain'][str(eps)]['lo']:.3f}, {S['oracle_gain'][str(eps)]['hi']:.3f}])\n")
        out.append("| method | " + " | ".join(f"N={n}" for n in N_BUDGETS) + " | AUC(logN) |")
        out.append("|---|" + "---|" * (len(N_BUDGETS) + 1))
        for m in ORDER + [k for k in S["methods"] if k not in ORDER]:
            if m not in S["methods"]:
                continue
            r = S["methods"][m]["regret"][str(eps)]; auc = S["methods"][m]["auc_regret"][str(eps)]
            out.append(f"| {m} | " + " | ".join(f"{a:.3f} [{lo:.3f},{hi:.3f}]" for a, lo, hi in zip(r["mean"], r["lo"], r["hi"])) + f" | {auc['value']:.3f} [{auc['lo']:.3f},{auc['hi']:.3f}] |")
        out.append("")
    out.append("### Table R2. Fraction of oracle-safe gain recovered F (opponents with gain ≥ 0.02 only)\n")
    for eps in [e for e in EPSILONS if e > 0]:
        og = S["oracle_gain"][str(eps)]
        out.append(f"**ε = {eps}** ({og['n_valid_F']} of 300 opponents with valid denominator)\n")
        out.append("| method | " + " | ".join(f"N={n}" for n in N_BUDGETS) + " | N50 | N80 | N90 | AUC_F |")
        out.append("|---|" + "---|" * (len(N_BUDGETS) + 4))
        for m in ORDER + [k for k in S["methods"] if k not in ORDER]:
            if m not in S["methods"]:
                continue
            f = S["methods"][m]["frac"][str(eps)]; nt = S["methods"][m]["N_thresholds"][str(eps)]; auc = S["methods"][m]["auc_frac"][str(eps)]
            out.append(f"| {m} | " + " | ".join(f"{a:.2f} [{lo:.2f},{hi:.2f}]" for a, lo, hi in zip(f["mean"], f["lo"], f["hi"])) + f" | {nt['N50']} | {nt['N80']} | {nt['N90']} | {auc['value']:.3f} [{auc['lo']:.3f},{auc['hi']:.3f}] |")
        out.append("")
    out.append("### Table R3. Bootstrap distribution of N-thresholds (median [2.5%, 97.5%] over 2000 opponent resamples; 1000 = not reached by 500)\n")
    out.append("| ε | method | N50 | N80 | N90 |")
    out.append("|---|---|---|---|---|")
    for eps in [e for e in EPSILONS if e > 0]:
        for m in ORDER:
            if m not in S["methods"]:
                continue
            b = S["methods"][m]["N_thresholds_boot"][str(eps)]
            out.append(f"| {eps} | {m} | " + " | ".join(f"{b[k]['median']:.0f} [{b[k]['p2.5']:.0f}, {b[k]['p97.5']:.0f}]" for k in ["N50", "N80", "N90"]) + " |")
    out.append("")
    out.append("### Table R4. Paired differences (decision − reconstruction), mean over opponents with 95% CI\n")
    key = "NEURAL_DECISION_minus_NEURAL_RECON"
    if key in S["pairwise"]:
        out.append("| ε | quantity | " + " | ".join(f"N={n}" for n in N_BUDGETS) + " |")
        out.append("|---|---|" + "---|" * len(N_BUDGETS))
        for eps in EPSILONS:
            d = S["pairwise"][key][str(eps)]
            out.append(f"| {eps} | regret diff | " + " | ".join(f"{a:+.3f} [{lo:+.3f},{hi:+.3f}]" for a, lo, hi in zip(d["regret_diff_mean"], d["regret_diff_lo"], d["regret_diff_hi"])) + " |")
            if eps > 0:
                out.append(f"| {eps} | F diff | " + " | ".join(f"{a:+.2f} [{lo:+.2f},{hi:+.2f}]" for a, lo, hi in zip(d["frac_diff_mean"], d["frac_diff_lo"], d["frac_diff_hi"])) + " |")
    out.append("")
    out.append("### Table R5. Per-seed neural results (regret at ε=0.1)\n")
    out.append("| run | " + " | ".join(f"N={n}" for n in N_BUDGETS) + " | best step |")
    out.append("|---|" + "---|" * (len(N_BUDGETS) + 1))
    for m in S["methods"]:
        if m.startswith("NEURAL_") and "_s" in m:
            r = S["methods"][m]["regret"]["0.1"]["mean"]
            out.append(f"| {m} | " + " | ".join(f"{a:.3f}" for a in r) + " | |")
    out.append("")
    out.append("### Table R6. Per-family regret at ε=0.1 (mean) and N80\n")
    for fam, d in S["per_family"].items():
        out.append(f"**{fam}** (oracle gain at ε=0.1: {S['oracle_gain']['by_family'][fam]['0.1']:.3f})\n")
        out.append("| method | " + " | ".join(f"N={n}" for n in N_BUDGETS) + " | N50 | N80 | N90 |")
        out.append("|---|" + "---|" * (len(N_BUDGETS) + 3))
        for m in ORDER:
            if m in d:
                r = d[m]["regret"]["0.1"]["mean"]; nt = d[m]["N_thresholds"]["0.1"]
                out.append(f"| {m} | " + " | ".join(f"{a:.3f}" for a in r) + f" | {nt['N50']} | {nt['N80']} | {nt['N90']} |")
        out.append("")
    out.append("### Table R7. Safety audit (OpenSpiel C++ best response on every deployed strategy)\n")
    out.append("| method | strategies | LP failures | max e−ε | # e−ε > 1e−7 | 99.9% quantile of e−ε | max |fast − OpenSpiel| |")
    out.append("|---|---|---|---|---|---|---|")
    for m, s in S["safety"].items():
        out.append(f"| {m} | {s['n_strategies']} | {s['lp_failures']} | {s['max_violation']:.2e} | {s['n_violations_gt_1e-7']} | {s['violation_quantiles']['0.999']:.2e} | {s['max_abs_fast_minus_os']:.2e} |")
    out.append("")
    out.append("### Table R8. Prediction errors vs N (mean over held-out opponents)\n")
    out.append("| method | quantity | " + " | ".join(f"N={n}" for n in N_BUDGETS) + " |")
    out.append("|---|---|" + "---|" * len(N_BUDGETS))
    for m, p in S["prediction"].items():
        for k in ["q_err_mean", "g_nmse_mean", "g_raw_mean"]:
            if k in p:
                out.append(f"| {m} | {k} | " + " | ".join(f"{a:.3f}" for a in p[k]) + " |")
    out.append("")
    out.append("### Table R9. Effective latent dimension (participation ratio of z across test opponents) vs N\n")
    out.append("| run | " + " | ".join(f"N={n}" for n in N_BUDGETS) + " |")
    out.append("|---|" + "---|" * len(N_BUDGETS))
    for m, d in S["latent_dim"].items():
        out.append(f"| {m} | " + " | ".join(f"{a:.1f}" for a in d) + " |")
    out.append("")
    out.append("### Table R10. Strategic geometry (test opponents, ε=0.1, 20 000 pairs)\n")
    out.append(f"ρ(d_beh, d_resp) = {G['rho_beh_resp']:.3f}; ρ(d_beh reach-weighted, d_resp) = {G['rho_behw_resp']:.3f}; ρ(‖g−g'‖, d_resp) = {G['rho_g_resp']:.3f}; ρ(‖g−g'‖, d_beh) = {G['rho_g_beh']:.3f}\n")
    out.append("| representation | N | ρ(d_z, d_beh) | ρ(d_z, d_resp) | behavior-matched separation far/near [95% CI] | d_beh far / near | d_resp far / near |")
    out.append("|---|---|---|---|---|---|---|")
    for m, e in G["latent"].items():
        for N in ["20", "100", "500"]:
            s = e[N]["separation"]
            out.append(f"| {m} | {N} | {e[N]['rho_z_beh']:.3f} | {e[N]['rho_z_resp']:.3f} | {s['ratio']:.3f} [{s['lo']:.3f}, {s['hi']:.3f}] | {s['d_beh_far']:.3f} / {s['d_beh_near']:.3f} | {s['d_resp_far']:.3f} / {s['d_resp_near']:.3f} |")
    for lab, key in [("true g(q)", "separation_true_g"), ("d_beh itself (control)", "separation_behavior_itself")]:
        s = G[key]
        out.append(f"| {lab} | – | – | – | {s['ratio']:.3f} [{s['lo']:.3f}, {s['hi']:.3f}] | {s['d_beh_far']:.3f} / {s['d_beh_near']:.3f} | {s['d_resp_far']:.3f} / {s['d_resp_near']:.3f} |")
    out.append("")
    if "family_identifiability" in P:
        fi = P["family_identifiability"]
        out.append("### Table R11. Family identifiability from short histories (train-bank posterior MAP family accuracy)\n")
        out.append("| | " + " | ".join(f"N={n}" for n in N_BUDGETS) + " |")
        out.append("|---|" + "---|" * len(N_BUDGETS))
        out.append("| all families | " + " | ".join(f"{a:.2f}" for a in fi["map_accuracy_by_N"]) + " |")
        for fam, acc in fi["per_family_accuracy_by_N"].items():
            out.append(f"| {fam} | " + " | ".join(f"{a:.2f}" for a in acc) + " |")
        out.append("| posterior mass on true family | " + " | ".join(f"{a:.2f}" for a in fi["true_family_mass_by_N"]) + " |")
    print("\n".join(out))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "eval" / "test")
