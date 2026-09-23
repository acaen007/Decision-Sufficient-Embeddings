"""Figures 1-10 (PNG + PDF) with the raw plotted data saved next to each figure."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..common import N_BUDGETS, EPSILONS, save_json

COLORS = {"NEURAL_SAFE_REGRET": "#7b2cbf", "NEURAL_DECISION": "#1b6ca8", "NEURAL_RECON": "#d1495b", "TABULAR_EM_UNIFORM": "#8a8a8a",
          "TABULAR_EM_NASH": "#5c5c5c", "BANK_POSTERIOR": "#2a9d8f", "NASH": "#000000", "ORACLE_SAFE": "#e9a100"}
LABELS = {"NEURAL_SAFE_REGRET": "SAFE_REGRET (end-to-end)", "NEURAL_DECISION": "G_MSE (neural decision)", "NEURAL_RECON": "RECON (neural reconstruction)",
          "TABULAR_EM_UNIFORM": "tabular EM (uniform prior)", "TABULAR_EM_NASH": "tabular EM (Nash prior)",
          "BANK_POSTERIOR": "train-bank posterior", "NASH": "Nash blueprint", "ORACLE_SAFE": "safe oracle"}
MAIN = ["NEURAL_SAFE_REGRET", "NEURAL_DECISION", "NEURAL_RECON", "BANK_POSTERIOR", "TABULAR_EM_NASH", "TABULAR_EM_UNIFORM"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})


def _save(fig, fig_dir, name, data):
    fig.savefig(fig_dir / f"{name}.png", bbox_inches="tight")
    fig.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight")
    save_json(data, fig_dir / f"{name}_data.json")
    plt.close(fig)


def fig1_regret(summary, ed, fig_dir):
    fig, axes = plt.subplots(1, len(EPSILONS), figsize=(3.6 * len(EPSILONS), 3.2), sharex=True)
    data = {}
    for k, eps in enumerate(EPSILONS):
        ax = axes[k]
        for m in MAIN:
            if m not in summary["methods"]:
                continue
            r = summary["methods"][m]["regret"][str(eps)]
            ax.plot(N_BUDGETS, r["mean"], "-o", ms=3, color=COLORS[m], label=LABELS[m])
            ax.fill_between(N_BUDGETS, r["lo"], r["hi"], color=COLORS[m], alpha=0.15, lw=0)
            data.setdefault(str(eps), {})[m] = r
        nash = ed.G_oracle[:, k].mean()
        ax.axhline(nash, color=COLORS["NASH"], ls="--", lw=1, label=LABELS["NASH"])
        ax.axhline(0, color=COLORS["ORACLE_SAFE"], ls=":", lw=1.2, label=LABELS["ORACLE_SAFE"])
        data[str(eps)]["NASH"] = float(nash)
        ax.set_xscale("log"); ax.set_title(f"ε = {eps:.2f}"); ax.set_xlabel("hands observed N")
        if k == 0:
            ax.set_ylabel("safe response regret  V_ε(q) − u(x̂, q)  [chips/hand]")
        ax.set_xticks(N_BUDGETS); ax.set_xticklabels([str(n) for n in N_BUDGETS])
    axes[-1].legend(fontsize=7, loc="upper right")
    fig.suptitle("Figure 1: safe response regret vs hands observed (held-out opponents, paired bootstrap 95% CI)", y=1.02)
    _save(fig, fig_dir, "fig1_regret_vs_N", data)


def fig2_frac(summary, fig_dir):
    eps_list = [e for e in EPSILONS if e > 0]
    fig, axes = plt.subplots(1, len(eps_list), figsize=(3.6 * len(eps_list), 3.2), sharex=True, sharey=True)
    data = {}
    for k, eps in enumerate(eps_list):
        ax = axes[k]
        for m in MAIN:
            if m not in summary["methods"]:
                continue
            r = summary["methods"][m]["frac"][str(eps)]
            ax.plot(N_BUDGETS, r["mean"], "-o", ms=3, color=COLORS[m], label=LABELS[m])
            ax.fill_between(N_BUDGETS, r["lo"], r["hi"], color=COLORS[m], alpha=0.15, lw=0)
            data.setdefault(str(eps), {})[m] = r
        for t, ls in [(0.5, ":"), (0.8, "--"), (0.9, "-.")]:
            ax.axhline(t, color="gray", ls=ls, lw=0.8)
        ax.axhline(0, color="k", lw=0.8); ax.set_ylim(-0.6, 1.05)
        ax.set_xscale("log"); ax.set_title(f"ε = {eps:.2f}"); ax.set_xlabel("hands observed N")
        ax.set_xticks(N_BUDGETS); ax.set_xticklabels([str(n) for n in N_BUDGETS])
        if k == 0:
            ax.set_ylabel("fraction of oracle-safe gain recovered  F")
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Figure 2: fraction of oracle-safe exploitable value recovered vs N", y=1.02)
    _save(fig, fig_dir, "fig2_frac_vs_N", data)


def fig3_thresholds(summary, fig_dir):
    eps_list = [e for e in EPSILONS if e > 0]
    fig, axes = plt.subplots(1, len(eps_list), figsize=(3.6 * len(eps_list), 3.0), sharey=True)
    data = {}
    for k, eps in enumerate(eps_list):
        ax = axes[k]
        ms = [m for m in MAIN if m in summary["methods"]]
        w = 0.25
        for t_i, t in enumerate(["N50", "N80", "N90"]):
            vals = []
            for m in ms:
                v = summary["methods"][m]["N_thresholds"][str(eps)][t]
                vals.append(1000 if v == ">500" else v)
                data.setdefault(str(eps), {}).setdefault(m, {})[t] = v
            xs = np.arange(len(ms)) + (t_i - 1) * w
            bars = ax.bar(xs, vals, w, color=[COLORS[m] for m in ms], alpha=[0.45, 0.7, 1.0][t_i], label=t)
            for x, v in zip(xs, vals):
                ax.text(x, min(v, 700) * 1.05, ">500" if v == 1000 else str(v), ha="center", fontsize=6, rotation=90)
        ax.set_yscale("log"); ax.set_ylim(3, 1500); ax.set_xticks(np.arange(len(ms)))
        ax.set_xticklabels([LABELS[m].replace(" (", "\n(") for m in ms], fontsize=5.5, rotation=30, ha="right")
        ax.set_title(f"ε = {eps:.2f}")
        if k == 0:
            ax.set_ylabel("hands needed (log)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Figure 3: minimum tested N reaching 50 / 80 / 90 % of oracle-safe gain (>500 shown at top)", y=1.03)
    _save(fig, fig_dir, "fig3_N_thresholds", data)


def fig4_safety(summary, ed, fig_dir):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.0))
    data = {}
    ms = list(summary["safety"].keys())
    # (a) exploitability - eps distribution per eps (all methods pooled)
    ax = axes[0]
    for k, eps in enumerate(EPSILONS):
        viol = np.concatenate([ed.solve[m]["e_os"][:, :, k].ravel() - eps for m in ms])
        ax.hist(viol, bins=60, histtype="step", label=f"ε={eps:.2f}", log=True)
        data[f"viol_quantiles_eps{eps}"] = {q: float(np.quantile(viol, q)) for q in [0.5, 0.9, 0.99, 1.0]}
    ax.set_xlabel("audited exploitability − ε  [chips]"); ax.set_ylabel("count (log)"); ax.legend(fontsize=7)
    ax.set_title("(a) slack of the exact safety bound")
    ax = axes[1]
    maxv = [summary["safety"][m]["max_violation"] for m in ms]
    ax.bar(range(len(ms)), np.maximum(maxv, 1e-13), color="#5c5c5c"); ax.set_yscale("log")
    ax.axhline(1e-7, color="r", ls="--", lw=1, label="tolerance 1e-7"); ax.legend(fontsize=7)
    ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms, rotation=90, fontsize=6); ax.set_title("(b) max violation per method")
    data["max_violation"] = dict(zip(ms, maxv))
    ax = axes[2]
    nv = [summary["safety"][m]["n_violations_gt_1e-7"] for m in ms]
    ax.bar(range(len(ms)), nv, color="#d1495b"); ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms, rotation=90, fontsize=6)
    ax.set_ylim(0, max(1, max(nv) * 1.2)); ax.set_title("(c) # strategies with e(x) > ε + 1e-7")
    for i, v in enumerate(nv):
        ax.text(i, 0.05, str(v), ha="center", fontsize=7)
    data["n_violations"] = dict(zip(ms, nv))
    data["n_strategies_per_method"] = {m: summary["safety"][m]["n_strategies"] for m in ms}
    fig.suptitle("Figure 4: independent OpenSpiel best-response safety audit of every deployed strategy", y=1.03)
    _save(fig, fig_dir, "fig4_safety_audit", data)


def fig5_errors(summary, ed, fig_dir):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.0))
    data = {}
    groups = ed.neural_groups()
    def curve(name, key):
        if name in summary["prediction"]:
            return summary["prediction"][name][key]
        if name in groups:
            return np.mean([summary["prediction"][m][key] for m in groups[name] if key in summary["prediction"][m]], 0).tolist()
        return None
    for name in ["NEURAL_RECON", "TABULAR_EM_UNIFORM", "TABULAR_EM_NASH"]:
        c = curve(name, "q_err_mean")
        if c is not None:
            axes[0].plot(N_BUDGETS, c, "-o", ms=3, color=COLORS[name], label=LABELS[name]); data.setdefault("q_err", {})[name] = c
    axes[0].set_xscale("log"); axes[0].set_title("(a) behavioral error  RMS over rank infosets ||q̂−q||"); axes[0].legend(fontsize=7)
    for name in MAIN:
        c = curve(name, "g_nmse_mean")
        if c is not None:
            axes[1].plot(N_BUDGETS, c, "-o", ms=3, color=COLORS[name], label=LABELS[name]); data.setdefault("g_nmse", {})[name] = c
        c2 = curve(name, "g_raw_mean")
        if c2 is not None:
            axes[2].plot(N_BUDGETS, c2, "-o", ms=3, color=COLORS[name], label=LABELS[name]); data.setdefault("g_raw", {})[name] = c2
    axes[1].set_xscale("log"); axes[1].set_title("(b) standardized g error (NMSE, valid dims)"); axes[1].axhline(1, color="gray", ls=":", lw=0.8)
    axes[2].set_xscale("log"); axes[2].set_title("(c) raw g error ||ĝ − g(q)||₂"); axes[2].legend(fontsize=7)
    for ax in axes:
        ax.set_xlabel("hands observed N"); ax.set_xticks(N_BUDGETS); ax.set_xticklabels([str(n) for n in N_BUDGETS])
    fig.suptitle("Figure 5: behavioral reconstruction error and decision-vector error vs N (held-out opponents)", y=1.03)
    _save(fig, fig_dir, "fig5_prediction_errors", data)


def fig6_family(summary, fig_dir, eps=0.1):
    fams = list(summary["per_family"].keys())
    fig, axes = plt.subplots(2, len(fams), figsize=(3.4 * len(fams), 5.6), sharex=True)
    data = {}
    for f, fam in enumerate(fams):
        for m in MAIN:
            if m not in summary["per_family"][fam]:
                continue
            e = summary["per_family"][fam][m]
            r = e["regret"][str(eps)]; fr = e["frac"][str(eps)]
            axes[0, f].plot(N_BUDGETS, r["mean"], "-o", ms=3, color=COLORS[m], label=LABELS[m])
            axes[0, f].fill_between(N_BUDGETS, r["lo"], r["hi"], color=COLORS[m], alpha=0.15, lw=0)
            axes[1, f].plot(N_BUDGETS, fr["mean"], "-o", ms=3, color=COLORS[m], label=LABELS[m])
            axes[1, f].fill_between(N_BUDGETS, fr["lo"], fr["hi"], color=COLORS[m], alpha=0.15, lw=0)
            data.setdefault(fam, {})[m] = {"regret": r, "frac": fr, "N_thresholds": e["N_thresholds"][str(eps)]}
        axes[0, f].set_title(fam, fontsize=8); axes[0, f].set_xscale("log"); axes[1, f].set_xscale("log")
        axes[1, f].set_ylim(-0.6, 1.05); axes[1, f].axhline(0, color="k", lw=0.8)
        axes[1, f].set_xlabel("hands observed N"); axes[1, f].set_xticks(N_BUDGETS); axes[1, f].set_xticklabels([str(n) for n in N_BUDGETS])
    axes[0, 0].set_ylabel(f"safe regret (ε={eps})"); axes[1, 0].set_ylabel("fraction recovered F")
    axes[1, -1].legend(fontsize=6, loc="lower right")
    fig.suptitle(f"Figure 6: sample efficiency by opponent family (ε = {eps})", y=1.01)
    _save(fig, fig_dir, "fig6_by_family", data)


def fig7_geometry(geom, raw, fig_dir):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    data = {"summary": geom}
    ax = axes[0]
    ax.scatter(raw["d_beh"], raw["d_resp"], s=2, alpha=0.2, color="#5c5c5c")
    ax.set_xlabel("behavioral distance d_beh (uniform-infoset L2)"); ax.set_ylabel("response confusion cost d_resp (ε=0.1)")
    ax.set_title(f"(a) ρ = {geom['rho_beh_resp']:.2f}", fontsize=8)
    ax = axes[1]
    names = []; ratios = []; los = []; his = []
    for m, e in geom["latent"].items():
        s = e["N100"]["separation"]; names.append(m); ratios.append(s["ratio"]); los.append(s["lo"]); his.append(s["hi"])
    for lab, key in [("true g(q)", "separation_true_g"), ("d_beh (control)", "separation_behavior_itself")]:
        s = geom[key]; names.append(lab); ratios.append(s["ratio"]); los.append(s["lo"]); his.append(s["hi"])
    cols = [COLORS.get(m.rsplit("_s", 1)[0], "#5c5c5c") if m.startswith("NEURAL") else "#e9a100" for m in names]
    ax.bar(range(len(names)), ratios, color=cols, yerr=[np.array(ratios) - np.array(los), np.array(his) - np.array(ratios)], capsize=2)
    ax.axhline(1, color="k", lw=0.8); ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_title("(b) behavior-matched separation, N=100", fontsize=8); ax.set_ylabel("mean d_z(far) / mean d_z(near)")
    data["separation"] = dict(zip(names, zip(ratios, los, his)))
    ax = axes[2]
    for m, e in geom["latent"].items():
        c = COLORS.get(m.rsplit("_s", 1)[0], "#5c5c5c")
        ax.plot(N_BUDGETS, [e[str(N)]["rho_z_resp"] for N in N_BUDGETS], "-o", ms=3, color=c, label=f"{m}: ρ(d_z,d_resp)")
        ax.plot(N_BUDGETS, [e[str(N)]["rho_z_beh"] for N in N_BUDGETS], "--s", ms=3, color=c, label=f"{m}: ρ(d_z,d_beh)")
    ax.set_xscale("log"); ax.set_xlabel("hands observed N"); ax.set_ylabel("Spearman ρ"); ax.legend(fontsize=5)
    ax.set_title("(c) latent-distance correlations vs N", fontsize=8)
    fig.suptitle("Figure 7: strategic geometry diagnostics on held-out opponents", y=1.03)
    _save(fig, fig_dir, "fig7_geometry", data)


def fig8_latent_dim(summary, ed, fig_dir):
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    data = {}
    for m, dims in summary["latent_dim"].items():
        obj = ed.meta["methods"][m]["objective"]
        c = COLORS["NEURAL_" + obj.upper()]
        ax.plot(N_BUDGETS, dims, "-o", ms=3, color=c, alpha=0.8, label=m); data[m] = dims
    ax.set_xscale("log"); ax.set_xlabel("hands observed N"); ax.set_ylabel("participation ratio of z across opponents")
    ax.set_xticks(N_BUDGETS); ax.set_xticklabels([str(n) for n in N_BUDGETS]); ax.legend(fontsize=6)
    ax.set_title("Figure 8: effective latent dimension vs N (linear diagnostic)")
    _save(fig, fig_dir, "fig8_latent_dim", data)


def fig9_schematic(fig_dir):
    fig, ax = plt.subplots(figsize=(6, 4.2)); ax.axis("off")
    boxes = {"hands": (0.5, 0.93, "observed hands H_N\n(event tokens, no hidden cards)"),
             "enc": (0.5, 0.76, "hierarchical Transformer\nhand encoder → history encoder"),
             "z": (0.5, 0.61, "z ∈ R^128"),
             "q": (0.22, 0.44, "reconstruction decoder\nq̂(I,·) for every opponent infoset"),
             "g2": (0.78, 0.44, "decision head\nĝ ∈ R^1093"),
             "Ay": (0.22, 0.29, "y(q̂) → ĝ = A ŷ"),
             "lp": (0.5, 0.15, "exact ε-safe LP:  max ĝᵀx  s.t.  Ex=e, x≥0,\nmin_y xᵀAy ≥ v* − ε  (sequence-form dual)"),
             "x": (0.5, 0.03, "deployed strategy x  → audited by OpenSpiel best response")}
    for k, (x, y, t) in boxes.items():
        ax.text(x, y, t, ha="center", va="center", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.4", fc="#f4f4f4" if k not in ("q", "g2") else ("#fbe3e6" if k == "q" else "#dbe9f6"), ec="#666"))
    arrows = [("hands", "enc"), ("enc", "z"), ("z", "q"), ("z", "g2"), ("q", "Ay"), ("Ay", "lp"), ("g2", "lp"), ("lp", "x")]
    for a, b in arrows:
        xa, ya, _ = boxes[a]; xb, yb, _ = boxes[b]
        ax.annotate("", xy=(xb, yb + 0.045), xytext=(xa, ya - 0.045), arrowprops=dict(arrowstyle="->", color="#444"))
    ax.set_title("Figure 9: two objectives, one safe solver")
    _save(fig, fig_dir, "fig9_architecture", {"note": "schematic"})


def fig10_tokenized_hands(tab, tree, fig_dir, examples):
    from ..data.tokenizer import render_tokens
    fig, ax = plt.subplots(figsize=(11, 0.32 * sum(tab.length[t] + 3 for t in examples))); ax.axis("off")
    text = []
    for t in examples:
        text.append(f"observation type {t} (representative terminal {tab.representative[t]}):")
        text.append(render_tokens(tab.cat[t], tab.num[t], tab.length[t])); text.append("")
    ax.text(0, 1, "\n".join(text), family="monospace", fontsize=6.2, va="top")
    ax.set_title("Figure 10: tokenized Leduc hands as seen by the learner (opponent rank only at SHOWDOWN)", loc="left")
    _save(fig, fig_dir, "fig10_tokenized_hands", {"types": [int(t) for t in examples], "text": "\n".join(text)})
    return "\n".join(text)


def fig11_training_curves(run_dirs, fig_dir):
    """Validation loss (own objective) and validation g-NMSE (at N=500 and N=20) vs step for every run."""
    import json as _json
    from pathlib import Path as _P
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
    data = {}
    for name, rd in run_dirs.items():
        recs = [_json.loads(l) for l in open(_P(rd) / "log.jsonl")]
        vals = [r for r in recs if "val_loss" in r]
        steps = [r["step"] + 1 for r in vals]
        obj = "SAFE_REGRET" if "safe_regret" in name.lower() else ("DECISION" if "decision" in name.lower() else "RECON")
        c = COLORS["NEURAL_" + obj]
        ax = axes[0] if obj == "DECISION" else (axes[1] if obj == "RECON" else axes[3])
        ax.plot(steps, [r["val_loss"] for r in vals], "-", color=c, alpha=0.8, label=name)
        axes[2].plot(steps, [r["val_per_N"]["500"]["g_nmse"] for r in vals], "-", color=c, alpha=0.8, label=f"{name} N=500")
        axes[2].plot(steps, [r["val_per_N"]["20"]["g_nmse"] for r in vals], ":", color=c, alpha=0.6, label=f"{name} N=20")
        tr = [r for r in recs if "loss" in r]
        data[name] = {"val_steps": steps, "val_loss": [r["val_loss"] for r in vals],
                      "val_g_nmse_N500": [r["val_per_N"]["500"]["g_nmse"] for r in vals],
                      "val_g_nmse_N20": [r["val_per_N"]["20"]["g_nmse"] for r in vals],
                      "train_loss_smoothed": np.convolve([r["loss"] for r in tr], np.ones(50) / 50, mode="valid")[::50].tolist(),
                      "best_step": int(min(vals, key=lambda r: r["val_loss"])["step"]) + 1}
    axes[0].set_title("(a) G_MSE: val NMSE(g)", fontsize=8); axes[1].set_title("(b) RECON: val CE(q)", fontsize=8)
    axes[2].set_title("(c) val g-NMSE (solid N=500, dotted N=20)", fontsize=8)
    axes[3].set_title("(d) SAFE_REGRET: val exact-LP regret (subset, ε=0.1)", fontsize=8)
    for ax in axes:
        ax.set_xlabel("training step"); ax.legend(fontsize=5)
    fig.suptitle("Figure 11: training curves (validation opponents; 3 seeds per objective)", y=1.03)
    _save(fig, fig_dir, "fig11_training_curves", data)


def fig12_annealing(run_dirs, fig_dir):
    """Annealing diagnostics for the SAFE_REGRET sweep: tau, encoder gradient norm, determinism,
    active-set / LP-support changes, validation exact-LP and QP regret vs step."""
    import json as _json
    from pathlib import Path as _P
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.2))
    data = {}
    palette = ["#7b2cbf", "#1b6ca8", "#d1495b", "#2a9d8f", "#e9a100", "#5c5c5c"]
    for ci, (name, rd) in enumerate(run_dirs.items()):
        recs = [_json.loads(l) for l in open(_P(rd) / "log.jsonl")]
        c = palette[ci % len(palette)]
        steps = np.array([r["step"] + 1 for r in recs]); tau = np.array([r.get("tau", np.nan) for r in recs])
        gz = np.array([r.get("grad_norm_z", np.nan) for r in recs], dtype=float)
        genc = np.array([r.get("grad_norm_enc", np.nan) for r in recs], dtype=float)
        k = 50
        sm = lambda v: np.convolve(np.nan_to_num(v, nan=0.0), np.ones(k) / k, mode="valid")
        axes[0, 0].plot(steps, tau, color=c, label=name); axes[0, 0].set_yscale("log")
        axes[0, 1].plot(steps[k - 1:], sm(gz), color=c, label=name); axes[0, 1].set_yscale("log")
        axes[0, 2].plot(steps[k - 1:], sm(genc), color=c, label=name); axes[0, 2].set_yscale("log")
        vals = [r for r in recs if "val_loss" in r]
        vs = [r["step"] + 1 for r in vals]
        det = [r["val_per_N"].get("val_policy_det_frac", np.nan) for r in vals]
        ent = [r["val_per_N"].get("val_policy_entropy", np.nan) for r in vals]
        ach = [r["val_per_N"].get("probe_active_set_change_frac", np.nan) for r in vals]
        sch = [r["val_per_N"].get("probe_lp_support_change_frac", np.nan) for r in vals]
        lpr = [r["val_per_N"].get("val_lp_regret", np.nan) for r in vals]; qpr = [r["val_per_N"].get("val_qp_regret", np.nan) for r in vals]
        gn5 = [r["val_per_N"]["500"]["g_nmse"] for r in vals]
        axes[1, 0].plot(vs, det, "-o", ms=3, color=c, label=f"{name} determinism"); axes[1, 0].plot(vs, ent, ":", color=c)
        axes[1, 1].plot(vs, ach, "-o", ms=3, color=c, label=f"{name} QP active set"); axes[1, 1].plot(vs, sch, "--s", ms=3, color=c, label=f"{name} LP support")
        axes[1, 2].plot(vs, lpr, "-o", ms=3, color=c, label=f"{name} exact LP"); axes[1, 2].plot(vs, qpr, ":", color=c)
        data[name] = {"val_steps": vs, "tau_at_val": [float(tau[min(len(tau) - 1, v - 1)]) for v in vs], "det_frac": det, "entropy": ent,
                      "active_set_change": ach, "lp_support_change": sch, "val_lp_regret": lpr, "val_qp_regret": qpr, "val_g_nmse_N500": gn5,
                      "grad_norm_z_smoothed": sm(gz)[::100].tolist(), "grad_norm_enc_smoothed": sm(genc)[::100].tolist()}
    axes[0, 0].set_title("(a) regularization τ_t", fontsize=8); axes[0, 1].set_title("(b) ‖∂L/∂z‖ into the encoder (50-step mean)", fontsize=8)
    axes[0, 2].set_title("(c) encoder parameter gradient norm (pre-clip)", fontsize=8)
    axes[1, 0].set_title("(d) validation policies: deterministic fraction (solid), entropy (dotted)", fontsize=8)
    axes[1, 1].set_title("(e) fraction of active-set / LP-support entries changed between validations", fontsize=8)
    axes[1, 2].set_title("(f) validation regret: exact LP (solid), regularized QP (dotted)", fontsize=8)
    for ax in axes.ravel():
        ax.set_xlabel("training step"); ax.legend(fontsize=5)
    fig.suptitle("Figure 12: SAFE_REGRET annealing diagnostics (validation-only sweep)", y=1.01)
    _save(fig, fig_dir, "fig12_annealing", data)
