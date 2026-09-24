"""V3 figures for T1, T2, T4, T5. Each reads the saved analysis JSON (outputs/v3_t*_analysis.json) and writes figures/figV3_T*.{png,pdf} + _data.json."""
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import OUT, FIG, N_BUDGETS, save_json, load_json

FAMS = ["NASH_LOGIT_PERTURB", "NASH_RANDOM_MIX", "STRUCTURED_CORRELATED", "UNSTRUCTURED_DIRICHLET"]
STYLE = {"NEURAL_DEC133K": ("C0", "-", "o", "DECISION 133k"), "NEURAL_RECON": ("C1", "-", "s", "RECON 131k (V1)"),
         "NEURAL_DECISION": ("C0", "--", "o", "DECISION 889k (V1)"), "NEURAL_REC889K": ("C1", "--", "s", "RECON 889k"),
         "NEURAL_RECJAC889K": ("C2", "-", "^", "RECON-Jacobian-weighted 889k"), "NEURAL_RECREACH889K": ("C4", "-", "v", "RECON-reach-weighted 889k"),
         "BANK_POSTERIOR": ("k", ":", "d", "train-bank posterior"), "TABULAR_EM_UNIFORM": ("gray", "-.", "x", "tabular EM (uniform prior)"),
         "TABULAR_EM_NASH": ("silver", "-.", "+", "tabular EM (Nash prior)"),
         "NEURAL_SPO0_133K": ("C3", "-", "^", "SPO+ only 133k"), "NEURAL_SPO1_133K": ("C2", "-", "v", "MSE + 1.0·SPO+ 133k"), "NEURAL_SPO03_133K": ("C5", "-", "<", "MSE + 0.3·SPO+ 133k"),
         "NEURAL_COUNT133K": ("C6", "-", "p", "DECISION + count features 133k"), "HYB_BLEND": ("C8", "-", "*", "BLEND λ_N ĝ_net + (1−λ_N) ĝ_EM"),
         "HYB_PRIOR_EM": ("C9", "-", "h", "LEARNED-PRIOR EM (3-seed q̂ prior)"), "HYB_PRIOR_EM_S0": ("C9", "--", "h", "LEARNED-PRIOR EM (single-seed prior)"), "HYB_ENSREC": ("C1", ":", "s", "3-seed RECON ensemble, no EM"), "HYB_PRIOR_EM_889K": ("C9", "-.", "H", "LEARNED-PRIOR EM (RECON-889k prior)"), "NEURAL_DEC133K3K": ("C0", ":", "o", "DECISION 133k @3000 steps"), "NEURAL_REC131K3K": ("C1", ":", "s", "RECON 131k @3000 steps")}


def _curve(ax, name, c, fill=True):
    col, ls, mk, lab = STYLE.get(name, ("C7", "-", ".", name))
    m = np.array(c["mean"]); ax.plot(N_BUDGETS, m, ls=ls, marker=mk, ms=4, color=col, label=lab)
    if fill and "lo" in c:
        ax.fill_between(N_BUDGETS, c["lo"], c["hi"], color=col, alpha=0.12)


def _save(fig, name, data):
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=130); fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    save_json(data, FIG / f"{name}_data.json"); print("wrote", name)


def t1():
    d = load_json(OUT / "v3_t1_analysis.json"); cv = d["curves_eps0.1"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    ax = axes[0]
    for k in ["NEURAL_DEC133K", "NEURAL_RECON", "NEURAL_DECISION", "NEURAL_REC889K", "NEURAL_RECJAC889K", "NEURAL_RECREACH889K", "BANK_POSTERIOR", "TABULAR_EM_UNIFORM"]:
        if k in cv: _curve(ax, k, cv[k])
    ax.set_xscale("log"); ax.set_xlabel("N hands"); ax.set_ylabel("safe regret at ε=0.10 [chips/hand]"); ax.legend(fontsize=6); ax.set_title("(a) regret vs N, parameter-matched arms", fontsize=8)
    ax = axes[1]
    for key, lab, col in [("NEURAL_DEC133K_minus_NEURAL_RECON_eps0.1", "DEC133k − REC131k", "C0"), ("NEURAL_DECISION_minus_NEURAL_REC889K_eps0.1", "DEC889k − REC889k", "C1"),
                          ("NEURAL_RECJAC889K_minus_NEURAL_REC889K_eps0.1", "RECJAC889k − REC889k", "C2"), ("NEURAL_RECREACH889K_minus_NEURAL_REC889K_eps0.1", "RECREACH889k − REC889k", "C4")]:
        c = d["comparisons"].get(key)
        if c:
            ax.errorbar(N_BUDGETS, c["diff_mean"], yerr=[np.array(c["diff_mean"]) - np.array(c["diff_lo"]), np.array(c["diff_hi"]) - np.array(c["diff_mean"])], fmt="-o", ms=3, color=col, capsize=2, label=lab)
    ax.axhline(0, color="k", lw=0.8); ax.set_xscale("log"); ax.set_xlabel("N hands"); ax.set_ylabel("paired regret difference (neg = first better)"); ax.legend(fontsize=6); ax.set_title("(b) matched-pair differences, 95% paired bootstrap", fontsize=8)
    ax = axes[2]
    for r, info in d["runs"].items():
        if not r.startswith(("dec133k", "rec889k", "recjac", "recreach")): continue
        vc = np.array(info["val_curve"]); ax.plot(vc[:, 0], vc[:, 1], lw=1, label=f"{r} (stop {info['steps_run']})")
    ax.set_xlabel("step"); ax.set_ylabel("validation loss (head objective)"); ax.set_yscale("log"); ax.legend(fontsize=5, ncol=2); ax.set_title("(c) validation curves of V3 runs", fontsize=8)
    fig.suptitle("Figure V3-T1: confound fixes (parameter matching, convergence, reach-weighted reconstruction)", y=1.03)
    _save(fig, "figV3_T1_matched", {"curves": cv, "comparisons": {k: {kk: v for kk, v in c.items() if kk != "by_family"} for k, c in d["comparisons"].items()}})


def t2():
    d = load_json(OUT / "v3_t2_analysis.json"); from .data.datasets import load_population
    from .analysis.metrics import EvalData
    gaps = np.load(OUT / "t2_covariance" / "gaps.npz"); pop = load_population(); ed = EvalData(OUT / "eval" / "test", pop)
    gap_g = ed.per_opp(gaps["gap_g"]); j = N_BUDGETS.index(10)
    from .v3_analysis import group_regret
    Rd, _ = group_regret(ed, "NEURAL_DECISION", 2); Rr, _ = group_regret(ed, "NEURAL_REC889K", 2)
    if Rr is None: Rr, _ = group_regret(ed, "NEURAL_RECON", 2)
    adv = Rr - Rd
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    ax = axes[0]
    for f, fn in enumerate(FAMS):
        sel = ed.family == f; ax.scatter(gap_g[sel, j], adv[sel, j], s=8, alpha=0.6, label=fn)
    ax.set_xlabel("gap_g at N=10 (‖A(E[y]−y_{E[q]})‖)"); ax.set_ylabel("decision advantage at N=10 (recon − decision regret)"); ax.legend(fontsize=6)
    ax.set_title(f"(a) per-opponent gap vs advantage, ρ={d['spearman']['10']['pooled_gap_g']:.2f}", fontsize=8)
    ax = axes[1]
    rho_g = [d["spearman"][str(N)]["pooled_gap_g"] for N in N_BUDGETS]; rho_r = [d["spearman"][str(N)]["pooled_gap_reg"] for N in N_BUDGETS]
    ax.plot(N_BUDGETS, rho_g, "-o", ms=4, label="Spearman(gap_g, advantage)"); ax.plot(N_BUDGETS, rho_r, "-s", ms=4, label="Spearman(gap_reg, advantage)")
    for f, fn in enumerate(FAMS):
        ax.plot(N_BUDGETS, [d["spearman"][str(N)][fn]["gap_g"] for N in N_BUDGETS], ":", lw=0.8, label=f"gap_g, {fn}")
    ax.axhline(0, color="k", lw=0.8); ax.set_xscale("log"); ax.set_xlabel("N hands"); ax.set_ylabel("Spearman ρ"); ax.legend(fontsize=5); ax.set_title("(b) correlation vs N", fontsize=8)
    ax = axes[2]
    if "censoring" in d and "by_family" in d["censoring"]:
        bf = d["censoring"]["by_family"]; x = np.arange(len(FAMS)); w = 0.38; jj = N_BUDGETS.index(50)
        ax.bar(x - w / 2, [bf[f]["adv_censored"][jj] for f in FAMS], w, yerr=[[bf[f]["adv_censored"][jj] - bf[f]["adv_censored_lo"][jj] for f in FAMS], [bf[f]["adv_censored_hi"][jj] - bf[f]["adv_censored"][jj] for f in FAMS]], capsize=2, label="censored (standard)")
        ax.bar(x + w / 2, [bf[f]["adv_revealed"][jj] for f in FAMS], w, yerr=[[bf[f]["adv_revealed"][jj] - bf[f]["adv_revealed_lo"][jj] for f in FAMS], [bf[f]["adv_revealed_hi"][jj] - bf[f]["adv_revealed"][jj] for f in FAMS]], capsize=2, label="opponent card revealed")
        ax.set_xticks(x); ax.set_xticklabels([f[:12] for f in FAMS], fontsize=6); ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("decision advantage at N=50 (recon − decision)"); ax.legend(fontsize=6)
        ax.set_title("(c) censoring toggle, 3000-step matched 133k/131k models", fontsize=8)
    else:
        ax.text(0.5, 0.5, "censoring toggle not available", ha="center"); ax.set_axis_off()
    fig.suptitle("Figure V3-T2: covariance gap and censoring", y=1.03)
    _save(fig, "figV3_T2_covariance", {"spearman": d["spearman"], "censoring": d.get("censoring", {}).get("by_family")})


def t4():
    d = load_json(OUT / "v3_t4_analysis.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    for k, c in d["curves_eps0.1"].items(): _curve(ax, k, c)
    ax.set_xscale("log"); ax.set_xlabel("N hands"); ax.set_ylabel("safe regret at ε=0.10"); ax.legend(fontsize=6); ax.set_title("(a) regret vs N", fontsize=8)
    ax = axes[1]
    for k, c in d["g_nmse"].items():
        col, ls, mk, lab = STYLE.get(k, ("C7", "-", ".", k)); ax.plot(N_BUDGETS, c, ls=ls, marker=mk, ms=4, color=col, label=lab)
    ax.set_xscale("log"); ax.set_xlabel("N hands"); ax.set_ylabel("g-NMSE (test)"); ax.legend(fontsize=6); ax.set_title("(b) g-NMSE vs N", fontsize=8)
    fig.suptitle("Figure V3-T4: SPO+ vs MSE at matched 133k decision head", y=1.03)
    _save(fig, "figV3_T4_spo", d)


def t5():
    d = load_json(OUT / "v3_t5_analysis.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    for k, c in d["curves_eps0.1"].items(): _curve(ax, k, c, fill=(k in ("HYB_BLEND", "HYB_PRIOR_EM", "NEURAL_COUNT133K")))
    envk = [k for k in d["envelope"] if k.startswith("lower_envelope")]
    if envk: ax.plot(N_BUDGETS, d["envelope"][envk[0]], "k--", lw=1.5, label="lower envelope of bank/decision/EM")
    ax.set_xscale("log"); ax.set_xlabel("N hands"); ax.set_ylabel("safe regret at ε=0.10"); ax.legend(fontsize=6); ax.set_title("(a) hybrids vs envelope", fontsize=8)
    ax = axes[1]
    for k in ["HYB_BLEND", "HYB_PRIOR_EM", "NEURAL_COUNT133K"]:
        e = d["envelope"].get(k, {}).get("excess_ci")
        if e:
            col = STYLE[k][0]; ax.errorbar(N_BUDGETS, e["mean"], yerr=[np.array(e["mean"]) - np.array(e["lo"]), np.array(e["hi"]) - np.array(e["mean"])], fmt="-o", ms=3, color=col, capsize=2, label=STYLE[k][3])
    ax.axhline(0, color="k", lw=0.8); ax.set_xscale("log"); ax.set_xlabel("N hands"); ax.set_ylabel("regret − best single method at that N (paired)"); ax.legend(fontsize=6); ax.set_title("(b) excess over the per-N best baseline", fontsize=8)
    fig.suptitle("Figure V3-T5: empirical-Bayes hybrids", y=1.03)
    _save(fig, "figV3_T5_hybrids", d)


if __name__ == "__main__":
    {"t1": t1, "t2": t2, "t4": t4, "t5": t5}[sys.argv[1]]()
