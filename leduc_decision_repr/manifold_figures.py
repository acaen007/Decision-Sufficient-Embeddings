"""Figures for REPORT_LEDUC_MANIFOLD from outputs/manifold/analysis.json and meta.json."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import OUT

D = OUT / "manifold"; FAMS = ["ID-REF", "NEAR", "FAR-ARCH", "FAR-CFR", "FAR-EXPL"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})


def main():
    r = json.loads((D / "analysis.json").read_text()); meta = json.loads((D / "meta.json").read_text()); cov = r["coverage"]; ref = r["references_N500"]
    # ---- Fig 1: ceilings vs achieved performance per family
    series = [("1-NN training opponent (discrete bank ceiling)", lambda f: cov["BANK-1NN"][f], "#4a3aa7", ""),
              ("JAC-opp encoder, N = 500 (achieved)", lambda f: ref["JAC-opp"][f], "#eb6834", ""),
              ("latent ceiling, inside training cloud", lambda f: cov["JAC-opp/full"][f], "#e34948", "//"),
              ("latent ceiling, unconstrained (not converged: lower bound)", lambda f: cov["JAC-opp/free"][f], "#e87ba4", ".."),
              ("TAB-EM, N = 500 (achieved)", lambda f: ref["TAB-EM"][f], "#2a78d6", ""),
              ("PRIOR-EM (JAC-opp), N = 500 (achieved)", lambda f: ref["PRIOR-EM (JAC-opp)"][f], "#008300", "")]
    fig, ax = plt.subplots(figsize=(14, 5.2)); x = np.arange(len(FAMS)); w = 0.13
    for i, (lab, get, c, h) in enumerate(series):
        vals = np.array([get(f)["fraction"] for f in FAMS]); lo = np.array([get(f)["ci"][0] for f in FAMS]); hi = np.array([get(f)["ci"][1] for f in FAMS])
        ax.bar(x + (i - 2.5) * w, vals, width=w * 0.92, color=c, alpha=0.85, hatch=h, edgecolor=c, label=lab)
        ax.errorbar(x + (i - 2.5) * w, vals, yerr=[vals - lo, hi - vals], fmt="none", ecolor=INK, lw=0.7, capsize=1.5)
    ax.axhline(0.9, color="#d03b3b", lw=1, ls="--"); ax.text(len(FAMS) - 0.5, 0.905, "decision bar 0.90", fontsize=7, ha="right", color=INK)
    ax.set_xticks(x); ax.set_xticklabels(FAMS); ax.set_ylim(0.5, 1.02); ax.set_ylabel("fraction of attainable safe gain (ratio of means, 95% CI)")
    ax.set_title("How much could the JAC-opp latent space earn with perfect inference? (ceilings vs what methods achieve at N = 500)", fontsize=10, color=INK)
    ax.legend(fontsize=7.5, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout(); fig.savefig(D / "fig1_ceilings.png", dpi=130); plt.close(fig)
    # ---- Fig 2: ceiling vs latent dimension k (subset) and cloud variance explained
    kg = r["k_grid_subset"]; ks = [2, 4, 8, 16, 32]; FC = {"ID-REF": "#2a78d6", "NEAR": "#eb6834", "FAR-ARCH": "#1baf7a", "FAR-CFR": "#eda100", "FAR-EXPL": "#e87ba4", "TRAIN": "#3d3d3a"}
    MK = {"ID-REF": "o", "NEAR": "s", "FAR-ARCH": "^", "FAR-CFR": "D", "FAR-EXPL": "v", "TRAIN": "*"}
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [2, 1]})
    ax = axs[0]
    for f in FAMS + ["TRAIN"]:
        ys = [kg[f"k{k}"][f] for k in ks]; ax.plot(ks, ys, color=FC[f], marker=MK[f], ms=6 if MK[f] != "*" else 9, lw=1.6, label=f)
        ax.plot([64], [kg["free"][f]], marker=MK[f], color=FC[f], ms=6 if MK[f] != "*" else 9, mfc="none")
    ax.set_xscale("log", base=2); ax.set_xticks(ks + [64]); ax.set_xticklabels([str(k) for k in ks] + ["free"]); ax.minorticks_off()
    ax.set_xlabel("latent dimensions allowed (top-k principal components of the training cloud; 'free' = unconstrained, hollow)")
    ax.set_ylabel("ceiling fraction (subset: 30 per family)"); ax.axhline(0.9, color="#d03b3b", lw=1, ls="--")
    ax.set_title("Ceiling saturates by k ≈ 8: the latent space is low-dimensional, but its ceiling is below 0.90 in-distribution", fontsize=9, color=INK)
    ax.legend(fontsize=8, frameon=False, ncol=3)
    ax = axs[1]; ve = meta["clouds"]["JAC-opp"]["var_explained"]; ax.plot(ks, [ve[str(k)] for k in ks], color="#3d3d3a", marker="o", lw=1.6)
    ve2 = meta["clouds"]["RECON-889k"]["var_explained"]; ax.plot(ks, [ve2[str(k)] for k in ks], color="#8a8984", marker="s", lw=1.2, ls="--")
    ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels(ks); ax.minorticks_off(); ax.set_ylim(0.4, 1.01)
    ax.set_xlabel("k"); ax.set_ylabel("variance of training latent cloud explained"); ax.set_title("training latent cloud (N = 500): JAC-opp solid, RECON-889k dashed", fontsize=9, color=INK)
    fig.tight_layout(); fig.savefig(D / "fig2_dimension.png", dpi=130); plt.close(fig)
    # ---- Fig 3: drift paths
    dp = r["drift_paths"]; lams = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
    for tag, c, ls, lab in (("full", "#e34948", "-", "fitted inside the cloud"), ("free", "#e87ba4", "-.", "fitted, unconstrained"), ("latent-line", "#3d3d3a", "--", "straight line in latent space between fitted endpoints")):
        axs[0].plot(lams, [dp[tag][str(l)]["median_rel_err"] for l in lams], color=c, ls=ls, marker="o", lw=1.6, label=lab)
        axs[1].plot(lams, [dp[tag][str(l)]["fraction"] for l in lams], color=c, ls=ls, marker="o", lw=1.6, label=lab)
    axs[0].set_ylabel("median relative g error ‖g(z) − g*‖ / ‖g* − ḡ‖"); axs[1].set_ylabel("ceiling fraction")
    for ax in axs:
        ax.set_xlabel("λ (behavioural interpolation q_A → q_B)")
    axs[0].set_title(f"40 DRIFT paths: distance to the manifold along the path (fitted-path tortuosity {dp['tortuosity']['median']:.2f})", fontsize=9, color=INK)
    axs[1].set_title("what the best latent point along the path would earn", fontsize=9, color=INK); axs[0].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(D / "fig3_drift_paths.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
