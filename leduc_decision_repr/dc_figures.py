"""Figures for REPORT_LEDUC_DECISION_COMPRESSION from outputs/dcomp/{oracle.json, analysis.json}.
Encoding: each learned arm shares a hue with its oracle partition analogue: behaviour (RECON / BEH) blue, g-variance
(G-MSE / GVAR) green, decision (REGRET / DEC) indigo; JAC-OPP orange.  Learned arms solid, oracle partitions dashed."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .dc_common import D, EPS_LIST, OOD_FAMS
from .dc_ae import ARMS, DIMS

INK, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e4e3df"
AC = {"RECON": "#2a78d6", "JAC-OPP": "#eb6834", "G-MSE": "#1baf7a", "REGRET": "#4a3aa7"}
AM = {"RECON": "o", "JAC-OPP": "s", "G-MSE": "^", "REGRET": "D"}
PC = {"BEH": "#2a78d6", "GVAR": "#1baf7a", "DEC": "#4a3aa7"}
PL = {"BEH": "BEH partition (behaviour k-means)", "GVAR": "GVAR partition (g k-means)", "DEC": "DEC partition (regret Lloyd)"}
KS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.6,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})


def main():
    orc = json.loads((D / "oracle.json").read_text())["partitions"]; r = json.loads((D / "analysis.json").read_text()); tb = r["partB"]
    bits = np.log2(KS); ball = np.log2(1200)
    # ---- Fig 1: oracle partitions
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8))
    for t in ("BEH", "GVAR", "DEC"):
        y = [orc[f"{t}_{K}"]["test"]["frac_0.1"] for K in KS]; lo = [orc[f"{t}_{K}"]["test"]["frac_0.1_ci"][0] for K in KS]; hi = [orc[f"{t}_{K}"]["test"]["frac_0.1_ci"][1] for K in KS]
        axs[0].plot(bits, y, color=PC[t], ls="--", marker="o", lw=1.8, label=PL[t]); axs[0].fill_between(bits, lo, hi, color=PC[t], alpha=0.12, lw=0)
        axs[0].plot([ball], [orc[f"{t}_all"]["test"]["frac_0.1"]], color=PC[t], marker="*", ms=11)
        axs[1].plot(bits, [orc[f"{t}_{K}"]["test"]["beh_fraction"] for K in KS], color=PC[t], ls="--", marker="o", lw=1.8, label=PL[t])
        axs[1].plot([ball], [orc[f"{t}_all"]["test"]["beh_fraction"]], color=PC[t], marker="*", ms=11)
    axs[0].axhline(0.9, color="#d03b3b", lw=0.9, ls=":"); axs[0].set_ylabel("test decision fraction (ε = 0.10, 95% CI)")
    axs[1].set_ylabel("test behavioural fraction (hand-distribution KL)")
    for ax in axs:
        ax.set_xticks(list(bits) + [ball]); ax.set_xticklabels([f"{K}\n({int(b)} bits)" for K, b in zip(KS, bits)] + ["1-NN\n(★)"], fontsize=7.5)
        ax.set_xlabel("cells K in the partition of opponent space")
    axs[0].set_title("How many cells do safe decisions need? (oracle partitions, 300 test opponents)", fontsize=9.5, color=INK)
    axs[1].set_title("…and how much behaviour do the same cells keep?", fontsize=9.5, color=INK); axs[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(D / "fig1_oracle_partitions.png", dpi=130); plt.close(fig)
    # ---- Fig 2: autoencoder arms vs d
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8)); x = np.log2(DIMS)
    for a in ARMS:
        y = [tb[f"{a}_d{d}"]["test_frac_0.1"] for d in DIMS]; lo = [tb[f"{a}_d{d}"]["test_frac_0.1_ci"][0] for d in DIMS]; hi = [tb[f"{a}_d{d}"]["test_frac_0.1_ci"][1] for d in DIMS]
        sd = np.array([tb[f"{a}_d{d}"]["test_frac_0.1_seeds"] for d in DIMS])
        axs[0].plot(x, y, color=AC[a], marker=AM[a], lw=2, label=a); axs[0].fill_between(x, lo, hi, color=AC[a], alpha=0.10, lw=0)
        axs[0].vlines(x, sd.min(1), sd.max(1), color=AC[a], lw=3, alpha=0.35)
        axs[1].plot(x, [tb[f"{a}_d{d}"]["test_beh"] for d in DIMS], color=AC[a], marker=AM[a], lw=2, label=a)
        axs[2].plot(x, [tb[f"{a}_d{d}"]["test_gR2"] for d in DIMS], color=AC[a], marker=AM[a], lw=2, label=a)
    axs[0].axhline(0.9, color="#d03b3b", lw=0.9, ls=":")
    for ax, t, yl in zip(axs, ("decision value kept (ε = 0.10; band = CI, bar = seed range)", "behaviour kept (observable hand distribution)", "g reconstructed (R²)"),
                         ("test decision fraction", "test behavioural fraction", "test g R²")):
        ax.set_xticks(x); ax.set_xticklabels(DIMS); ax.set_xlabel("latent dimension d"); ax.set_ylabel(yl); ax.set_title(t, fontsize=9.5, color=INK)
    axs[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("Policy autoencoder q → z → q̂: same network, four losses, 3 seeds each (300 test opponents)", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(D / "fig2_autoencoder.png", dpi=130); plt.close(fig)
    # ---- Fig 3: the trade-off plane
    fig, ax = plt.subplots(figsize=(9.5, 7))
    for a in ARMS:
        bx = [tb[f"{a}_d{d}"]["test_beh"] for d in DIMS]; by = [tb[f"{a}_d{d}"]["test_frac_0.1"] for d in DIMS]
        ax.plot(bx, by, color=AC[a], marker=AM[a], lw=1.8, ms=6, label=f"{a} (autoencoder, d = 2…64)")
        for d, xx, yy in zip(DIMS, bx, by):
            ax.annotate(str(d), (xx, yy), xytext=(4, -9), textcoords="offset points", fontsize=7, color=AC[a])
    for t in ("BEH", "DEC"):
        bx = [orc[f"{t}_{K}"]["test"]["beh_fraction"] for K in KS[1:]]; by = [orc[f"{t}_{K}"]["test"]["frac_0.1"] for K in KS[1:]]
        ax.plot(bx, by, color=PC[t], ls="--", lw=1.1, marker=".", alpha=0.8, label=f"{PL[t]}, K = 2…512")
        for K, xx, yy in zip(KS[1:], bx, by):
            if K in (4, 16, 64, 256):
                ax.annotate(f"K{K}", (xx, yy), xytext=(3, 4), textcoords="offset points", fontsize=6.5, color=PC[t])
    ax.axhline(0.9, color="#d03b3b", lw=0.8, ls=":"); ax.plot([0, 1], [0, 1], color=MUTED, lw=0.6, ls=":")
    ax.set_xlabel("behaviour kept (test behavioural fraction)"); ax.set_ylabel("decision value kept (test decision fraction, ε = 0.10)")
    ax.set_title("Decisions vs behaviour: points above the diagonal keep more decision value than behaviour", fontsize=9.5, color=INK)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right"); fig.tight_layout(); fig.savefig(D / "fig3_tradeoff.png", dpi=130); plt.close(fig)
    # ---- Fig 4: cost side at d = 8 (eps transfer, OOD families)
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.6)); w = 0.2
    for i, a in enumerate(ARMS):
        axs[0].bar(np.arange(3) + (i - 1.5) * w, [tb[f"{a}_d8"][f"test_frac_{e}"] for e in EPS_LIST], width=w * 0.92, color=AC[a], label=a)
        vals = [tb[f"{a}_d8"]["test_frac_0.1"]] + [tb[f"{a}_d8"][f"ood_frac_{f}"] for f in OOD_FAMS]
        axs[1].bar(np.arange(5) + (i - 1.5) * w, vals, width=w * 0.92, color=AC[a], label=a)
    axs[0].set_xticks(range(3)); axs[0].set_xticklabels([f"ε = {e}" for e in EPS_LIST]); axs[0].set_ylabel("test decision fraction")
    axs[0].set_title("d = 8: transfer across safety budgets (REGRET's bank is built at ε = 0.10)", fontsize=9.5, color=INK)
    axs[1].set_xticks(range(5)); axs[1].set_xticklabels(["test (ID)"] + OOD_FAMS); axs[1].set_ylabel("decision fraction (ε = 0.10)")
    axs[1].set_title("d = 8: in-distribution vs unseen opponent families", fontsize=9.5, color=INK)
    for ax in axs:
        ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(); fig.savefig(D / "fig4_cost_side.png", dpi=130); plt.close(fig)
    # ---- Fig 5: decision-equivalence structure
    pc = r["partC"]; m = pc["models"]; fig, axs = plt.subplots(1, 2, figsize=(14, 4.6))
    for a in ARMS:
        axs[0].plot(x, [m[f"{a}_d{d}"]["SR"] for d in DIMS], color=AC[a], marker=AM[a], lw=2, label=a)
        axs[1].plot(x, [m[f"{a}_d{d}"]["partial_dec_given_beh"] for d in DIMS], color=AC[a], marker=AM[a], lw=2, label=a)
    axs[0].axhline(1, color=MUTED, lw=0.9, ls=":")
    axs[0].set_ylabel("SR = latent distance of decision-equivalent/behaviourally-far pairs\n÷ latent distance of behaviourally-close/decision-different pairs")
    axs[0].set_title(f"< 1: the code groups opponents by decision ({pc['n_DEQ_BF']} vs {pc['n_BC_DD']} test pairs)", fontsize=9.5, color=INK)
    axs[1].set_ylabel("partial Spearman ρ(latent distance, cross-regret | behavioural distance)"); axs[1].axhline(0, color=MUTED, lw=0.9, ls=":")
    axs[1].set_title("how much the latent geometry tracks decisions beyond behaviour", fontsize=9.5, color=INK)
    for ax in axs:
        ax.set_xticks(x); ax.set_xticklabels(DIMS); ax.set_xlabel("latent dimension d"); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(D / "fig5_equivalence.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
