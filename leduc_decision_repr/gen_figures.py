"""Figures 1-5 of REPORT_LEDUC_GENERALIZATION from outputs/gen/analysis.json (+ solve.npz for the Expl distribution).
Consistent method encoding across all figures: classical = cool hues, neural = warm hues, hybrid = green/aqua (thicker
lines); every method also has its own marker, so identity never rests on colour alone."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from .common import OUT

D = OUT / "gen"; EPS = 0.10
COL = {"FIXED-NE": "#3d3d3a", "BANK": "#4a3aa7", "TAB-EM": "#2a78d6",
       "DEC-889k": "#e87ba4", "RECON-889k": "#eda100", "RECON-JAC-global": "#e34948", "JAC-opp": "#eb6834",
       "PRIOR-EM (RECON-131k)": "#1baf7a", "PRIOR-EM (JAC-opp)": "#008300"}
MARK = {"FIXED-NE": "", "BANK": "s", "TAB-EM": "o", "DEC-889k": "D", "RECON-889k": "v", "RECON-JAC-global": "^", "JAC-opp": "o",
        "PRIOR-EM (RECON-131k)": "s", "PRIOR-EM (JAC-opp)": "*"}
GROUP = {"FIXED-NE": "classical", "BANK": "classical", "TAB-EM": "classical", "DEC-889k": "neural", "RECON-889k": "neural",
         "RECON-JAC-global": "neural", "JAC-opp": "neural", "PRIOR-EM (RECON-131k)": "hybrid", "PRIOR-EM (JAC-opp)": "hybrid"}
HIGHLIGHT = ["FIXED-NE", "TAB-EM", "BANK", "JAC-opp", "PRIOR-EM (RECON-131k)", "PRIOR-EM (JAC-opp)"]
FAMCOL = {"ID-REF": "#2a78d6", "NEAR": "#eb6834", "FAR-ARCH": "#1baf7a", "FAR-CFR": "#eda100", "FAR-EXPL": "#e87ba4", "NE": "#4a3aa7"}
FAMMARK = {"ID-REF": "o", "NEAR": "s", "FAR-ARCH": "^", "FAR-CFR": "D", "FAR-EXPL": "v", "NE": "*"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})


def style(m):
    hy = GROUP[m] == "hybrid"
    return dict(color=COL[m], marker=MARK[m] or None, ms=5 if MARK[m] != "*" else 8, lw=2.4 if hy else 1.6,
                ls="--" if m == "FIXED-NE" else "-", label=f"{m}  [{GROUP[m]}]", zorder=3 if hy else 2)


def band(ax, x, d, m):
    ax.plot(x, d["mean"] if "mean" in d else d["ratio_of_means"], **style(m))
    ax.fill_between(x, d["lo"], d["hi"], color=COL[m], alpha=0.14, lw=0)


def xaxis(ax, N):
    ax.set_xscale("log"); ax.set_xticks(N); ax.set_xticklabels(N); ax.minorticks_off(); ax.set_xlabel("N (hands observed)")


def fig_lines(r, key, fams, methods, fname, title, ylab, hlines=()):
    N = r["N"]; n = len(fams); cols = 3; legend_axis = n % cols != 0; rows = int(np.ceil((n + legend_axis) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(14, 4.0 * rows + (0 if legend_axis else 0.9)), squeeze=False); axs = axs.ravel()
    for i, f in enumerate(fams):
        ax = axs[i]; e = r["per_family"][f]
        data = e["fraction"] if key == "fraction" else (e["regret"] if f != "NE" else e["ne_loss"])
        top = None
        if key != "fraction" and f != "NE":                                   # FIXED-NE is far above everything: mark it off scale
            top = 1.12 * max(max(data[m]["hi"]) for m in methods if m != "FIXED-NE")
        for m in methods:
            if top is not None and m == "FIXED-NE" and min(data[m]["mean"]) > top:
                ax.plot([], [], **style(m))
                ax.text(0.02, 0.97, f"FIXED-NE regret = {data[m]['mean'][0]:.2f} (off scale)", transform=ax.transAxes, fontsize=7.5, va="top", color=INK)
                continue
            band(ax, N, data[m], m)
        if top is not None:
            ax.set_ylim(0, top)
        for h, lab in hlines:
            ax.axhline(h, color=MUTED, lw=0.9, ls=":", zorder=1)
        if key != "fraction" and f == "NE":
            ax.axhline(EPS, color="#d03b3b", lw=1.0, ls="--"); ax.text(N[0], EPS, " ε = 0.10 (max allowed loss)", color=INK, fontsize=7, va="bottom")
        xaxis(ax, N)
        excl = e.get("n_excluded_headroom", 0)
        sub = f"{e['n_opponents']} opponents" + (f", {excl} excluded (headroom < 0.01)" if key == "fraction" else "") + (" (loss v* − u)" if (key != "fraction" and f == "NE") else "")
        ax.set_title(f"{f}\n{sub}", fontsize=9, color=INK); ax.set_ylabel(ylab if (key != "fraction" or True) else "")
    h, l = axs[0].get_legend_handles_labels()
    for ax in axs[n:]:
        ax.axis("off")
    if legend_axis:
        axs[n].legend(h, l, loc="center", fontsize=8, frameon=False, title="method  [group]", title_fontsize=8)
        fig.suptitle(title, fontsize=11, color=INK); fig.tight_layout()
    else:
        fig.legend(h, l, loc="lower center", ncol=min(len(l), 5), fontsize=8, frameon=False)
        fig.suptitle(title, fontsize=11, color=INK); fig.tight_layout(rect=(0, 0.07 if len(l) <= 5 else 0.1, 1, 1))
    fig.savefig(D / fname, dpi=130); plt.close(fig)


def fig_heat(r, fname):
    fams = ["ID-REF", "NEAR", "FAR-ARCH", "FAR-CFR", "FAR-EXPL"]; methods = r["methods"]; N = r["N"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"wspace": 0.05})
    vals = {Nq: np.array([[r["per_family"][f]["fraction"][m]["ratio_of_means"][N.index(Nq)] for f in fams] for m in methods]) for Nq in (20, 500)}
    vmin = min(-1.0, min(v.min() for v in vals.values())); norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=1.0)
    for ax, Nq in zip(axs, (20, 500)):
        V = vals[Nq]; im = ax.imshow(V, cmap="RdBu", norm=norm, aspect="auto")
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                c = im.cmap(norm(V[i, j])); lum = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
                ax.text(j, i, f"{V[i, j]:.2f}", ha="center", va="center", fontsize=8, color="#ffffff" if lum < 0.5 else INK)
        ax.set_xticks(range(len(fams))); ax.set_xticklabels(fams, fontsize=8); ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([f"{m}  [{GROUP[m][0].upper()}]" for m in methods] if Nq == 20 else [], fontsize=8); ax.grid(False)
        ax.set_title(f"fraction of attainable safe gain, N = {Nq}", fontsize=10, color=INK)
    cb = fig.colorbar(im, ax=axs, shrink=0.8); cb.set_label("0 = best equilibrium for this opponent, 1 = oracle safe exploit; < 0 = worse than best equilibrium", fontsize=7)
    fig.savefig(D / fname, dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_hybrid(r, fname):
    N = r["N"]; fig, axs = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, h in zip(axs, ["PRIOR-EM (RECON-131k)", "PRIOR-EM (JAC-opp)"]):
        ax.axhline(0, color=MUTED, lw=0.9)
        for f, d in r["hybrid_minus_tabem"][h].items():
            ax.plot(N, d["mean"], color=FAMCOL[f], marker=FAMMARK[f], ms=5 if FAMMARK[f] != "*" else 8, lw=1.6, label=f)
            ax.fill_between(N, d["lo"], d["hi"], color=FAMCOL[f], alpha=0.12, lw=0)
        xaxis(ax, N); ax.set_title(f"{h} − TAB-EM (regret, chips; below 0 = learned prior helps)", fontsize=9, color=INK)
    axs[0].set_ylabel("regret difference (chips)"); axs[1].legend(fontsize=8, frameon=False, title="opponent family", title_fontsize=8)
    fig.tight_layout(); fig.savefig(D / fname, dpi=130); plt.close(fig)


def fig_safety(r, fname):
    N = r["N"]; methods = r["methods"]; S = np.load(D / "solve.npz", allow_pickle=True)
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.0))
    ax = axs[0]; e = r["per_family"]["NE"]["ne_loss"]
    for m in methods:
        band(ax, N, e[m], m)
    ax.axhline(EPS, color="#d03b3b", lw=1.0, ls="--"); ax.text(N[0], EPS * 1.01, " ε = 0.10", fontsize=7, va="bottom", color=INK)
    ax.axhline(0, color=MUTED, lw=0.8); xaxis(ax, N); ax.set_ylabel("loss v* − u (chips)")
    ax.set_title("(a) loss against Nash opponents (5 × 20 streams)", fontsize=9, color=INK)
    ax = axs[1]; rng = np.random.default_rng(0)
    Xs = [np.array([float(S["expl_fixed_ne"])])] + [S["expl"][i].ravel() for i in range(len(S["methods"]))]
    for i, (m, x) in enumerate(zip(methods, Xs)):
        d = x - EPS; sub = d if len(d) <= 3000 else rng.choice(d, 3000, replace=False)
        ax.scatter(i + rng.uniform(-0.28, 0.28, len(sub)), sub, s=3, color=COL[m], alpha=0.35, lw=0)
        ax.plot([i - 0.35, i + 0.35], [d.max()] * 2, color=INK, lw=1.2)
    ax.set_yscale("symlog", linthresh=1e-12); ax.axhline(1e-7, color="#d03b3b", lw=1.0, ls="--")
    ax.text(-0.4, 1.4e-7, "audit tolerance: Expl − ε ≤ 1e−7", fontsize=7, color=INK, va="bottom")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("Expl(x) − ε (chips, symlog)")
    ax.set_title("(b) exact exploitability − ε of every deployed strategy (all families; bar = max)\n"
                 "every modelling strategy has Expl = ε to within 2e−10; FIXED-NE has Expl ≈ 0", fontsize=8.5, color=INK)
    ax = axs[2]; hr = r["harm_rate_pooled_nonNE"]
    for m in methods[1:]:
        ax.plot(N, hr[m], **style(m))
    xaxis(ax, N); ax.set_ylabel("fraction of (opponent, N) with u < u(FIXED-NE)"); ax.set_ylim(0, 0.2)
    ax.set_title("(c) harm rate vs FIXED-NE (5 non-NE families pooled)", fontsize=9, color=INK)
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(D / fname, dpi=130); plt.close(fig)


def main():
    r = json.loads((D / "analysis.json").read_text()); fams5 = ["ID-REF", "NEAR", "FAR-ARCH", "FAR-CFR", "FAR-EXPL"]
    for tag, meth in (("", HIGHLIGHT), ("_full", r["methods"])):
        fig_lines(r, "fraction", fams5, meth, f"fig1_fraction{tag}.png", "Fig 1 — fraction of attainable safe gain (ε = 0.10; 95% bootstrap CI over opponents)",
                  "fraction (ratio of means)", hlines=((0, "best equilibrium"), (1, "oracle")))
        fig_lines(r, "regret", fams5 + ["NE"], meth, f"fig2_regret{tag}.png", "Fig 2 — raw safe regret V_ε(g) − u (chips; NE panel: loss v* − u)", "chips")
    fig_heat(r, "fig3_heatmaps.png"); fig_hybrid(r, "fig4_hybrid_vs_tabem.png"); fig_safety(r, "fig5_safety.png")


if __name__ == "__main__":
    main()
