"""Figures 1-5 for REPORT_LEDUC_NONSTATIONARY from outputs/nonstat/analysis.json and solve.npz.
Groups: classical (cool), forgetting (amber; WIN-EM solid, DISC-EM dash-dot; grid by lightness), neural (warm),
hybrid (green/aqua, thick), oracle (black/gray, dashed).  Every method also has its own marker."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import OUT
from .ns_eval import SWITCH_AT

D = OUT / "nonstat"; EPS = 0.10
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STY = {
    "FIXED-NE": ("#3d3d3a", "", "--", 1.4, "classical"), "BANK": ("#4a3aa7", "s", "-", 1.5, "classical"), "TAB-EM": ("#2a78d6", "o", "-", 1.6, "classical"),
    "WIN-EM W=25": ("#f5c64d", "<", "-", 1.3, "forgetting"), "WIN-EM W=50": ("#eda100", "<", "-", 1.8, "forgetting"), "WIN-EM W=100": ("#a86f00", "<", "-", 1.3, "forgetting"),
    "DISC-EM g=0.95": ("#f5c64d", ">", "-.", 1.3, "forgetting"), "DISC-EM g=0.98": ("#eda100", ">", "-.", 1.8, "forgetting"), "DISC-EM g=0.99": ("#a86f00", ">", "-.", 1.3, "forgetting"),
    "JAC-opp": ("#eb6834", "o", "-", 1.6, "neural"), "JAC-opp-W50": ("#e34948", "^", "-.", 1.6, "neural"), "DEC-889k": ("#e87ba4", "D", "-", 1.5, "neural"),
    "PRIOR-EM": ("#008300", "*", "-", 2.4, "hybrid"), "PRIOR-EM-WIN": ("#1baf7a", "s", "-", 2.4, "hybrid"),
    "POST-EM": ("#8a8984", "x", "--", 1.6, "oracle"), "POST-PRIOR-EM": ("#0b0b0b", "+", "--", 1.8, "oracle")}
MAIN = ["FIXED-NE", "BANK", "TAB-EM", "WIN-EM W=50", "DISC-EM g=0.98", "JAC-opp", "JAC-opp-W50", "DEC-889k", "PRIOR-EM", "PRIOR-EM-WIN", "POST-EM", "POST-PRIOR-EM"]
LABEL = lambda m: f"{m.replace('g=', 'γ=')}  [{STY[m][4]}]"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})


def line(ax, x, d, m, band=True):
    c, mk, ls, lw, _ = STY[m]
    ax.plot(x, d["mean"], color=c, marker=mk or None, ms=5 if mk not in ("*", "+", "x") else 7, ls=ls, lw=lw, label=LABEL(m), zorder=3 if STY[m][4] in ("hybrid", "oracle") else 2)
    if band:
        ax.fill_between(x, d["ci"][0], d["ci"][1], color=c, alpha=0.12, lw=0)


def fig_panels(r, kind, methods, fname, title):
    groups = ("contrasting", "random"); x = np.array(r[kind]["checkpoints"]) - (SWITCH_AT if kind == "SWITCH" else 0)
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for col, g in enumerate(groups):
        e = r[kind][g]
        for row, key in enumerate(("fraction", "regret")):
            ax = axs[row, col]
            for m in methods:
                if m in e[key] and not (key == "regret" and m == "FIXED-NE"):
                    line(ax, x, e[key][m], m)
            if kind == "SWITCH":
                ax.axvline(0, color="#d03b3b", lw=1.0, ls="--"); ax.text(2, ax.get_ylim()[1] if False else 0.0, "", fontsize=7)
            if key == "fraction":
                ax.axhline(0, color=MUTED, lw=0.8, ls=":"); ax.axhline(1, color=MUTED, lw=0.8, ls=":")
            ax.set_title(f"{g} pairs ({e['n_processes']} processes) — {'fraction of attainable safe gain' if key == 'fraction' else 'safe regret (chips)'}", fontsize=9, color=INK)
            ax.set_xlabel("hands since the switch" if kind == "SWITCH" else "hand t"); ax.set_ylabel("fraction (ratio of means)" if key == "fraction" else "chips")
    if kind == "SWITCH":
        for ax in axs.ravel():
            ax.text(3, 0.98, "switch", transform=ax.get_xaxis_transform(), fontsize=7, color="#d03b3b", va="top")
            ax.set_xticks([-100, 0, 50, 100, 200, 300])
        fne = r[kind]["contrasting"]["regret"]["FIXED-NE"]["mean"]
        axs[1, 0].text(0.02, 0.97, f"FIXED-NE regret ≈ {np.mean(fne):.2f} (not shown)", transform=axs[1, 0].transAxes, fontsize=7, va="top")
    h, l = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=11, color=INK); fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(D / fname, dpi=130); plt.close(fig)


def fig_recovery(r, fname):
    ms = [m for m in MAIN if m != "FIXED-NE"]; fig, axs = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
    for ax, g in zip(axs, ("contrasting", "random")):
        rec = r["SWITCH"][g]["recovery"]; xs = np.arange(len(ms)); cap = 400
        for k, (p, off, hatch) in enumerate(((0.5, -0.2, ""), (0.8, 0.2, "//"))):
            vals = np.array([rec[str(p)][m]["hands"] for m in ms]); lo = np.array([rec[str(p)][m]["ci"][0] for m in ms]); hi = np.array([rec[str(p)][m]["ci"][1] for m in ms])
            v = np.where(np.isfinite(vals), vals, cap)
            ax.bar(xs + off, v, width=0.38, color=[STY[m][0] for m in ms], alpha=0.85 if p == 0.5 else 0.5, hatch=hatch, edgecolor=[STY[m][0] for m in ms], lw=0.8,
                   label=f"{int(p*100)}% of own pre-switch fraction")
            err_lo = np.clip(v - np.where(np.isfinite(lo), lo, cap), 0, None); err_hi = np.clip(np.where(np.isfinite(hi), hi, cap) - v, 0, None)
            ax.errorbar(xs + off, v, yerr=[err_lo, err_hi], fmt="none", ecolor=INK, lw=0.8, capsize=2)
            for i, val in enumerate(vals):
                if not np.isfinite(val):
                    ax.text(xs[i] + off, cap * 1.05, "never", ha="center", fontsize=6.5, rotation=90, color=INK)
        ax.set_yscale("log"); ax.set_ylim(3, 700); ax.axhline(300, color=MUTED, lw=0.8, ls=":"); ax.text(len(ms) - 0.5, 310, "t = 500", fontsize=7, ha="right")
        ax.set_xticks(xs); ax.set_xticklabels([m.replace("g=", "γ=") for m in ms], rotation=40, ha="right", fontsize=7.5)
        ax.set_title(f"{g} pairs: hands after the switch to regain 50% (solid) / 80% (hatched) of own pre-switch fraction", fontsize=8.5, color=INK)
    axs[0].set_ylabel("hands after the switch (log)"); axs[0].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(D / fname, dpi=130); plt.close(fig)


def fig_tradeoff(r, fname):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.6))
    chains = [["TAB-EM", "WIN-EM W=100", "WIN-EM W=50", "WIN-EM W=25"], ["TAB-EM", "DISC-EM g=0.99", "DISC-EM g=0.98", "DISC-EM g=0.95"],
              ["PRIOR-EM", "PRIOR-EM-WIN"], ["JAC-opp", "JAC-opp-W50"], ["DEC-889k"], ["BANK"], ["POST-EM"], ["POST-PRIOR-EM"]]
    for ax, g in zip(axs, ("contrasting", "random")):
        e = r["SWITCH"][g]
        for ch in chains:
            xs = [e["pre"][m] for m in ch]; ys = [e["post_mean"][m] for m in ch]
            if len(ch) > 1:
                ax.plot(xs, ys, color=STY[ch[-1]][0], lw=0.9, ls=STY[ch[-1]][2], zorder=1)
            for m in ch:
                c, mk, _, _, grp = STY[m]; x, y = e["pre"][m], e["post_mean"][m]
                xe = np.array([[x - e["pre_ci"][m][0]], [e["pre_ci"][m][1] - x]]); ye = np.array([[y - e["post_mean_ci"][m][0]], [e["post_mean_ci"][m][1] - y]])
                ax.errorbar([x], [y], xerr=xe, yerr=ye, fmt=mk or "o", color=c, ms=8 if mk in ("*", "+", "x") else 6, mfc=c if grp != "oracle" else "none", lw=0.8, capsize=2, zorder=3)
                ax.annotate(m.replace("g=", "γ=").replace("WIN-EM ", "").replace("DISC-EM ", ""), (x, y), xytext=(4, 3), textcoords="offset points", fontsize=6.5, color=INK)
        ax.set_xlabel("stationary performance: fraction at t = 200 (pre-switch)"); ax.set_ylabel("recovery: mean fraction over the first 100 hands after the switch")
        ax.set_title(f"{g} pairs — forgetting trade-off (up = recovers faster, right = better before the switch)", fontsize=8.5, color=INK)
    fig.tight_layout(); fig.savefig(D / fname, dpi=130); plt.close(fig)


def fig_safety(r, fname):
    S = np.load(D / "solve.npz", allow_pickle=True); fig, axs = plt.subplots(1, 3, figsize=(18, 5.2)); rng = np.random.default_rng(0)
    ms = [str(m) for m in S["SWITCH_methods"]]; ax = axs[0]
    for i, m in enumerate(ms):
        d = S["SWITCH_expl"][i].ravel() - EPS
        if m in [str(x) for x in S["DRIFT_methods"]]:
            d = np.concatenate([d, S["DRIFT_expl"][list(S["DRIFT_methods"]).index(m)].ravel() - EPS])
        sub = d if len(d) <= 2000 else rng.choice(d, 2000, replace=False)
        ax.scatter(i + rng.uniform(-0.3, 0.3, len(sub)), sub, s=3, color=STY[m][0], alpha=0.35, lw=0); ax.plot([i - 0.35, i + 0.35], [d.max()] * 2, color=INK, lw=1.2)
    ax.set_yscale("symlog", linthresh=1e-12); ax.axhline(1e-7, color="#d03b3b", lw=1, ls="--"); ax.text(-0.4, 1.4e-7, "audit tolerance 1e−7", fontsize=7, va="bottom")
    ax.axhline(0, color=MUTED, lw=0.8); ax.set_xticks(range(len(ms))); ax.set_xticklabels([m.replace("g=", "γ=") for m in ms], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Expl(x) − ε (chips, symlog)"); ax.set_title("(a) Expl − ε of every deployed strategy (SWITCH + DRIFT; bar = max)", fontsize=8.5, color=INK)
    for ax, kind in zip(axs[1:], ("SWITCH", "DRIFT")):
        x = np.array(r[kind]["checkpoints"]) - (SWITCH_AT if kind == "SWITCH" else 0)
        for m in MAIN:
            if m == "FIXED-NE" or m not in r[kind]["harm_rate"]:
                continue
            c, mk, ls, lw, _ = STY[m]; ax.plot(x, r[kind]["harm_rate"][m], color=c, marker=mk or None, ls=ls, lw=lw, ms=5, label=LABEL(m))
        if kind == "SWITCH":
            ax.axvline(0, color="#d03b3b", lw=1.0, ls="--")
        ax.set_xlabel("hands since the switch" if kind == "SWITCH" else "hand t"); ax.set_ylabel("fraction of processes with u < u(FIXED-NE)")
        ax.set_title(f"({'b' if kind == 'SWITCH' else 'c'}) harm rate vs FIXED-NE — {kind}", fontsize=8.5, color=INK); ax.set_ylim(0, None)
    h, l = axs[1].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=6, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(D / fname, dpi=130); plt.close(fig)


def main():
    r = json.loads((D / "analysis.json").read_text()); allm = list(STY)
    fig_panels(r, "SWITCH", MAIN, "fig1_switch.png", "Fig 1 — SWITCH (q_A for hands 1–200, q_B after): fraction and regret vs hands since the switch (95% bootstrap CI over processes)")
    fig_panels(r, "SWITCH", allm, "fig1_switch_full.png", "Fig 1 (full grid) — SWITCH")
    fig_panels(r, "DRIFT", [m for m in MAIN if not m.startswith("POST")], "fig2_drift.png", "Fig 2 — DRIFT (linear per-infoset interpolation q_A → q_B over 500 hands)")
    fig_panels(r, "DRIFT", [m for m in allm if not m.startswith("POST")], "fig2_drift_full.png", "Fig 2 (full grid) — DRIFT")
    fig_recovery(r, "fig3_recovery.png"); fig_tradeoff(r, "fig4_forgetting_tradeoff.png"); fig_safety(r, "fig5_safety.png")


if __name__ == "__main__":
    main()
