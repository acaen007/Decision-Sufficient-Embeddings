"""Figures 1-4 for REPORT_LEDUC_CHANGEDETECT from outputs/cpd/analysis.json.
Encoding follows the non-stationary study: hybrid PRIOR-EM family green/aqua (thick), forgetting amber (dash-dot),
detectors blue (practical, solid) / gray (oracle detector, dotted), oracle reset black dashed."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import OUT
from .ns_eval import SWITCH_AT, DR_T
from .cpd_eval import GRID, cfg_name

D = OUT / "cpd"; INK, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e4e3df"
STY = {"PRIOR-EM": ("#008300", "*", "-", 2.2, "full history"), "PRIOR-EM-WIN": ("#1baf7a", "s", "-", 2.2, "window 50"),
       "DISC-PRIOR-EM g=0.98": ("#eda100", ">", "-.", 1.8, "discount 0.98"), "DISC-PRIOR-EM g=0.95": ("#f5c64d", ">", "-.", 1.3, "discount 0.95, grid"),
       "CPD-PRIOR-EM": ("#2a78d6", "o", "-", 2.6, "CUSUM change-detect + reset"), "BOCPD-PRIOR-EM": ("#4a3aa7", "v", "-", 2.2, "Bayesian change-point mixture"), "KNOWN-CUSUM-RESET": ("#8a8984", "D", ":", 1.8, "oracle detector (knows q_A, q_B)"),
       "POST-PRIOR-EM": ("#0b0b0b", "+", "--", 1.8, "oracle reset (knows switch time)")}
LAB = lambda m: f"{m.replace('g=', 'γ=')} — {STY[m][4]}"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.6,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})


def line(ax, x, y, lo, hi, m, band=True):
    c, mk, ls, lw, _ = STY[m]
    ax.plot(x, y, color=c, marker=mk, ms=7 if mk in ("*", "+") else 4.5, ls=ls, lw=lw, label=LAB(m), zorder=3 if m in ("CPD-PRIOR-EM", "POST-PRIOR-EM") else 2)
    if band:
        ax.fill_between(x, lo, hi, color=c, alpha=0.10, lw=0)


def main():
    r = json.loads((D / "analysis.json").read_text()); MS = [m for m in STY if m in r["SWITCH"]["contrasting"]["fine_grid"]["methods"]]
    # ---- Fig 1: SWITCH fraction vs hands since the switch (fine grid), contrasting / random, full and zoom
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), gridspec_kw={"width_ratios": [1.2, 1]})
    for row, g in enumerate(("contrasting", "random")):
        e = r["SWITCH"][g]["fine_grid"]; x = np.array(e["checkpoints"]) - SWITCH_AT
        for col, (xlim, band) in enumerate((((-105, 305), False), ((-2, 82), True))):
            ax = axs[row, col]
            for m in MS:
                d = e["methods"][m]; line(ax, x, d["fraction"], d["fraction_ci"][0], d["fraction_ci"][1], m, band=band)
            ax.axvline(0, color=MUTED, lw=0.8, ls=":"); ax.set_xlim(*xlim); ax.set_ylim(0.2, 0.95)
            ax.set_ylabel("fraction of attainable safe gain (pooled)") if col == 0 else None
            ax.set_title(f"{g} pairs ({e['n_processes']} processes): " + ("whole process" if col == 0 else "first 80 hands after the switch, 95% CI bands"), fontsize=9.5, color=INK)
            if col == 1:
                pre = e["methods"]["POST-PRIOR-EM"]["pre"]; ax.axhline(0.8 * pre, color=INK, lw=0.7, ls=(0, (1, 3)))
                ax.text(81, 0.8 * pre + 0.008, "0.8 × oracle's pre-switch level", ha="right", fontsize=7.5, color=MUTED)
    for ax in axs[1]:
        ax.set_xlabel("hands since the switch (switch after hand 200)")
    axs[0, 0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("Change detection + reset vs forgetting and the oracle reset (ε = 0.10, test SWITCH processes)", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(D / "fig1_switch.png", dpi=130); plt.close(fig)
    # ---- Fig 2: R80 with CIs (contrasting; original and fine grid) with decision bands
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    order = [m for m in ["POST-PRIOR-EM", "KNOWN-CUSUM-RESET", "CPD-PRIOR-EM", "BOCPD-PRIOR-EM", "PRIOR-EM-WIN", "DISC-PRIOR-EM g=0.98", "DISC-PRIOR-EM g=0.95", "PRIOR-EM"] if m in MS]
    for ax, gname in zip(axs, ("original_grid", "fine_grid")):
        e = r["SWITCH"]["contrasting"][gname]["methods"]; ro = e["POST-PRIOR-EM"]["R80"]; cap = 120
        ax.axvspan(0, 1.5 * ro, color="#1baf7a", alpha=0.10, lw=0); ax.axvspan(1.5 * ro, 2.0 * ro, color="#8a8984", alpha=0.12, lw=0)
        ax.axvspan(2.0 * ro, cap, color="#e34948", alpha=0.07, lw=0)
        for i, m in enumerate(order):
            v = e[m]["R80"]; lo, hi = e[m]["R80_ci"]; c = STY[m][0]; y = len(order) - 1 - i
            if v > cap:
                ax.barh(y, cap, color=c, alpha=0.35, height=0.6); ax.text(cap - 1, y, f"{v:.0f} (off scale)", va="center", ha="right", fontsize=8, color=INK)
            else:
                ax.barh(y, v, color=c, alpha=0.85, height=0.6); ax.errorbar(v, y, xerr=[[v - lo], [min(hi, cap) - v]], fmt="none", ecolor=INK, lw=0.8, capsize=2.5)
                ax.text(min(hi, cap) + 1, y + 0.22, f"{v:.1f}", fontsize=8, color=INK)
        ax.set_yticks(range(len(order))); ax.set_yticklabels([m.replace("g=", "γ=") for m in order][::-1]); ax.set_xlim(0, cap)
        ax.set_xlabel("R₈₀: hands after the switch to regain 80% of own pre-switch fraction (95% CI)")
        ax.set_title(f"{'original checkpoint grid (decides)' if gname == 'original_grid' else 'fine checkpoint grid (robustness)'}; "
                     f"bands: ≤1.5×, 1.5–2×, >2× oracle ({ro:.1f})", fontsize=9, color=INK)
    fig.tight_layout(); fig.savefig(D / "fig2_recovery.png", dpi=130); plt.close(fig)
    # ---- Fig 3: detection delays (ECDF), change-point error, hindsight grid trade-off
    det = r["detection"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axs[0]
    P = np.load(D / "processes.npz", allow_pickle=True); hsel = np.repeat(P["SW_sel"], 2)
    DLAB = {"CPD": "CUSUM (first alarm)", "BOCPD": "BOCPD (posterior on new segment > 0.5)", "KNOWN-CUSUM": "known-model CUSUM (oracle detector)"}
    for name, c, ls in (("CPD", STY["CPD-PRIOR-EM"][0], "-"), ("BOCPD", STY["BOCPD-PRIOR-EM"][0], "-"), ("KNOWN-CUSUM", STY["KNOWN-CUSUM-RESET"][0], ":")):
        if name not in det:
            continue
        for g, lw in (("contrasting", 2.2), ("random", 1.1)):
            dl = np.array(det[name]["_delay"], float)[hsel == g]; xs = np.sort(np.where(np.isfinite(dl), dl, 1e9)); ys = np.arange(1, len(xs) + 1) / len(xs)
            ax.step(xs, ys, where="post", color=c, ls=ls, lw=lw, label=f"{DLAB[name]}, {g}")
    if "floor" in r:
        fl = r["floor"]["contrasting"]["median_lorden_delay_log1000"]; ax.axvline(fl, color="#e34948", lw=1, ls="--")
        ax.text(fl + 1, 0.05, f"Lorden floor\nlog(1000)/KL,\nmedian {fl:.1f}", fontsize=7.5, color=INK)
    ax.set_xlim(0, 100); ax.set_ylim(0, 1.01); ax.set_xlabel("detection delay (hands after the switch to the first alarm)"); ax.set_ylabel("share of histories detected")
    ax.set_title("how fast is the switch detected?", fontsize=9.5, color=INK); ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax = axs[1]
    T_al = np.load(D / "test_alarms.npy"); sel = r["meta"]["selected_name"]; si = [cfg_name(c) for c in GRID].index(sel); nsw = len(hsel)
    a = T_al[(T_al[:, 0] == si) & (T_al[:, 1] < nsw) & (T_al[:, 2] >= SWITCH_AT)]; first = {}
    for h, t, s in a[np.argsort(a[:, 2])][:, 1:]:
        first.setdefault(int(h), int(s) - SWITCH_AT)
    ak = np.load(D / "test_alarms_known.npy"); ak = ak[ak[:, 2] >= SWITCH_AT]; firstk = {}
    for h, t, s in ak[np.argsort(ak[:, 2])][:, 1:]:
        firstk.setdefault(int(h), int(s) - SWITCH_AT)
    bins = np.arange(-30, 32, 3)
    ax.hist(np.clip(list(first.values()), -30, 30), bins=bins, color=STY["CPD-PRIOR-EM"][0], alpha=0.75, label="CPD")
    ax.hist(np.clip(list(firstk.values()), -30, 30), bins=bins, histtype="step", color=INK, lw=1.2, ls=":", label="known-model CUSUM")
    ax.axvline(0, color=MUTED, lw=0.8); ax.set_xlabel("estimated change point − true change point (hands; clipped at ±30)"); ax.set_ylabel("histories (first post-switch alarm)")
    ax.set_title("where does the reset start the new segment?", fontsize=9.5, color=INK); ax.legend(fontsize=8, frameon=False)
    ax = axs[2]; hg = r["SWITCH"]["contrasting"]["hindsight_grid"]; cost = r["STAT"]["hindsight_grid_cost"]
    WC = {0: "#2a78d6", 10: "#eb6834", 25: "#1baf7a"}
    for c in GRID:
        m = cfg_name(c); rt = hg[m]["ratio"]; y = min(rt, 6) if np.isfinite(rt) else 6
        ax.scatter(cost[m], y, s=18 + 8 * c[2], color=WC[c[0]], alpha=0.8, edgecolor="none", marker="o" if np.isfinite(rt) else "^")
    m = sel; rt = hg[m]["ratio"]; ax.scatter(cost[m], min(rt, 6), s=160, facecolor="none", edgecolor=INK, lw=1.5, zorder=4)
    ax.annotate("calibrated CUSUM", (cost[m], min(rt, 6)), xytext=(10, 10), textcoords="offset points", fontsize=8, color=INK)
    for m in [k for k in hg if k.startswith("BOCPD[")]:
        ax.scatter(cost[m], min(hg[m]["ratio"], 6), s=70, marker="v", color=STY["BOCPD-PRIOR-EM"][0], zorder=4)
        if "bocpd_calibration" in r and m == r["bocpd_calibration"]["selected_name"]:
            ax.scatter(cost[m], min(hg[m]["ratio"], 6), s=160, facecolor="none", edgecolor=STY["BOCPD-PRIOR-EM"][0], lw=1.5, zorder=4)
            ax.annotate("calibrated BOCPD", (cost[m], min(hg[m]["ratio"], 6)), xytext=(10, -14), textcoords="offset points", fontsize=8, color=STY["BOCPD-PRIOR-EM"][0])
    for yv in (1.5, 2.0):
        ax.axhline(yv, color="#e34948" if yv == 2 else "#1baf7a", lw=0.9, ls="--")
    for wa, c in WC.items():
        ax.scatter([], [], color=c, label=f"W_a = {wa}{' (population predictive)' if wa == 0 else ' hands'}")
    ax.axvline(0.02, color=MUTED, lw=0.8, ls=":"); ax.set_xlabel("TEST-STAT stationary cost vs full PRIOR-EM (fraction)"); ax.set_ylabel("R₈₀ ratio to oracle (capped at 6)")
    ax.set_title("hindsight on test: 45 CUSUM configs (size ∝ τ), 3 BOCPD hazards (▼)", fontsize=9.5, color=INK); ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig(D / "fig3_detection.png", dpi=130); plt.close(fig)
    # ---- Fig 4: stationary control and drift
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.6))
    st = r["STAT"]["fraction_by_ckpt"]; ax = axs[0]
    for m in st:
        c, mk, ls, lw, _ = STY[m]; ax.plot([100, 200, 300, 400, 500], st[m], color=c, marker=mk, ls=ls, lw=lw, label=f"{LAB(m)}  (mean {r['STAT']['mean_fraction'][m]:.3f})")
    ax.set_xlabel("hands t (stationary opponent)"); ax.set_ylabel("fraction (pooled)"); ax.set_title("TEST-STAT: 40 stationary opponents — the price of watching for switches", fontsize=9.5, color=INK)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    dr = r["DRIFT"]; ax = axs[1]
    for m in dr["fraction"]:
        line(ax, DR_T, dr["fraction"][m], dr["fraction_ci"][m][0], dr["fraction_ci"][m][1], m, band=m in ("CPD-PRIOR-EM", "BOCPD-PRIOR-EM", "PRIOR-EM-WIN"))
    ax.set_xlabel("hands t (opponent drifts linearly from q_A to q_B over 500 hands)"); ax.set_ylabel("fraction (pooled)")
    ax.set_title("DRIFT: 40 processes", fontsize=9.5, color=INK); ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(D / "fig4_stationary_drift.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
