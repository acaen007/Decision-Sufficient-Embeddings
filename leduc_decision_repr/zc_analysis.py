"""Analysis + figures for REPORT_LEDUC_ZCODE.md: regret / fraction by N (eps = 0.10) for the decision-code arms and the
existing baselines on the same 300 x 8 test histories; paired bootstrap over opponents; Z1-Z5 and the practical verdict.
Writes outputs/zcode/{analysis.json, fig1_regret.png, fig2_differences.png}."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import OUT, N_BUDGETS, save_json
from .data.datasets import load_population

E = OUT / "eval" / "test"; D = OUT / "zcode"; K = 2; EPS = 0.10; B = 2000
ARMS = {"ZC-DEC (d=8)": [f"NEURAL_ZC_DEC8_s{s}" for s in range(3)], "ZC-DISTILL (d=8)": [f"NEURAL_ZC_DIST8_s{s}" for s in range(3)],
        "ZC-DEC (d=4)": [f"NEURAL_ZC_DEC4_s{s}" for s in range(2)], "PRIOR-EM (ZC prior)": ["HYB_PRIOR_EM_ZC"],
        "JAC-opp": [f"NEURAL_JO_A3_s{s}" for s in range(3)], "PRIOR-EM (JAC-opp prior)": ["HYB_PRIOR_EM_JOBEST"],
        "TAB-EM": ["TABULAR_EM_UNIFORM"], "BANK": ["BANK_POSTERIOR"]}
NEW = ["ZC-DEC (d=8)", "ZC-DISTILL (d=8)", "ZC-DEC (d=4)", "PRIOR-EM (ZC prior)"]
STY = {"ZC-DEC (d=8)": ("#4a3aa7", "D", "-", 2.6), "ZC-DISTILL (d=8)": ("#8f84d6", "d", "--", 1.8), "ZC-DEC (d=4)": ("#4a3aa7", "v", ":", 1.8),
       "PRIOR-EM (ZC prior)": ("#1baf7a", "s", "--", 1.8), "JAC-opp": ("#eb6834", "o", "-", 2.2), "PRIOR-EM (JAC-opp prior)": ("#008300", "*", "-", 2.4),
       "TAB-EM": ("#2a78d6", "o", "-", 1.4), "BANK": ("#8a8984", "s", "-", 1.2)}
INK, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.6,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})


def main():
    D.mkdir(parents=True, exist_ok=True); pop = load_population(); ho = np.load(E / "hist_opp.npy"); Ve = pop["V_oracle"][ho, K]; V0 = pop["V_oracle"][ho, 0]
    opps = np.unique(ho); oix = np.searchsorted(opps, ho); rng = np.random.default_rng(0); res = {"arms": {}, "audit": {}}
    R, U = {}, {}
    for arm, names in ARMS.items():
        got = [n for n in names if (E / f"solve_{n}.npz").exists()]
        if not got:
            continue
        us = []
        for n in got:
            S = np.load(E / f"solve_{n}.npz"); u = S["u"][:, :, K]; assert np.isfinite(u).all(), n; us.append(u)
            if arm in NEW:
                ex = S["e_os"][:, :, K] - EPS; ok = S["ok"][:, :, K]
                res["audit"][n] = {"n": int(ex.size), "max_expl_minus_eps": float(ex.max()), "violations": int((ex > 1e-7).sum()), "lp_failures": int((~ok).sum())}
        U[arm] = np.mean(us, 0); R[arm] = Ve[:, None] - U[arm]
        per_seed = [(Ve[:, None] - u).mean(0) for u in us]
        hd = (Ve - V0)[:, None]; frac = (U[arm] - V0[:, None]).sum(0) / hd.sum(0)
        res["arms"][arm] = {"runs": got, "regret": R[arm].mean(0).tolist(), "fraction": frac.tolist(),
                            "regret_seed_range": [np.min(per_seed, 0).tolist(), np.max(per_seed, 0).tolist()] if len(us) > 1 else None}
    def per_opp(x):                                                    # (H, nN) -> (n_opp, nN) opponent means over streams
        return np.stack([np.bincount(oix, weights=x[:, j], minlength=len(opps)) / np.bincount(oix, minlength=len(opps)) for j in range(x.shape[1])], 1)
    def diff(a, b, cols):
        d = per_opp(R[a] - R[b])[:, cols].mean(1); ii = rng.integers(0, len(d), (B, len(d))); bs = d[ii].mean(1)
        return {"diff": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    idx = {N: j for j, N in enumerate(N_BUDGETS)}; small = [idx[5], idx[10], idx[20]]; large = [idx[100], idx[200], idx[500]]
    res["diff_by_N"] = {f"{a} - {b}": {str(N): diff(a, b, [idx[N]]) for N in N_BUDGETS}
                        for a in NEW for b in ("JAC-opp", "PRIOR-EM (JAC-opp prior)") if a in R and b in R}
    for a, b in (("ZC-DEC (d=8)", "ZC-DISTILL (d=8)"), ("ZC-DEC (d=4)", "ZC-DEC (d=8)"), ("PRIOR-EM (ZC prior)", "PRIOR-EM (JAC-opp prior)")):
        if a in R and b in R:
            res["diff_by_N"][f"{a} - {b}"] = {str(N): diff(a, b, [idx[N]]) for N in N_BUDGETS}
    v = {}
    if "ZC-DEC (d=8)" in R and "JAC-opp" in R:
        z1 = diff("ZC-DEC (d=8)", "JAC-opp", small); v["Z1"] = {"holds": z1["ci"][1] < 0, **z1}
    if "ZC-DEC (d=8)" in R and "ZC-DISTILL (d=8)" in R:
        z2 = diff("ZC-DEC (d=8)", "ZC-DISTILL (d=8)", small); v["Z2"] = {"holds": z2["ci"][1] < 0, **z2}
    if "ZC-DEC (d=8)" in R:
        f500 = res["arms"]["ZC-DEC (d=8)"]["fraction"][idx[500]]; v["Z3"] = {"holds": f500 >= 0.85, "fraction_N500": f500, "true_policy_ceiling": 0.889}
    if "ZC-DEC (d=8)" in R and "PRIOR-EM (JAC-opp prior)" in R:
        z4 = diff("ZC-DEC (d=8)", "PRIOR-EM (JAC-opp prior)", large); v["Z4"] = {"holds": z4["ci"][0] > 0, **z4}
    if "PRIOR-EM (ZC prior)" in R and "PRIOR-EM (JAC-opp prior)" in R:
        d5 = {str(N): float((R["PRIOR-EM (ZC prior)"] - R["PRIOR-EM (JAC-opp prior)"])[:, idx[N]].mean()) for N in N_BUDGETS}
        v["Z5"] = {"holds": all(x > 0 for x in d5.values()), "diff_by_N": d5}
    if "PRIOR-EM (JAC-opp prior)" in R:
        wins = {a: [N for N in N_BUDGETS if diff(a, "PRIOR-EM (JAC-opp prior)", [idx[N]])["ci"][1] < 0] for a in ("ZC-DEC (d=8)", "ZC-DEC (d=4)", "PRIOR-EM (ZC prior)") if a in R}
        v["PRACTICAL"] = {"helps_in_play": any(wins.values()), "N_where_ZC_arm_beats_PRIOR_EM_JAC": wins}
    res["verdicts"] = v
    res["audit_total"] = {"n": sum(a["n"] for a in res["audit"].values()), "max_expl_minus_eps": max([a["max_expl_minus_eps"] for a in res["audit"].values()], default=None),
                          "violations": sum(a["violations"] for a in res["audit"].values()), "lp_failures": sum(a["lp_failures"] for a in res["audit"].values())}
    save_json(res, D / "analysis.json")
    # ---- figures
    x = np.array(N_BUDGETS); fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for arm in ARMS:
        if arm not in R:
            continue
        c, mk, ls, lw = STY[arm]
        axs[0].plot(x, res["arms"][arm]["regret"], color=c, marker=mk, ls=ls, lw=lw, label=arm)
        axs[1].plot(x, res["arms"][arm]["fraction"], color=c, marker=mk, ls=ls, lw=lw, label=arm)
    axs[1].axhline(0.889, color="#4a3aa7", lw=0.9, ls=(0, (1, 3))); axs[1].text(6, 0.895, "ZC-DEC (d=8) with the true policy: 0.889", fontsize=7.5, color="#4a3aa7")
    for ax, yl in zip(axs, ("safe regret V_ε − u (chips/hand), ε = 0.10", "fraction of attainable safe gain")):
        ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels(x); ax.minorticks_off(); ax.set_xlabel("observed hands N"); ax.set_ylabel(yl)
    axs[0].legend(fontsize=7.5, frameon=False); axs[0].set_title("Predicting the 8-number regret code from hands vs existing methods (300 test opponents × 8 streams)", fontsize=9.5, color=INK)
    fig.tight_layout(); fig.savefig(D / "fig1_regret.png", dpi=130); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.6))
    for ax, base in zip(axs, ("JAC-opp", "PRIOR-EM (JAC-opp prior)")):
        for a in NEW:
            key = f"{a} - {base}"
            if key not in res["diff_by_N"]:
                continue
            c, mk, ls, lw = STY[a]; dd = res["diff_by_N"][key]
            y = [dd[str(N)]["diff"] for N in N_BUDGETS]; lo = [dd[str(N)]["ci"][0] for N in N_BUDGETS]; hi = [dd[str(N)]["ci"][1] for N in N_BUDGETS]
            ax.plot(x, y, color=c, marker=mk, ls=ls, lw=lw, label=a); ax.fill_between(x, lo, hi, color=c, alpha=0.12, lw=0)
        ax.axhline(0, color=MUTED, lw=0.9); ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels(x); ax.minorticks_off()
        ax.set_xlabel("observed hands N"); ax.set_ylabel(f"regret difference vs {base} (chips; < 0 = better)"); ax.legend(fontsize=7.5, frameon=False)
        ax.set_title(f"paired difference vs {base} (95% CI over opponents)", fontsize=9.5, color=INK)
    fig.tight_layout(); fig.savefig(D / "fig2_differences.png", dpi=130); plt.close(fig)
    print(json.dumps({k: (x_.get("holds", x_.get("helps_in_play"))) for k, x_ in v.items()}, indent=1)); print(json.dumps(res["audit_total"]))
    for arm in res["arms"]:
        print(f"{arm:26s} regret " + " ".join(f"{r:.4f}" for r in res["arms"][arm]["regret"]) + " | frac " + " ".join(f"{f:.3f}" for f in res["arms"][arm]["fraction"]))


if __name__ == "__main__":
    main()
