"""Figures for REPORT_LEDUC_JACOPP from outputs/jacopp/analysis.json:
  jacopp_diffs.png  - test safe regret difference vs RECON-JAC-global (arm 0) by N, eps in {0.05, 0.10, 0.20}, 95% CIs
  jacopp_val.png    - validation selection curves (exact regret, 450 fixed LPs) of every run"""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import OUT, N_BUDGETS

COL = {"A1": "C0", "A2": "C1", "A3": "C3", "A4": "C2", "A0": "k"}
NAMES = {"A0": "RECON-JAC-global", "A1": "JAC-global-proj", "A2": "OPP-REACH", "A3": "JAC-opp", "A4": "JAC-opp + g-term"}


def main():
    r = json.loads((OUT / "jacopp" / "analysis.json").read_text()); D = OUT / "jacopp"
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), sharey=True)
    x = np.arange(len(N_BUDGETS))
    for i, e in enumerate(("eps0.05", "eps0.1", "eps0.2")):
        a_ = ax[i]; a_.axhline(0, color="k", lw=0.8)
        for j, (a, c) in enumerate(sorted(r[e]["vs_A0"].items())):
            if not c:
                continue
            m, lo, hi = (np.array(c["diff"][k]) for k in ("mean", "lo", "hi")); off = (j - 1.5) * 0.08
            a_.errorbar(x + off, m, yerr=[m - lo, hi - m], color=COL[a], marker="o", ms=4, capsize=2, lw=1.2,
                        label=f"{NAMES[a]} ({len(c['seeds'])} seeds)")
        if e == "eps0.1":
            a_.axhline(-0.003, color="grey", ls=":", lw=1); a_.text(0, -0.0033, "P1 bar (-0.003)", fontsize=7, color="grey", va="top")
        a_.set_xticks(x); a_.set_xticklabels(N_BUDGETS); a_.set_xlabel("N (hands observed)")
        a_.set_title(f"test regret - RECON-JAC-global, eps = {e[3:]}", fontsize=9)
    ax[0].set_ylabel("chips (negative = better than arm 0)"); ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(D / "jacopp_diffs.png", dpi=130); plt.close(fig)

    val = r.get("validation", {})
    if val:
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
        for name, v in sorted(val.items()):
            arm = name.split("_")[0][:2].upper(); seed = int(name.rsplit("_s", 1)[1])
            c = np.array(v["curve"])
            if len(c) == 0:
                continue
            ls = "--" if name.startswith("a4") and not name.startswith(f"a4l{r.get('decisions', {}).get('lambda_star', -1):g}_") else "-"
            ax[seed].plot(c[:, 0], c[:, 1], ls, color=COL.get(arm, "grey"), lw=1, label=name)
            ax[seed].plot(v["best_step"], v["best_val_regret"], "o", color=COL.get(arm, "grey"), ms=4)
        for s in range(3):
            ax[s].set_title(f"seed {s}: exact validation regret (eps 0.10, 450 fixed LPs)", fontsize=9); ax[s].set_xlabel("step")
            ax[s].set_ylim(0.105, 0.16); ax[s].legend(fontsize=6)
        ax[0].set_ylabel("chips")
        fig.tight_layout(); fig.savefig(D / "jacopp_val.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
