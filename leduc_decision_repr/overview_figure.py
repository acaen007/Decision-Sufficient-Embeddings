"""Figure 1 of REPORT_PROJECT_OVERVIEW.md: in-distribution safe regret vs N for the six final methods (data from
outputs/mixprior/analysis.json; same colours as the MIXPRIOR figures)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import OUT, N_BUDGETS
from .mix_analysis import COL, MARK, INK, MUTED, xaxis

LABEL = {"BANK": "BANK (bank posterior)", "TAB-EM": "TAB-EM (uniform-prior EM)", "JAC-opp": "JAC-opp (network alone)",
         "PRIOR-EM": "PRIOR-EM (learned prior + EM)", "MIX-BANK": "MIX-BANK (classical mixture, true anchors)",
         "MIX-LATENT": "MIX-LATENT (classical mixture, decoded anchors)"}

if __name__ == "__main__":
    r = json.loads((OUT / "mixprior" / "analysis.json").read_text())["regret"]["ID"]
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    for m in LABEL:
        d = r[m]; big = m in ("PRIOR-EM", "MIX-BANK", "MIX-LATENT")
        ax.plot(N_BUDGETS, d["mean"], color=COL[m], marker=MARK[m], ms=6 if MARK[m] == "*" else 4, lw=2.0 if big else 1.3, label=LABEL[m])
        ax.fill_between(N_BUDGETS, d["lo"], d["hi"], color=COL[m], alpha=0.10, lw=0)
    xaxis(ax); ax.set_ylabel("safe regret (chips / hand)"); ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.set_title("Held-out opponents, ε = 0.10 (300 opponents × 4 streams; 95% bootstrap bands)", fontsize=8.5, color=INK)
    fig.tight_layout(); fig.savefig(OUT / "overview_fig1.png", dpi=200); plt.close(fig)
