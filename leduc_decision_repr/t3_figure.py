"""T3 figure: k needed vs eps, component split, kappa scatter."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from .common import OUT, FIG, save_json

d = json.load(open(OUT / "t3_eps_rank" / "t3_results.json")); kp = np.load(OUT / "t3_eps_rank" / "kappa_points.npz")
eps = d["eps_grid"]; ranks = d["ranks"]; mf = np.array(d["mean_frac"]); pf = np.array(d["pooled_frac"])
fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
ax = axes[0]
for j, e in enumerate(eps):
    if e == 0: continue
    ax.plot(ranks, pf[j], "-o", ms=3, label=f"ε={e}")
ax.set_xscale("log"); ax.set_xlabel("PCA rank k of g (fit on training opponents)"); ax.set_ylabel("fraction of oracle gain retained (pooled)"); ax.axhline(0.9, color="gray", ls=":", lw=0.8); ax.legend(fontsize=6); ax.set_title("(a) retained value vs rank", fontsize=8)
ax = axes[1]
for key, lab in [("k50_pooled", "k50"), ("k80_pooled", "k80"), ("k90_pooled", "k90")]:
    ks = [d["k_needed"][str(e)][key] for e in eps[1:]]
    ax.plot(eps[1:], [k if k else np.nan for k in ks], "-o", ms=4, label=lab)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("ε"); ax.set_ylabel("rank needed"); ax.legend(fontsize=7); ax.set_title("(b) k_50 / k_80 / k_90 vs ε (signature)", fontsize=8)
ax = axes[2]
cs = d["component_split"]
ax.plot(eps[1:], cs["within_nash_value_by_eps"][1:], "-o", ms=4, label="within-Nash-hull component")
ax.plot(eps[1:], cs["orthogonal_value_by_eps"][1:], "-s", ms=4, label="orthogonal component")
ax.plot(eps[1:], cs["gain_by_eps"][1:], "--", color="gray", label="full g (oracle gain)")
ax.set_xscale("log"); ax.set_xlabel("ε"); ax.set_ylabel("mean value over mean-g response [chips]"); ax.legend(fontsize=6); ax.set_title(f"(c) g components (hull dim {d['nash_hull']['dimension_tol1e-6']})", fontsize=8)
ax = axes[3]
ax.scatter(kp["expl"], kp["dist"], s=6, alpha=0.6); k = d["kappa"]["kappa_distance_max_ratio"]; km = d["kappa"]["kappa_distance_median_ratio"]
xx = np.array([kp["expl"].min(), kp["expl"].max()]); ax.plot(xx, k * xx, "r--", lw=1, label=f"max ratio κ_dist = {k:.0f}"); ax.plot(xx, km * xx, "g:", lw=1, label=f"median ratio = {km:.0f}")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Expl(x) [chips]"); ax.set_ylabel("dist(x, Nash set) [L2, realization plan]"); ax.legend(fontsize=7); ax.set_title("(d) dist(x, N) vs Expl(x) (log-log)", fontsize=8)
fig.suptitle("Figure V3-T3: ε-rank oracle curve, Nash-hull component split, κ", y=1.03)
fig.savefig(FIG / "figV3_T3_eps_rank.png", bbox_inches="tight", dpi=130); fig.savefig(FIG / "figV3_T3_eps_rank.pdf", bbox_inches="tight")
save_json({"k_needed": d["k_needed"], "component_split": cs, "kappa": d["kappa"], "nash_hull_dim": d["nash_hull"]["dimension_tol1e-6"]}, FIG / "figV3_T3_eps_rank_data.json")
print("figure written")
