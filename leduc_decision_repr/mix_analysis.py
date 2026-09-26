"""Analysis + figures for REPORT_LEDUC_MIXPRIOR.md: MIX-BANK / MIX-LATENT vs the reused BANK, TAB-EM, JAC-opp and
PRIOR-EM (JAC-opp) results on exactly the same histories (eps = 0.10).  Writes outputs/mixprior/analysis.json and fig1-5."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import OUT, N_BUDGETS, save_json

D = OUT / "mixprior"; EPS = 0.10; E = OUT / "eval" / "test"; GROUPS = ("ID", "NEAR", "FAR-EXPL")
METHODS = ("BANK", "TAB-EM", "JAC-opp", "PRIOR-EM", "MIX-BANK", "MIX-LATENT")
ID_FILES = {"BANK": "BANK_POSTERIOR", "TAB-EM": "TABULAR_EM_UNIFORM", "JAC-opp": "NEURAL_JO_A3_s0", "PRIOR-EM": "HYB_PRIOR_EM_JOBEST"}
GEN_NAMES = {"BANK": "BANK", "TAB-EM": "TAB-EM", "JAC-opp": "JAC-opp", "PRIOR-EM": "PRIOR-EM (JAC-opp)"}
PAIRS = [("MIX-BANK", "PRIOR-EM"), ("MIX-LATENT", "MIX-BANK"), ("MIX-BANK", "BANK"), ("MIX-BANK", "TAB-EM"), ("MIX-LATENT", "PRIOR-EM")]
COL = {"BANK": "#4a3aa7", "TAB-EM": "#2a78d6", "JAC-opp": "#eb6834", "PRIOR-EM": "#008300", "MIX-BANK": "#c0187c", "MIX-LATENT": "#1baf7a"}
MARK = {"BANK": "s", "TAB-EM": "o", "JAC-opp": "^", "PRIOR-EM": "D", "MIX-BANK": "*", "MIX-LATENT": "P"}
FAMCOL = {"ID": "#2a78d6", "NEAR": "#eb6834", "FAR-EXPL": "#e87ba4"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})


def load():
    S = np.load(D / "solve.npz"); group, src, opp = S["group"], S["src"], S["opp"]; nN = len(N_BUDGETS)
    u = {a: S["u"][i] for i, a in enumerate(S["arms"])}; ex = {a: S["expl"][i] for i, a in enumerate(S["arms"])}; ok = {a: S["ok"][i] for i, a in enumerate(S["arms"])}
    is_id = group == "ID"; gen = np.load(OUT / "gen" / "solve.npz"); gm = list(gen["methods"])
    for m in ID_FILES:
        f = np.load(E / f"solve_{ID_FILES[m]}.npz"); k = GEN_NAMES[m]
        u[m] = np.zeros((len(group), nN)); ex[m] = np.zeros_like(u[m]); ok[m] = np.zeros(u[m].shape, bool)
        u[m][is_id] = f["u"][src[is_id], :, 2]; ex[m][is_id] = f["e_os"][src[is_id], :, 2]; ok[m][is_id] = f["ok"][src[is_id], :, 2]
        u[m][~is_id] = gen["u"][gm.index(k)][src[~is_id]]; ex[m][~is_id] = gen["expl"][gm.index(k)][src[~is_id]]; ok[m][~is_id] = gen["ok"][gm.index(k)][src[~is_id]]
    diag = {a: {k: S[f"{k}_{a}"] for k in ("eff", "kept", "wmax")} for a in S["arms"]}
    return u, ex, ok, group, opp, S["V0"], S["Veps"], diag


def per_opp(x, opp):
    """(H, ...) -> (n_opp, ...) means over each opponent's streams (opponents in first-appearance order)."""
    keys, inv = np.unique(opp, return_inverse=True); out = np.zeros((len(keys),) + x.shape[1:])
    np.add.at(out, inv, x); return out / np.bincount(inv).reshape((-1,) + (1,) * (x.ndim - 1))


def boot(n, B=2000, seed=0):
    return np.random.default_rng(seed).integers(0, n, (B, n))


def analysis():
    u, ex, ok, group, opp, V0, Ve, diag = load(); sel = json.loads((D / "select.json").read_text()); tm = json.loads((D / "test_meta.json").read_text())
    res = {"n_hist": {}, "n_opp": {}, "headroom_excluded": {}, "regret": {}, "fraction": {}, "paired_regret": {}, "paired_fraction": {},
           "eff_components": {}, "kept_mass": {}, "selected_kappa": sel["selected"], "K": sel["K"], "audit": {}}
    for g in GROUPS:
        h = group == g; o = opp[h]; Uo = {m: per_opp(u[m][h], o) for m in METHODS}; V0o = per_opp(V0[h], o); Veo = per_opp(Ve[h], o)
        hd = Veo - V0o; keep = hd >= 0.01; n = len(V0o); ii = boot(n); ik = boot(int(keep.sum()), seed=1)
        res["n_hist"][g] = int(h.sum()); res["n_opp"][g] = n; res["headroom_excluded"][g] = int((~keep).sum())
        R = {m: Veo[:, None] - Uo[m] for m in METHODS}
        frac = lambda U, idx: (U[keep][idx] - V0o[keep][idx][..., None]).sum(-2) / hd[keep][idx].sum(-1)[..., None]
        res["regret"][g] = {m: {"mean": R[m].mean(0).tolist(), "lo": np.percentile(R[m][ii].mean(1), 2.5, 0).tolist(),
                                "hi": np.percentile(R[m][ii].mean(1), 97.5, 0).tolist()} for m in METHODS}
        res["fraction"][g] = {}
        for m in METHODS:
            bs = frac(Uo[m], ik); res["fraction"][g][m] = {"mean": frac(Uo[m], np.arange(int(keep.sum()))).tolist(),
                                                           "lo": np.percentile(bs, 2.5, 0).tolist(), "hi": np.percentile(bs, 97.5, 0).tolist()}
        res["paired_regret"][g] = {}; res["paired_fraction"][g] = {}
        for a, b in PAIRS + [("MIX-BANK", "JAC-opp")]:
            d = R[a] - R[b]; bs = d[ii].mean(1)
            res["paired_regret"][g][f"{a} - {b}"] = {"mean": d.mean(0).tolist(), "lo": np.percentile(bs, 2.5, 0).tolist(), "hi": np.percentile(bs, 97.5, 0).tolist()}
            df = frac(Uo[a], ik) - frac(Uo[b], ik)
            res["paired_fraction"][g][f"{a} - {b}"] = {"mean": (np.array(res["fraction"][g][a]["mean"]) - np.array(res["fraction"][g][b]["mean"])).tolist(),
                                                       "lo": np.percentile(df, 2.5, 0).tolist(), "hi": np.percentile(df, 97.5, 0).tolist()}
        res["eff_components"][g] = {a: {"median": np.median(diag[a]["eff"][h], 0).tolist(), "q25": np.percentile(diag[a]["eff"][h], 25, 0).tolist(),
                                        "q75": np.percentile(diag[a]["eff"][h], 75, 0).tolist(), "mean": diag[a]["eff"][h].mean(0).tolist()} for a in diag}
        res["kept_mass"][g] = {a: np.median(diag[a]["kept"][h], 0).tolist() for a in diag}
    for m in METHODS:
        res["audit"][m] = {"n": int(ex[m].size), "violations": int((ex[m] > EPS + 1e-7).sum()), "max_expl": float(ex[m].max()), "lp_fail": int((~ok[m]).sum())}
    # ---------------- pre-registered hypotheses
    j = {N: i for i, N in enumerate(N_BUDGETS)}; pr = res["paired_regret"]; pf = res["paired_fraction"]; fr = res["fraction"]
    M1 = {str(N): pr["ID"]["MIX-BANK - BANK"]["mean"][j[N]] for N in (5, 10)}
    M2a = {g: fr[g]["MIX-BANK"]["mean"][j[500]] - fr[g]["TAB-EM"]["mean"][j[500]] for g in GROUPS}
    M2b = {g: {k: pf[g]["MIX-BANK - BANK"][k][j[500]] for k in ("mean", "lo", "hi")} for g in ("NEAR", "FAR-EXPL")}
    M3 = {str(N): pr["ID"]["MIX-BANK - PRIOR-EM"]["mean"][j[N]] for N in (5, 10, 20)}
    res["hypotheses"] = {
        "M1": {"MIX-BANK - BANK regret": M1, "pass": all(v <= 0.005 for v in M1.values())},
        "M2": {"frac MIX-BANK - TAB-EM at N=500": M2a, "frac MIX-BANK - BANK at N=500": M2b,
               "pass": all(v >= -0.02 for v in M2a.values()) and all(v["lo"] > 0 for v in M2b.values())},
        "M3": {"MIX-BANK - PRIOR-EM regret": M3, "pass": all(v <= 0.005 for v in M3.values())},
        "M4": {g: {k: pr[g]["MIX-LATENT - MIX-BANK"][k] for k in ("mean", "lo", "hi")} for g in GROUPS}}
    cells = {(g, N): pr[g]["MIX-BANK - PRIOR-EM"]["mean"][j[N]] for g in GROUPS for N in N_BUDGETS}
    not_needed = all(v <= 0.005 for v in cells.values())
    adds = [{"who": w, "group": g, "N": N, "gain": sgn * pr[g][k]["mean"][j[N]], "ci_of_diff": [pr[g][k]["lo"][j[N]], pr[g][k]["hi"][j[N]]]}
            for w, k, sgn in (("PRIOR-EM", "MIX-BANK - PRIOR-EM", 1), ("MIX-LATENT", "MIX-LATENT - MIX-BANK", -1)) for g in GROUPS for N in N_BUDGETS
            if (sgn * pr[g][k]["mean"][j[N]] >= 0.01 and (pr[g][k]["lo"][j[N]] > 0 if sgn == 1 else pr[g][k]["hi"][j[N]] < 0))]
    res["decision"] = {"MIX-BANK - PRIOR-EM regret cells": {f"{g} N={N}": v for (g, N), v in cells.items()},
                       "max_cell": max(cells.values()), "not_needed_branch": not_needed, "learned_adds_value_cells": adds,
                       "verdict": ("NOT NEEDED" if not_needed else "") + (" + " if not_needed and adds else "") + ("LEARNED ADDS VALUE" if adds else "") or "NEITHER"}
    res["test_meta"] = tm; save_json(res, D / "analysis.json"); return res


def xaxis(ax):
    ax.set_xscale("log"); ax.set_xticks(N_BUDGETS); ax.set_xticklabels([str(n) for n in N_BUDGETS]); ax.minorticks_off(); ax.set_xlabel("hands observed N")


def band(ax, d, m, lab=None, **kw):
    ax.plot(N_BUDGETS, d["mean"], color=COL[m], marker=MARK[m], ms=7 if MARK[m] == "*" else 4.5, lw=2.4 if m.startswith("MIX") else 1.5, label=lab or m, **kw)
    ax.fill_between(N_BUDGETS, d["lo"], d["hi"], color=COL[m], alpha=0.13, lw=0)


def figures(res):
    # Fig 1 regret ID
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, zoom in zip(axs, (False, True)):
        for m in METHODS:
            band(ax, res["regret"]["ID"][m], m)
        xaxis(ax); ax.set_ylabel("safe regret V_ε − u (chips per hand)")
        if zoom:
            ax.set_xlim(4.2, 60); ax.set_ylim(0.09, 0.2); ax.set_title("(b) zoom on N ≤ 50", fontsize=9, color=INK)
        else:
            ax.set_title("(a) all N", fontsize=9, color=INK); ax.legend(fontsize=7.5, frameon=False)
    fig.suptitle(f"Fig 1. In-distribution safe regret (ε = 0.10; {res['n_opp']['ID']} test opponents × 4 streams; 95% bootstrap bands over opponents)", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(D / "fig1_regret_id.png", dpi=150); plt.close(fig)
    # Fig 2 fraction OOD
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, g in zip(axs, ("NEAR", "FAR-EXPL")):
        for m in METHODS:
            band(ax, res["fraction"][g][m], m)
        xaxis(ax); ax.axhline(0, color=MUTED, lw=0.8); ax.set_title(f"{g} ({res['n_opp'][g]} opponents × 2 streams; {res['headroom_excluded'][g]} excluded for headroom < 0.01)", fontsize=9, color=INK)
    axs[0].set_ylabel("fraction of attainable safe gain (u − V₀)/(V_ε − V₀)"); axs[0].legend(fontsize=7.5, frameon=False)
    fig.suptitle("Fig 2. Off-distribution: fraction of the attainable safe gain (pooled; 95% bootstrap bands)", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(D / "fig2_fraction_ood.png", dpi=150); plt.close(fig)
    # Fig 3 paired differences
    pairs = [("MIX-BANK", "PRIOR-EM"), ("MIX-LATENT", "MIX-BANK"), ("MIX-BANK", "BANK")]
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, (a, b) in zip(axs, pairs):
        for g in GROUPS:
            d = res["paired_regret"][g][f"{a} - {b}"]
            ax.plot(N_BUDGETS, d["mean"], color=FAMCOL[g], marker="o", ms=4, lw=1.6, label=g)
            ax.fill_between(N_BUDGETS, d["lo"], d["hi"], color=FAMCOL[g], alpha=0.14, lw=0)
        ax.axhline(0, color=INK, lw=0.9)
        for y in (0.005, -0.005, 0.01, -0.01):
            ax.axhline(y, color=MUTED, lw=0.7, ls=":" if abs(y) == 0.005 else "--")
        xaxis(ax); ax.set_title(f"{a} − {b}\n(regret, chips; below 0 = {a} better)", fontsize=9, color=INK)
    axs[0].set_ylabel("paired regret difference"); axs[0].legend(fontsize=7.5, frameon=False)
    fig.suptitle("Fig 3. Paired differences over opponents with 95% bootstrap bands (dotted ±0.005 = M1/M3 margin; dashed ±0.01 = decision-rule margin)", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(D / "fig3_paired.png", dpi=150); plt.close(fig)
    # Fig 4 effective components
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, a in zip(axs, ("MIX-BANK", "MIX-LATENT")):
        for g in GROUPS:
            d = res["eff_components"][g][a]
            ax.plot(N_BUDGETS, d["median"], color=FAMCOL[g], marker="o", ms=4, lw=1.6, label=f"{g} (median; band = IQR)")
            ax.fill_between(N_BUDGETS, d["q25"], d["q75"], color=FAMCOL[g], alpha=0.14, lw=0)
        ax.plot(N_BUDGETS, [res["K"][str(N)] for N in N_BUDGETS], color=MUTED, ls="--", lw=1, label="K(N) kept anchors")
        xaxis(ax); ax.set_yscale("log"); ax.set_title(a, fontsize=9, color=INK)
    axs[0].set_ylabel("effective number of components 1/Σw²"); axs[0].legend(fontsize=7.5, frameon=False)
    fig.suptitle("Fig 4. How many mixture components carry the belief", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(D / "fig4_eff_components.png", dpi=150); plt.close(fig)
    # Fig 5 selected kappa
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for k, a in enumerate(("MIX-BANK", "MIX-LATENT")):
        ax.plot(N_BUDGETS, [res["selected_kappa"][a][str(N)] * (1.06 if k else 0.94) for N in N_BUDGETS], color=COL[a], marker=MARK[a], ms=8, lw=1.4, label=a)
    xaxis(ax); ax.set_yscale("log"); ax.set_yticks([3, 10, 30, 100]); ax.set_yticklabels(["3", "10", "30", "100"]); ax.minorticks_off()
    ax.set_ylabel("selected κ (validation)"); ax.legend(fontsize=8, frameon=False)
    ax.set_title("Fig 5. Concentration κ selected per N on validation\n(points offset vertically for visibility)", fontsize=9.5, color=INK)
    fig.tight_layout(); fig.savefig(D / "fig5_kappa.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    r = analysis(); figures(r)
    print(json.dumps({"hypotheses": {k: v.get("pass") for k, v in r["hypotheses"].items()}, "decision": r["decision"]["verdict"],
                      "max_cell": r["decision"]["max_cell"], "adds": r["decision"]["learned_adds_value_cells"], "audit": r["audit"]}, indent=1))
