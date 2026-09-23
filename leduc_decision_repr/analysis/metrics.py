"""Aggregation of raw evaluation outputs into per-opponent metrics, bootstrap CIs, N-thresholds."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..common import N_BUDGETS, EPSILONS, load_json

F_DENOM_THRESHOLD = 0.02      # chips/hand: opponents with G_oracle below this are excluded from F
BOOT = 2000
F_TARGETS = [0.5, 0.8, 0.9]


class EvalData:
    """Loads solve_*.npz / pred_*.npz and the population; exposes per-opponent arrays."""

    def __init__(self, eval_dir: Path, pop: dict):
        self.dir = Path(eval_dir)
        self.meta = load_json(self.dir / "predict_meta.json")
        self.pop = pop
        self.hist_opp = np.load(self.dir / "hist_opp.npy")
        self.n_streams = self.meta["n_streams"]
        self.opp_ids = self.hist_opp.reshape(-1, self.n_streams)[:, 0]
        self.n_opp = len(self.opp_ids)
        self.family = pop["family_index"][self.opp_ids]
        self.V = pop["V_oracle"][self.opp_ids]                     # (n_opp, nE)
        self.u_nash = pop["nash_val"][self.opp_ids]                 # (n_opp,)
        self.G_oracle = self.V - self.u_nash[:, None]               # (n_opp, nE)
        self.methods = [m for m in self.meta["methods"] if (self.dir / f"solve_{m}.npz").exists()]
        self.solve = {m: dict(np.load(self.dir / f"solve_{m}.npz")) for m in self.methods}
        self.pred = {m: dict(np.load(self.dir / f"pred_{m}.npz")) for m in self.meta["methods"]
                     if (self.dir / f"pred_{m}.npz").exists()}
        # robustness: if a solve covers only a prefix of the histories (debug runs), truncate everything
        # to the opponents whose streams are all solved
        H_solved = min([s["u"].shape[0] for s in self.solve.values()] + [len(self.hist_opp)])
        n_full = H_solved // self.n_streams
        if n_full < self.n_opp:
            print(f"WARNING: only {H_solved} histories solved -> restricting to first {n_full} opponents")
            self.n_opp = n_full; H = n_full * self.n_streams
            self.opp_ids = self.opp_ids[:n_full]; self.family = self.family[:n_full]
            self.V = self.V[:n_full]; self.u_nash = self.u_nash[:n_full]; self.G_oracle = self.G_oracle[:n_full]
            for s in self.solve.values():
                for k in list(s.keys()):
                    if hasattr(s[k], "shape") and s[k].ndim >= 1 and s[k].shape[0] >= H:
                        s[k] = s[k][:H]
            for p in self.pred.values():
                for k in list(p.keys()):
                    if hasattr(p[k], "shape") and p[k].ndim >= 1 and p[k].shape[0] >= H:
                        p[k] = p[k][:H]

    def per_opp(self, arr):
        """(H, ...) -> (n_opp, ...) mean over streams (streams are consecutive)."""
        return arr.reshape(self.n_opp, self.n_streams, *arr.shape[1:]).mean(1)

    def u(self, method):
        return self.per_opp(self.solve[method]["u"])                # (n_opp, nN, nE)

    def regret(self, method):
        return self.V[:, None, :] - self.u(method)

    def frac(self, method):
        """Fraction of oracle-safe gain recovered, NaN where the denominator is below threshold."""
        num = self.u(method) - self.u_nash[:, None, None]
        den = self.G_oracle[:, None, :]
        return np.where(den >= F_DENOM_THRESHOLD, num / np.where(den >= F_DENOM_THRESHOLD, den, 1.0), np.nan)

    def nash_regret(self):
        return self.G_oracle                                         # (n_opp, nE) regret of Nash

    # --------------------------------------------------------- seed grouping
    def neural_groups(self):
        groups = {}
        for m, info in self.meta["methods"].items():
            if info["kind"] == "neural" and m in self.methods:
                groups.setdefault("NEURAL_" + info["objective"].upper(), []).append(m)
        return groups

    def method_curves(self):
        """dict name -> (regret (n_opp,nN,nE), frac (n_opp,nN,nE)); neural methods averaged over seeds."""
        out = {}
        for m in self.methods:
            if self.meta["methods"][m]["kind"] != "neural":
                out[m] = (self.regret(m), self.frac(m))
        for name, members in self.neural_groups().items():
            out[name] = (np.mean([self.regret(m) for m in members], 0), np.nanmean([self.frac(m) for m in members], 0))
            for m in members:
                out[m] = (self.regret(m), self.frac(m))
        return out


def bootstrap_mean(x, rng, n_boot=BOOT, axis=0):
    """Bootstrap the nan-mean over axis 0 of x (n_opp, ...). Returns (mean, lo, hi)."""
    n = x.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    reps = np.array([np.nanmean(x[i], axis=0) for i in idx])
    return np.nanmean(x, 0), np.nanpercentile(reps, 2.5, 0), np.nanpercentile(reps, 97.5, 0), reps


def paired_bootstrap_diff(a, b, rng, n_boot=BOOT):
    """Paired bootstrap of mean(a - b) over opponents (axis 0)."""
    d = a - b
    return bootstrap_mean(d, rng, n_boot)


def n_threshold(mean_curve, targets=F_TARGETS):
    """mean_curve: (nN,) mean fraction recovered vs N -> dict target -> N or '>500'."""
    out = {}
    for t in targets:
        hit = [N for N, v in zip(N_BUDGETS, mean_curve) if np.isfinite(v) and v >= t]
        out[f"N{int(t*100)}"] = int(hit[0]) if hit else ">500"
    return out


def auc_log_n(curve):
    """Area under curve vs log(N), normalized by the log-N range (i.e. log-N-average)."""
    x = np.log(np.array(N_BUDGETS, dtype=float))
    c = np.asarray(curve, dtype=float)
    ok = np.isfinite(c)
    if ok.sum() < 2:
        return float("nan")
    return float(np.trapezoid(c[ok], x[ok]) / (x[ok][-1] - x[ok][0]))


def summarize(ed: EvalData, rng_seed=0):
    """Full metric summary (means, CIs, N-thresholds, AUCs) for all methods, eps, families."""
    rng = np.random.default_rng(rng_seed)
    curves = ed.method_curves()
    fam_names = ["NASH_LOGIT_PERTURB", "NASH_RANDOM_MIX", "STRUCTURED_CORRELATED", "UNSTRUCTURED_DIRICHLET"]
    res = {"epsilons": EPSILONS, "N": N_BUDGETS, "F_denom_threshold": F_DENOM_THRESHOLD, "methods": {},
           "nash": {}, "oracle_gain": {}, "per_family": {}, "pairwise": {}}
    # nash and oracle gain
    for k, eps in enumerate(EPSILONS):
        m, lo, hi, _ = bootstrap_mean(ed.G_oracle[:, k], rng)
        res["oracle_gain"][str(eps)] = {"mean": float(m), "lo": float(lo), "hi": float(hi),
                                        "n_valid_F": int((ed.G_oracle[:, k] >= F_DENOM_THRESHOLD).sum()),
                                        "median": float(np.median(ed.G_oracle[:, k]))}
        for f, fn in enumerate(fam_names):
            sel = ed.family == f
            res["oracle_gain"].setdefault("by_family", {}).setdefault(fn, {})[str(eps)] = float(ed.G_oracle[sel, k].mean())
    boot_reps = {}
    for name, (R, Fr) in curves.items():
        entry = {"regret": {}, "frac": {}, "N_thresholds": {}, "auc_frac": {}, "auc_regret": {}}
        for k, eps in enumerate(EPSILONS):
            m, lo, hi, reps = bootstrap_mean(R[:, :, k], rng)
            entry["regret"][str(eps)] = {"mean": m.tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
                                         "median": np.median(R[:, :, k], 0).tolist()}
            mf, lof, hif, repsf = bootstrap_mean(Fr[:, :, k], rng)
            entry["frac"][str(eps)] = {"mean": mf.tolist(), "lo": lof.tolist(), "hi": hif.tolist()}
            entry["N_thresholds"][str(eps)] = n_threshold(mf)
            # bootstrap distribution of N thresholds
            dist = {}
            for t in F_TARGETS:
                vals = []
                for rep in repsf:
                    hit = [N for N, v in zip(N_BUDGETS, rep) if np.isfinite(v) and v >= t]
                    vals.append(hit[0] if hit else 1000)
                vals = np.array(vals)
                dist[f"N{int(t*100)}"] = {"p2.5": float(np.percentile(vals, 2.5)), "median": float(np.median(vals)),
                                          "p97.5": float(np.percentile(vals, 97.5)),
                                          "frac_reached": float((vals < 1000).mean())}
            entry["N_thresholds_boot"] = entry.get("N_thresholds_boot", {}); entry["N_thresholds_boot"][str(eps)] = dist
            auc = auc_log_n(mf)
            auc_reps = np.array([auc_log_n(rep) for rep in repsf])
            entry["auc_frac"][str(eps)] = {"value": auc, "lo": float(np.nanpercentile(auc_reps, 2.5)),
                                           "hi": float(np.nanpercentile(auc_reps, 97.5))}
            auc_r = auc_log_n(m); auc_r_reps = np.array([auc_log_n(rep) for rep in reps])
            entry["auc_regret"][str(eps)] = {"value": auc_r, "lo": float(np.nanpercentile(auc_r_reps, 2.5)),
                                             "hi": float(np.nanpercentile(auc_r_reps, 97.5))}
            boot_reps[(name, eps)] = (reps, repsf)
        res["methods"][name] = entry
        # per family
        for f, fn in enumerate(fam_names):
            sel = ed.family == f
            fe = {"regret": {}, "frac": {}, "N_thresholds": {}}
            for k, eps in enumerate(EPSILONS):
                m, lo, hi, _ = bootstrap_mean(R[sel, :, k], rng, n_boot=500)
                mf, lof, hif, _ = bootstrap_mean(Fr[sel, :, k], rng, n_boot=500)
                fe["regret"][str(eps)] = {"mean": m.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
                fe["frac"][str(eps)] = {"mean": mf.tolist(), "lo": lof.tolist(), "hi": hif.tolist(),
                                        "n_valid": int(np.isfinite(Fr[sel, :, k]).all(1).sum())}
                fe["N_thresholds"][str(eps)] = n_threshold(mf)
            res["per_family"].setdefault(fn, {})[name] = fe
    # paired comparisons of interest
    pairs = [("NEURAL_DECISION", "NEURAL_RECON"), ("NEURAL_DECISION", "BANK_POSTERIOR"),
             ("NEURAL_RECON", "BANK_POSTERIOR"), ("NEURAL_DECISION", "TABULAR_EM_NASH"),
             ("NEURAL_RECON", "TABULAR_EM_NASH"), ("NEURAL_SAFE_REGRET", "NEURAL_DECISION"),
             ("NEURAL_SAFE_REGRET", "NEURAL_RECON"), ("NEURAL_SAFE_REGRET", "BANK_POSTERIOR")]
    for a, b in pairs:
        if a in curves and b in curves:
            Ra, Fa = curves[a]; Rb, Fb = curves[b]
            entry = {}
            for k, eps in enumerate(EPSILONS):
                m, lo, hi, _ = paired_bootstrap_diff(Ra[:, :, k], Rb[:, :, k], rng)
                mf, lof, hif, _ = paired_bootstrap_diff(Fa[:, :, k], Fb[:, :, k], rng)
                entry[str(eps)] = {"regret_diff_mean": m.tolist(), "regret_diff_lo": lo.tolist(), "regret_diff_hi": hi.tolist(),
                                   "frac_diff_mean": mf.tolist(), "frac_diff_lo": lof.tolist(), "frac_diff_hi": hif.tolist()}
            res["pairwise"][f"{a}_minus_{b}"] = entry
    # safety audit summary per method
    res["safety"] = {}
    for m in ed.methods:
        s = ed.solve[m]
        viol = s["e_os"] - np.array(EPSILONS)[None, None]
        vv = viol[np.isfinite(viol)]
        res["safety"][m] = {"max_violation": float(vv.max()), "n_violations_gt_1e-7": int((vv > 1e-7).sum()),
                            "n_strategies": int(vv.size), "lp_failures": int(s["lp_failures"]),
                            "violation_quantiles": {q: float(np.quantile(vv, q)) for q in [0.5, 0.9, 0.99, 0.999, 1.0]},
                            "max_abs_fast_minus_os": float(np.nanmax(np.abs(s["e_fast"] - s["e_os"]))),
                            "mean_exploitability_by_eps": np.nanmean(s["e_os"], (0, 1)).tolist()}
    # prediction errors
    res["prediction"] = {}
    for m, p in ed.pred.items():
        e = {"g_nmse_mean": ed.per_opp(p["g_nmse"]).mean(0).tolist(), "g_raw_mean": ed.per_opp(p["g_raw"]).mean(0).tolist()}
        if "q_err" in p and np.isfinite(p["q_err"]).any():
            e["q_err_mean"] = ed.per_opp(p["q_err"]).mean(0).tolist()
        res["prediction"][m] = e
    # effective latent dimension vs N (participation ratio across opponents of stream-averaged z)
    from ..game.policy_utils import participation_ratio
    res["latent_dim"] = {}
    for m, p in ed.pred.items():
        if "z" in p:
            Z = ed.per_opp(p["z"])                               # (n_opp, nN, d)
            res["latent_dim"][m] = [participation_ratio(Z[:, j]) for j in range(len(N_BUDGETS))]
    return res
