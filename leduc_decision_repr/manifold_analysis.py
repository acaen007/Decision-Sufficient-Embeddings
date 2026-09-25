"""Analysis of the latent-manifold coverage diagnostic (REPORT_LEDUC_MANIFOLD.md).  Writes outputs/manifold/analysis.json."""
import json
import numpy as np
from .common import OUT, save_json
from .manifold_diag import FAMS, LAMS, KS

D = OUT / "manifold"; B = 2000; EPS = 0.10


def main():
    R = np.load(D / "results.npz", allow_pickle=True); meta = json.loads((D / "meta.json").read_text()); rng = np.random.default_rng(0)
    lab, V0, Ve, G = R["labels"], R["V0"], R["Veps"], R["G"]
    keys = sorted({k.split("::")[1] for k in R.files if k.startswith("fit::")}) + ["BANK-1NN"]
    def arr(key, what):
        return R[f"fit::{key}::{what}"] if f"fit::{key}::{what}" in R.files else None
    def idx_of(key):
        return np.arange(len(lab)) if key == "BANK-1NN" else R[f"fit::{key}::idx"]
    def frac_ci(u, v0, ve):
        num, den = u - v0, ve - v0; keep = den >= 0.01; num, den = num[keep], den[keep]
        ii = rng.integers(0, len(num), (B, len(num))); bs = num[ii].mean(1) / den[ii].mean(1)
        return {"fraction": float(num.mean() / den.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n": int(keep.sum()),
                "median_per_opp": float(np.median(num / den))}
    # generalization-study references at N = 500 (streams averaged per opponent)
    GF = np.load(OUT / "gen" / "families.npz", allow_pickle=True); GS = np.load(OUT / "gen" / "solve.npz", allow_pickle=True)
    ms = [str(m) for m in GS["methods"]]; ho = GF["hist_opp"]; n_opp = len(GF["family"]); cnt = np.bincount(ho, minlength=n_opp)
    ref = {}
    for m in ("TAB-EM", "JAC-opp", "PRIOR-EM (JAC-opp)", "BANK"):
        u = np.bincount(ho, weights=GS["u"][ms.index(m), :, 6], minlength=n_opp) / cnt; ref[m] = u
    res = {"meta": meta, "coverage": {}, "references_N500": {}, "g_R2": {}, "convergence": {}}
    from .data.datasets import load_population
    pop = load_population(); Gtr = pop["G"][pop["split"] == 0]; gbar = Gtr.mean(0)
    for key in keys:
        idx = idx_of(key); u = R[f"dep::{key}::u"]; ex = R[f"dep::{key}::expl"]; ok = R[f"dep::{key}::ok"]
        e = {"audit": {"n": int(len(u)), "max_expl_minus_eps": float((ex - EPS).max()), "n_viol": int(((ex - EPS) > 1e-7).sum()), "lp_failures": int((~ok).sum())}}
        for f in FAMS + ["TRAIN", "NE", "DRIFT"]:
            sel = lab[idx] == f
            if not sel.any():
                continue
            ii = idx[sel]
            ent = {"regret": float((Ve[ii] - u[sel]).mean())}
            if f != "NE":
                ent.update(frac_ci(u[sel], V0[ii], Ve[ii]))
            if key != "BANK-1NN":
                g = arr(key, "g")[sel]; ent["g_R2"] = float(1 - ((g - G[ii]) ** 2).sum() / ((G[ii] - gbar) ** 2).sum())
                ent["median_rel_err"] = float(np.median(np.linalg.norm(g - G[ii], axis=1) / np.linalg.norm(G[ii] - gbar, axis=1)))
                if arr(key, "wnorm") is not None and arr(key, "wnorm").any():
                    ent["median_wnorm"] = float(np.median(arr(key, "wnorm")[sel]))
                if arr(key, "nn_dist") is not None:
                    ent["median_nn_dist"] = float(np.median(arr(key, "nn_dist")[sel]))
                ent["median_final_loss"] = float(np.median(arr(key, "loss")[sel])); ent["median_conv_last100"] = float(np.median(arr(key, "conv")[sel]))
                ent["p90_conv_last100"] = float(np.percentile(arr(key, "conv")[sel], 90))
            e[f] = ent
        res["coverage"][key] = e
    # references on the same gen opponents (non-NE families)
    for m, u in ref.items():
        res["references_N500"][m] = {}
        for f in FAMS:
            o = np.flatnonzero(GF["family"] == f)
            res["references_N500"][m][f] = frac_ci(u[o], GF["V0"][o], GF["Veps"][o])
    # cloud geometry and where the fits / encoder sit relative to the training 99% ball
    for dname, c in meta["clouds"].items():
        r_full = c["r"][str(c["d_full"])]; nn99 = c["nn_within_p99"]; loc = {}
        for tag in ("full", "free"):
            key = f"{dname}/{tag}"
            if f"fit::{key}::wnorm" not in R.files:
                continue
            idx = R[f"fit::{key}::idx"]; wn = R[f"fit::{key}::wnorm"]; nd = R[f"fit::{key}::nn_dist"] if f"fit::{key}::nn_dist" in R.files else None
            loc[tag] = {f: {"frac_outside_ball": float((wn[lab[idx] == f] > r_full * (1 + 1e-6)).mean()),
                            "frac_far_from_data": (float((nd[lab[idx] == f] > nn99).mean()) if nd is not None else None)} for f in FAMS + ["TRAIN", "DRIFT"] if (lab[idx] == f).any()}
        ew = R[f"{dname}/encoder_wnorm"]; ef = R[f"{dname}/encoder_hist_family"]
        loc["encoder_N500"] = {f: float((ew[ef == f] <= r_full).mean()) for f in FAMS}
        res[f"location_{dname}"] = {"r_full": r_full, "nn_within_p99": nn99, **loc}
    # k-grid on the subset (JAC-opp)
    kg = {}
    for tag in [f"k{k}" for k in KS] + ["full", "free", "full-muinit"]:
        key = f"JAC-opp/{tag}"; idx = idx_of(key); u = R[f"dep::{key}::u"]; sub = R["subset"]
        m = np.isin(idx, sub); kg[tag] = {f: frac_ci(u[m][lab[idx[m]] == f], V0[idx[m]][lab[idx[m]] == f], Ve[idx[m]][lab[idx[m]] == f])["fraction"] for f in FAMS + ["TRAIN"]}
    res["k_grid_subset"] = kg
    # drift paths
    pm = [s.split("|") for s in R["path_meta"]]; lam = np.array([float(p[1]) for p in pm]); psel = np.array([p[2] for p in pm])
    dp = {}
    for tag in ("full", "free", "latent-line"):
        key = f"JAC-opp/{tag}"; idx = idx_of(key); m = lab[idx] == "DRIFT"; ii = idx[m]; g = arr(key, "g")[m]; u = R[f"dep::{key}::u"][m]
        rel = np.linalg.norm(g - G[ii], axis=1) / np.linalg.norm(G[ii] - gbar, axis=1); fr = (u - V0[ii]) / np.maximum(Ve[ii] - V0[ii], 1e-12)
        dl = lam[np.searchsorted(np.flatnonzero(lab == "DRIFT"), ii)]
        dp[tag] = {str(l): {"median_rel_err": float(np.median(rel[dl == l])), "fraction": float((u[dl == l] - V0[ii][dl == l]).mean() / (Ve[ii][dl == l] - V0[ii][dl == l]).mean()),
                            "regret": float((Ve[ii][dl == l] - u[dl == l]).mean())} for l in LAMS}
    # tortuosity of the fitted (full) path in whitened coordinates
    key = "JAC-opp/full"; idx = idx_of(key); m = lab[idx] == "DRIFT"; Z = arr(key, "z")[m]
    cl = np.load(D / "cloud_JAC-opp.npz"); d_full = meta["clouds"]["JAC-opp"]["d_full"]
    W = (Z - cl["mu"]) @ cl["V"][:, :d_full] / np.sqrt(cl["lam"][:d_full]); W = W.reshape(-1, len(LAMS), d_full)
    seg = np.linalg.norm(np.diff(W, axis=1), axis=2).sum(1); endd = np.linalg.norm(W[:, -1] - W[:, 0], axis=1)
    dp["tortuosity"] = {"median": float(np.median(seg / np.maximum(endd, 1e-9))), "p90": float(np.percentile(seg / np.maximum(endd, 1e-9), 90))}
    res["drift_paths"] = dp
    # ---------------- verdicts
    cov = res["coverage"]["JAC-opp/full"]; covf = res["coverage"]["JAC-opp/free"]; loc = res["location_JAC-opp"]
    v = {"M1": {"holds": cov["ID-REF"]["fraction"] >= 0.95, "value": cov["ID-REF"]["fraction"]}}
    v["M2"] = {"holds": all(cov["ID-REF"]["fraction"] - cov[f]["fraction"] >= 0.10 for f in ("NEAR", "FAR-EXPL")),
               "gaps": {f: cov["ID-REF"]["fraction"] - cov[f]["fraction"] for f in ("NEAR", "FAR-EXPL")}}
    v["M3"] = {"holds": all(covf[f]["fraction"] - cov[f]["fraction"] >= 0.05 and loc["free"][f]["frac_outside_ball"] > 0.5 for f in ("NEAR", "FAR-EXPL")),
               "free_minus_manifold": {f: covf[f]["fraction"] - cov[f]["fraction"] for f in FAMS}, "free_outside_ball": {f: loc["free"][f]["frac_outside_ball"] for f in FAMS}}
    v["M4"] = {"holds": all(x >= 0.9 for x in loc["encoder_N500"].values()), "encoder_inside_ball": loc["encoder_N500"]}
    d5 = dp["full"]["0.5"]["median_rel_err"]; dend = max(dp["full"]["0.0"]["median_rel_err"], dp["full"]["1.0"]["median_rel_err"])
    v["M5"] = {"holds": bool(d5 > dend and dp["latent-line"]["0.5"]["median_rel_err"] > d5), "rel_err_mid": d5, "rel_err_end_max": dend,
               "latent_line_mid": dp["latent-line"]["0.5"]["median_rel_err"]}
    tab = res["references_N500"]["TAB-EM"]
    good = all(cov[f]["fraction"] >= 0.90 and cov[f]["fraction"] >= tab[f]["fraction"] for f in FAMS)
    partial = all(cov[f]["fraction"] >= 0.90 for f in ("ID-REF", "FAR-ARCH", "FAR-CFR"))
    v["DECISION"] = {"verdict": "GOOD" if good else ("PARTIAL" if partial else ("POOR" if cov["ID-REF"]["fraction"] < 0.90 else "OTHER")),
                     "ceiling": {f: cov[f]["fraction"] for f in FAMS}, "tab_em_N500": {f: tab[f]["fraction"] for f in FAMS}}
    res["verdicts"] = {k: {kk: (bool(vv) if isinstance(vv, (bool, np.bool_)) else vv) for kk, vv in x.items()} for k, x in v.items()}
    save_json(res, D / "analysis.json")
    print(json.dumps(res["verdicts"], indent=1, default=float))


if __name__ == "__main__":
    main()
