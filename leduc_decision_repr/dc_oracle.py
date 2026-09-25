"""Part A of the decision-compression study (REPORT_LEDUC_DECISION_COMPRESSION.md): oracle K-cell partitions of opponent
space built on the 1 200 training opponents: BEH (KL k-means on hand distributions), GVAR (Euclidean k-means on g),
DEC (regret Lloyd at eps = 0.10).  Evaluated on train / val / test / OOD with exact LPs; every distinct deployed
response audited.  Writes outputs/dcomp/{oracle.json, oracle_assign.npz}."""
from .dc_common import D, EPS_LIST, OOD_FAMS, log, load_sets, HandDist, kl_rows, fraction, boot_fraction, LPPool
import json, time
import numpy as np
from .common import save_json

KS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def kmeanspp(n, K, dist_to, rng):
    idx = [int(rng.integers(n))]; d = dist_to(idx[0])
    for _ in range(1, K):
        p = np.maximum(d, 0); i = int(rng.choice(n, p=p / p.sum())) if p.sum() > 0 else int(rng.integers(n)); idx.append(i); d = np.minimum(d, dist_to(i))
    return idx


def kmeans(X, K, rng, kind, iters=50, restarts=3):
    """kind 'euclid' on rows of X, or 'kl' (Bregman, KL(x || c)) on distributions X.  Returns labels, centroids, objective."""
    n = len(X); best = None
    lX = np.log(np.maximum(X, 1e-300)); H = (X * lX).sum(1) if kind == "kl" else None; sq = (X ** 2).sum(1)
    def D2C(C):
        if kind == "euclid":
            return sq[:, None] + (C ** 2).sum(1)[None] - 2 * X @ C.T
        return H[:, None] - X @ np.log(np.maximum(C, 1e-300)).T
    def dist_to(i):
        return D2C(X[i][None])[:, 0]
    for r in range(restarts):
        C = X[kmeanspp(n, K, dist_to, rng)].copy(); lab = None
        for it in range(iters):
            Dm = D2C(C); new = Dm.argmin(1)
            for k in range(K):                                       # reseed empty cells with the worst-fit point
                if not (new == k).any():
                    far = int(Dm[np.arange(n), new].argmax()); new[far] = k; Dm[far] = -np.inf; Dm[far, k] = 0
            if lab is not None and np.array_equal(new, lab):
                break
            lab = new; C = np.stack([X[lab == k].mean(0) for k in range(K)])
        obj = float(D2C(C)[np.arange(n), lab].sum())
        if best is None or obj < best[2]:
            best = (lab.copy(), C.copy(), obj)
    return best


class Responses:
    """Cached exact eps-safe responses to the mean g of a set of training opponents."""
    def __init__(self, pool, G):
        self.pool, self.G, self.cache = pool, G, {}

    def get(self, groups, eps):
        keys = [(eps, tuple(int(i) for i in g)) for g in groups]; need = [k for k in dict.fromkeys(keys) if k not in self.cache]
        if need:
            X, _, ok = self.pool.solve(np.stack([self.G[list(k[1])].mean(0) for k in need]), eps); assert ok.all()
            for k, x in zip(need, X):
                self.cache[k] = x
        return np.stack([self.cache[k] for k in keys])


def dec_lloyd(G, V, lab0, K, R, iters=20):
    lab = lab0.copy(); n = len(G); hist = []
    for it in range(iters):
        Xc = R.get([np.flatnonzero(lab == k) for k in range(K)], 0.10); val = G @ Xc.T; new = val.argmax(1)
        for k in range(K):                                           # reseed an empty cell with the worst-served opponent
            if not (new == k).any():
                reg = V - val[np.arange(n), new]; reg[np.isin(new, [kk for kk in range(K) if (new == kk).sum() <= 1])] = -np.inf
                i = int(reg.argmax()); new[i] = k
        hist.append(float(val[np.arange(n), lab].sum()))
        if np.array_equal(new, lab):
            break
        lab = new
    Xc = R.get([np.flatnonzero(lab == k) for k in range(K)], 0.10)
    return lab, Xc, float((G * Xc[lab]).sum()), hist


def main():
    t0 = time.time(); D.mkdir(parents=True, exist_ok=True); S, pop = load_sets(); hd = HandDist(pop); rng = np.random.default_rng(2027)
    for s in S.values():
        s["P"] = hd(s["Q"])
    tr = S["train"]; Gtr, Ptr = tr["G"], tr["P"]; pbar = Ptr.mean(0); pool = LPPool(4, audit=True); R = Responses(pool, Gtr)
    log(f"hand distributions: {Ptr.shape[1]} reachable types")
    parts = {}                                                         # (type, K) -> dict(lab, assign fn data)
    for K in KS:
        tk = time.time()
        if K == 1:
            for typ in ("BEH", "GVAR", "DEC"):
                parts[(typ, K)] = {"lab": np.zeros(len(Gtr), int)}
            continue
        lb, Cb, _ = kmeans(Ptr, K, rng, "kl"); lg, Cg, _ = kmeans(Gtr, K, rng, "euclid")
        parts[("BEH", K)] = {"lab": lb}; parts[("GVAR", K)] = {"lab": lg}
        best = None
        for init, l0 in (("GVAR", lg), ("BEH", lb)):
            lab, Xc, val, hist = dec_lloyd(Gtr, tr["V"][0.10], l0, K, R)
            if best is None or val > best[2]:
                best = (lab, Xc, val, hist, init)
        parts[("DEC", K)] = {"lab": best[0], "init": best[4], "lloyd_iters": len(best[3]), "lloyd_hist": best[3]}
        log(f"K={K}: partitions built ({time.time() - tk:.0f}s; DEC from {best[4]} init, {len(best[3])} Lloyd iterations; {len(R.cache)} cached LPs)")
    # ---------------- evaluation
    res = {"meta": {"K": KS, "n_train": len(Gtr), "n_types": int(Ptr.shape[1])}, "partitions": {}}; assign_out = {}
    used_X = {}                                                        # distinct deployed responses, for audit: key -> x
    def evaluate(typ, K, lab):
        groups = [np.flatnonzero(lab == k) for k in range(K)]; M = np.stack([Ptr[g].mean(0) for g in groups])
        Rk = {e: R.get(groups, e) for e in EPS_LIST}
        for e in EPS_LIST:
            for k, g in enumerate(groups):
                used_X[(e, tuple(int(i) for i in g))] = Rk[e][k]
        out = {}
        for name, s in S.items():
            if typ == "BEH":
                a = (-(s["P"] @ np.log(np.maximum(M, 1e-300)).T)).argmin(1)
            elif typ == "GVAR":
                Cg = np.stack([Gtr[g].mean(0) for g in groups]); a = ((s["G"] ** 2).sum(1)[:, None] + (Cg ** 2).sum(1)[None] - 2 * s["G"] @ Cg.T).argmin(1)
            else:
                a = (s["G"] @ Rk[0.10].T).argmax(1)
            kl = kl_rows(s["P"], M[a]); kl0 = kl_rows(s["P"], np.broadcast_to(pbar, s["P"].shape)); o = {"beh_fraction": float(1 - kl.sum() / kl0.sum())}
            for e in (EPS_LIST if name in ("train", "test") else [0.10]):
                u = (Rk[e][a] * s["G"]).sum(1); o[f"frac_{e}"] = fraction(u, s["V0"], s["V"][e])
                if name == "test":
                    o[f"frac_{e}_ci"] = boot_fraction(u, s["V0"], s["V"][e])
            if name == "ood":
                for f in OOD_FAMS:
                    m = s["family"] == f; u = (Rk[0.10][a] * s["G"]).sum(1); o[f"frac_0.1_{f}"] = fraction(u[m], s["V0"][m], s["V"][0.10][m])
                    o[f"beh_fraction_{f}"] = float(1 - kl[m].sum() / kl0[m].sum())
            out[name] = o
            if name == "test":
                assign_out[f"{typ}_{K}"] = a
        return out
    for (typ, K), p in parts.items():
        res["partitions"][f"{typ}_{K}"] = {"type": typ, "K": K, "bits": float(np.log2(K)), **{k: v for k, v in p.items() if k != "lab"}, **evaluate(typ, K, p["lab"])}
        assign_out[f"{typ}_{K}_trainlab"] = p["lab"]
    log("partitions evaluated")
    # ---------------- 'all': 1-NN over the 1 200 training opponents (deployed = that opponent's exact safe response)
    for typ in ("BEH", "GVAR", "DEC"):
        o = {}
        for name, s in S.items():
            if typ == "BEH":
                a = (-(s["P"] @ np.log(np.maximum(Ptr, 1e-300)).T)).argmin(1)
            elif typ == "GVAR":
                a = ((s["G"] ** 2).sum(1)[:, None] + (Gtr ** 2).sum(1)[None] - 2 * s["G"] @ Gtr.T).argmin(1)
            else:
                a = (s["G"] @ tr["X"][0.10].T).argmax(1)
            if name == "train":
                a = np.arange(len(Gtr)) if typ != "DEC" else a
            kl = kl_rows(s["P"], Ptr[a]); kl0 = kl_rows(s["P"], np.broadcast_to(pbar, s["P"].shape)); oo = {"beh_fraction": float(1 - kl.sum() / kl0.sum())}
            for e in (EPS_LIST if name in ("train", "test") else [0.10]):
                u = (tr["X"][e][a] * s["G"]).sum(1); oo[f"frac_{e}"] = fraction(u, s["V0"], s["V"][e])
                for i in np.unique(a):
                    used_X[(e, (int(i),))] = tr["X"][e][i]
                if name == "test":
                    oo[f"frac_{e}_ci"] = boot_fraction(u, s["V0"], s["V"][e])
            if name == "ood":
                for f in OOD_FAMS:
                    m = s["family"] == f; u = (tr["X"][0.10][a] * s["G"]).sum(1); oo[f"frac_0.1_{f}"] = fraction(u[m], s["V0"][m], s["V"][0.10][m])
                    oo[f"beh_fraction_{f}"] = float(1 - kl[m].sum() / kl0[m].sum())
            o[name] = oo
            if name == "test":
                assign_out[f"{typ}_all"] = a
        res["partitions"][f"{typ}_all"] = {"type": typ, "K": len(Gtr), "bits": float(np.log2(len(Gtr))), **o}
    log("1-NN rows evaluated")
    # ---------------- audit every distinct deployed response
    keys = list(used_X); ex = np.zeros(len(keys))
    for e in EPS_LIST:
        ii = [j for j, k in enumerate(keys) if k[0] == e]
        _, exx, _ = pool.audit(np.stack([used_X[keys[j]] for j in ii])); ex[ii] = exx - e
    res["audit"] = {"n": len(keys), "max_expl_minus_eps": float(ex.max()), "violations": int((ex > 1e-7).sum()), "lp_calls_cached": len(R.cache)}
    res["wall_s"] = time.time() - t0; pool.close()
    save_json(res, D / "oracle.json"); np.savez_compressed(D / "oracle_assign.npz", **assign_out)
    log(f"audit: {res['audit']}")
    for typ in ("BEH", "GVAR", "DEC"):
        print(typ, " ".join(f"K{K}:{res['partitions'][f'{typ}_{K}']['test']['frac_0.1']:.3f}/{res['partitions'][f'{typ}_{K}']['test']['beh_fraction']:.2f}" for K in KS + ["all"]))


if __name__ == "__main__":
    main()
