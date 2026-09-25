"""Latent-manifold coverage diagnostic (REPORT_LEDUC_MANIFOLD.md section 0; compute-driven deviations in section 'Deviations').
Fits z* = argmin ||g(q(z)) - g*||^2 through the frozen decoder (reconstruction head), either inside the training latent
cloud's 99% ball (PCA-whitened, top-k components) or unconstrained, then deploys the exact eps-safe LP on g(q(z*)).
Writes outputs/manifold/{fits.npz, deploy.npz, meta.json}."""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, time, multiprocessing as mp
import numpy as np
from .common import OUT, save_json

D = OUT / os.environ.get("MANIFOLD_DIR", "manifold"); EPS = 0.10; STEPS = int(os.environ.get("MANIFOLD_STEPS", 300)); LR = 0.05; KS = [2, 4, 8, 16, 32]
DECODERS = {"JAC-opp": "runs_jacopp/a3_s0", "RECON-889k": "runs_v3/rec889k_s0"}
FAMS = ["ID-REF", "NEAR", "FAR-ARCH", "FAR-CFR", "FAR-EXPL"]
LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]


def cloud(run):
    """Encoder z on the training opponents at N = 500 (4 streams) and PCA whitening."""
    import torch
    from .train import load_trained
    from .data.datasets import find_dataset_dir, load_split
    enc, head, ck = load_trained(OUT / run); enc.eval(); head.eval()
    tr = load_split(find_dataset_dir(), "train"); obs = tr["obs_types"][:, :, :500]; n_opp, S, _ = obs.shape
    flat = obs.reshape(-1, 500); Z = []
    with torch.no_grad():
        for s in range(0, len(flat), 100):
            Z.append(enc(torch.as_tensor(flat[s:s + 100].astype(np.int64))).numpy())
    Z = np.concatenate(Z).astype(np.float64); ids = np.repeat(tr["opp_ids"], S)
    mu = Z.mean(0); lam, V = np.linalg.eigh(np.cov(Z.T)); o = np.argsort(lam)[::-1]; lam, V = lam[o], V[:, o]
    d_full = int((lam > 1e-6 * lam[0]).sum())
    return enc, head, Z, ids, mu, lam, V, d_full


def radius(Z, mu, lam, V, k):
    W = (Z - mu) @ V[:, :k] / np.sqrt(lam[:k]); return float(np.percentile(np.linalg.norm(W, axis=1), 99))


def min_dist(A, B, chunk=256):
    """Nearest-neighbour Euclidean distance from each row of A to the rows of B (chunked dot-product form)."""
    bb = (B ** 2).sum(1); out = np.empty(len(A))
    for s in range(0, len(A), chunk):
        a = A[s:s + chunk]; d2 = (a ** 2).sum(1)[:, None] + bb[None] - 2 * a @ B.T; out[s:s + chunk] = np.sqrt(np.maximum(d2.min(1), 0))
    return out


def fit_groups(head, tg, mu, groups, steps=None, lr=LR):
    """One batched projected-Adam run over several groups; group = dict(G=targets, P=(128,k), r=radius or None, z0=init).
    Each group has its own parameters w_g (z = mu + w_g P_g^T); the decoder is evaluated once per step on all rows."""
    import torch
    mut = torch.as_tensor(mu, dtype=torch.float32); ws, Ps, Gs = [], [], []
    for g in groups:
        P = g["P"]; w0 = (g["z0"] - mu) @ np.linalg.pinv(P).T
        w = torch.tensor(w0, dtype=torch.float32, requires_grad=True)
        if g["r"] is not None:
            with torch.no_grad():
                w.mul_(torch.clamp(g["r"] / w.norm(dim=1, keepdim=True).clamp_min(1e-12), max=1.0))
        ws.append(w); Ps.append(torch.as_tensor(P, dtype=torch.float32)); Gs.append(torch.as_tensor(g["G"], dtype=torch.float32))
    steps = steps or STEPS; G = torch.cat(Gs); opt = torch.optim.Adam(ws, lr=lr); hist = None
    for s in range(steps):
        z = torch.cat([mut + w @ P.T for w, P in zip(ws, Ps)]); lp = ((tg(head(z)) - G) ** 2).sum(1)
        opt.zero_grad(); lp.sum().backward(); opt.step()
        with torch.no_grad():
            for w, g in zip(ws, groups):
                if g["r"] is not None:
                    w.mul_(torch.clamp(g["r"] / w.norm(dim=1, keepdim=True).clamp_min(1e-12), max=1.0))
        if s == max(steps - 101, 0):
            hist = lp.detach().numpy().copy()
        if (s + 1) % 50 == 0:
            print(f"    step {s+1}/{steps} mean loss {lp.mean().item():.4f}", flush=True)
    with torch.no_grad():
        z = torch.cat([mut + w @ P.T for w, P in zip(ws, Ps)]); gh = tg(head(z)); lp = ((gh - G) ** 2).sum(1).numpy()
    z, gh = z.numpy().astype(np.float64), gh.numpy().astype(np.float64); conv = (hist - lp) / np.maximum(lp, 1e-12)
    out, o = [], 0
    for g in groups:
        m = len(g["G"]); out.append((z[o:o + m], gh[o:o + m], lp[o:o + m], conv[o:o + m])); o += m
    return out


def _deploy(args):
    wid, Gh, Gs, need_values = args
    from .game.safe_lp import get_solver, OpenSpielAuditor
    from .game.sequence_form import get_sequence_form
    L = get_solver(); aud = OpenSpielAuditor(get_sequence_form(), L.v_star)
    n = len(Gh); u = np.zeros(n); ex = np.zeros(n); ok = np.zeros(n, bool); V0 = np.full(n, np.nan); Ve = np.full(n, np.nan)
    for i in range(n):
        o, pol, x = L.solve_safe(Gh[i], EPS); ok[i] = o; u[i] = x @ Gs[i]; ex[i] = aud.exploitability_of_learner(pol)
        if need_values:
            for eps, arr in ((0.0, V0), (EPS, Ve)):
                o2, _, x2 = L.solve_safe(Gs[i], eps); arr[i] = x2 @ Gs[i]
    return wid, u, ex, ok, V0, Ve


def deploy(Gh, Gs, need_values=False, workers=4):
    b = np.linspace(0, len(Gh), workers + 1).astype(int)
    with mp.get_context("spawn").Pool(workers) as pool:
        res = pool.map(_deploy, [(w, Gh[b[w]:b[w + 1]], Gs[b[w]:b[w + 1]], need_values) for w in range(workers)])
    res.sort(key=lambda t: t[0]); cat = lambda k: np.concatenate([r[k] for r in res])
    return cat(1), cat(2), cat(3), cat(4), cat(5)


def main():
    import torch
    from .game.leduc_tree import get_tree
    from .game.sequence_form import get_sequence_form
    from .game.symmetry import get_symmetry
    from .game.policy_utils import RankPolicyToG
    from .models.torch_g import TorchRankPolicyToG
    from .data.datasets import load_population
    torch.set_num_threads(4); t0 = time.time(); D.mkdir(parents=True, exist_ok=True); times = {}
    pop = load_population(); r2g = RankPolicyToG(get_tree(), get_sequence_form(), get_symmetry()); tg = TorchRankPolicyToG(r2g)
    for p in tg.parameters():
        p.requires_grad_(False)
    GF = np.load(OUT / "gen" / "families.npz", allow_pickle=True); NS = np.load(OUT / "nonstat" / "processes.npz", allow_pickle=True)
    tr_ids = np.flatnonzero(pop["split"] == 0); rng = np.random.default_rng([2028, 1])
    # ---------------- targets: gen families (non-NE + NE), training sanity sample, drift paths
    fam = GF["family"]; T_lab, T_G = [], []
    for f in FAMS + ["NE"]:
        for i in np.flatnonzero(fam == f):
            T_lab.append(f); T_G.append(GF["G"][i])
    train_s = np.sort(rng.choice(tr_ids, 100, replace=False))
    for o in train_s:
        T_lab.append("TRAIN"); T_G.append(pop["G"][o])
    dr = np.flatnonzero(NS["kind"] == "DRIFT"); path_meta = []
    for p in dr:
        for lam in LAMS:
            q = (1 - lam) * NS["pool_Q"][NS["A"][p]] + lam * NS["pool_Q"][NS["B"][p]]
            T_lab.append("DRIFT"); T_G.append(r2g.g(q[None])[0]); path_meta.append((int(p), lam, str(NS["selection"][p])))
    T_lab = np.array(T_lab); T_G = np.array(T_G); n = len(T_G)
    # subset for the k-grid / secondary decoder / init check: 30 per non-NE family + 30 training
    sub = np.concatenate([rng.choice(np.flatnonzero(T_lab == f), 30, replace=False) for f in FAMS + ["TRAIN"]]); sub.sort()
    print(f"{n} targets ({time.time()-t0:.0f}s)", flush=True)
    # ---------------- oracle values of every target
    t = time.time(); _, _, _, V0, Ve = deploy(T_G, T_G, need_values=True); times["values_s"] = time.time() - t
    out = {"labels": T_lab, "G": T_G, "V0": V0, "Veps": Ve, "subset": sub, "path_meta": np.array(path_meta, dtype=object)}
    # 1-NN bank ceiling (discrete manifold)
    Gtr = pop["G"][tr_ids]; d2 = (T_G ** 2).sum(1)[:, None] + (Gtr ** 2).sum(1)[None] - 2 * T_G @ Gtr.T
    nn_idx = np.argmin(d2, 1); out["nn_train_id"] = tr_ids[nn_idx]
    fits = {}; heads = {}
    for dname, run in DECODERS.items():
        t = time.time(); enc, head, Z, ids, mu, lam, V, d_full = cloud(run)
        for p in head.parameters():
            p.requires_grad_(False)
        zbar = np.stack([Z[ids == o].mean(0) for o in tr_ids]); z_nn = zbar[nn_idx]
        Wfull = (Z - mu) @ V[:, :d_full] / np.sqrt(lam[:d_full])
        half = np.arange(len(Wfull)) % 2 == 0                                   # within-cloud NN distance: even rows vs odd rows
        nn_within = min_dist(Wfull[half], Wfull[~half])
        out[f"{dname}/cloud"] = {"d_full": d_full, "lam": lam, "r": {str(k): radius(Z, mu, lam, V, k) for k in KS + [d_full]}, "nn_within_p99": float(np.percentile(nn_within, 99))}
        np.savez_compressed(D / f"cloud_{dname}.npz", Z=Z, ids=ids, mu=mu, lam=lam, V=V)
        times[f"{dname}/cloud_s"] = time.time() - t
        print(f"{dname}: cloud d_full={d_full}, r_full={out[f'{dname}/cloud']['r'][str(d_full)]:.2f} ({time.time()-t0:.0f}s)", flush=True)
        plan = [("full", d_full, np.arange(n) if dname == "JAC-opp" else sub), ("free", None, np.arange(n) if dname == "JAC-opp" else sub)]
        if dname == "JAC-opp":
            plan += [(f"k{k}", k, sub) for k in KS] + [("full-muinit", d_full, sub)]
        groups, tags = [], []
        for tag, k, idx in plan:
            if tag == "free":
                sc = np.sqrt(np.maximum(lam, 1e-3 * lam[0])); P = V * sc[None]; r = None
            else:
                P = V[:, :k] * np.sqrt(lam[:k])[None]; r = out[f"{dname}/cloud"]["r"][str(k)]
            zi = np.repeat(mu[None], len(idx), 0) if tag == "full-muinit" else z_nn[idx]
            groups.append({"G": T_G[idx], "P": P, "r": r, "z0": zi}); tags.append((tag, idx))
        t = time.time(); res = fit_groups(head, tg, mu, groups); times[f"{dname}/fit_s"] = time.time() - t
        for (tag, idx), (z, g, lp, conv) in zip(tags, res):
            W = (z - mu) @ V[:, :d_full] / np.sqrt(lam[:d_full])
            fits[f"{dname}/{tag}"] = {"idx": idx, "z": z, "g": g, "loss": lp, "conv": conv, "wnorm": np.linalg.norm(W, axis=1), "nn_dist": min_dist(W, Wfull)}
            print(f"  {dname}/{tag}: {len(idx)} targets, median loss {np.median(lp):.4f}, median rel change last 100 steps {np.median(conv):.2e}", flush=True)
        heads[dname] = head
        # the encoder's own z on the generalization-study histories (N = 500), for M4
        obs = GF["obs"]; Ze = []
        with torch.no_grad():
            for s in range(0, len(obs), 100):
                Ze.append(enc(torch.as_tensor(obs[s:s + 100, :500].astype(np.int64))).numpy())
        Ze = np.concatenate(Ze).astype(np.float64); We = (Ze - mu) @ V[:, :d_full] / np.sqrt(lam[:d_full])
        out[f"{dname}/encoder_wnorm"] = np.linalg.norm(We, axis=1); out[f"{dname}/encoder_hist_family"] = fam[GF["hist_opp"]]
    # latent straight lines between the fitted path endpoints (JAC-opp, on-manifold full)
    f_full = fits["JAC-opp/full"]; pos = {int(i): j for j, i in enumerate(f_full["idx"])}; drift_idx = np.flatnonzero(T_lab == "DRIFT")
    zl = np.zeros((len(drift_idx), f_full["z"].shape[1]))
    for j, ti in enumerate(drift_idx):
        pth = j // len(LAMS); lam = LAMS[j % len(LAMS)]
        z0 = f_full["z"][pos[int(drift_idx[pth * len(LAMS)])]]; z1 = f_full["z"][pos[int(drift_idx[pth * len(LAMS) + len(LAMS) - 1])]]
        zl[j] = (1 - lam) * z0 + lam * z1
    with torch.no_grad():
        gl = tg(heads["JAC-opp"](torch.as_tensor(zl, dtype=torch.float32))).numpy().astype(np.float64)
    fits["JAC-opp/latent-line"] = {"idx": drift_idx, "z": zl, "g": gl, "loss": ((gl - T_G[drift_idx]) ** 2).sum(1), "conv": np.zeros(len(drift_idx)), "wnorm": np.zeros(len(drift_idx)), "nn_dist": None}
    # ---------------- deploy every fitted g and the 1-NN bank g
    t = time.time(); dep = {}
    for key, fdict in fits.items():
        u, ex, ok, _, _ = deploy(fdict["g"], T_G[fdict["idx"]]); dep[key] = {"u": u, "expl": ex, "ok": ok}
    u, ex, ok, _, _ = deploy(Gtr[nn_idx], T_G); dep["BANK-1NN"] = {"u": u, "expl": ex, "ok": ok}
    times["deploy_s"] = time.time() - t
    flat = {}
    for key, fdict in fits.items():
        for kk, v in fdict.items():
            if v is not None:
                flat[f"fit::{key}::{kk}"] = v
    for key, d in dep.items():
        for kk, v in d.items():
            flat[f"dep::{key}::{kk}"] = v
    for kk in ("labels", "G", "V0", "Veps", "subset", "nn_train_id"):
        flat[kk] = out[kk]
    flat["path_meta"] = np.array([f"{a}|{b}|{c}" for a, b, c in out["path_meta"]])
    for dname in DECODERS:
        flat[f"{dname}/encoder_wnorm"] = out[f"{dname}/encoder_wnorm"]; flat[f"{dname}/encoder_hist_family"] = out[f"{dname}/encoder_hist_family"]
    np.savez_compressed(D / "results.npz", **flat)
    meta = {"times": times, "total_s": time.time() - t0, "steps": STEPS, "lr": LR,
            "clouds": {d: {"d_full": out[f"{d}/cloud"]["d_full"], "r": out[f"{d}/cloud"]["r"], "nn_within_p99": out[f"{d}/cloud"]["nn_within_p99"],
                           "var_explained": {str(k): float(out[f"{d}/cloud"]["lam"][:k].sum() / out[f"{d}/cloud"]["lam"].sum()) for k in KS}} for d in DECODERS}}
    save_json(meta, D / "meta.json"); print(json.dumps(meta, indent=1)); print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
