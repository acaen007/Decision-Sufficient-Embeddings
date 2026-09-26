"""Mixture-prior study (REPORT_LEDUC_MIXPRIOR.md): MIX-BANK and MIX-LATENT, eps = 0.10.
  prep    MIX-LATENT anchors: JAC-opp (seed 0) decoded q_hat of every training opponent from training stream 0 at N = 500.
  select  kappa in {3, 10, 30, 100} per N per arm, by mean validation regret (150 validation opponents x stream 0, as T5).
  test    g_hat on 300 test opponents x streams 0-3 and NEAR / FAR-EXPL (100 opponents x 2 streams each), exact eps-safe LP,
          u(x, q) and the OpenSpiel exploitability audit of every deployed strategy.
K(N) comes from outputs/mixprior/sanity.json (truncation rule); weights are the 'evidence' formula (mix_core.log_weight).
Usage: python -m leduc_decision_repr.mix_eval {prep,select,test}"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, sys, time
import numpy as np
from .common import OUT, N_BUDGETS, save_json
from .dc_common import LPPool
from . import mix_core as M

D = OUT / "mixprior"; EPS = 0.10; KAPPAS = (3.0, 10.0, 30.0, 100.0); ARMS = ("MIX-BANK", "MIX-LATENT"); ID_STREAMS = 4; OOD = ("NEAR", "FAR-EXPL")


def K_of(N):
    return int(json.loads((D / "sanity.json").read_text())["K"][str(N)])


def prep():
    import torch
    from .train import load_trained
    from .data.datasets import load_population, find_dataset_dir, load_split
    from .game.symmetry import get_symmetry
    torch.set_num_threads(4); t0 = time.time()
    enc, head, ck = load_trained(OUT / "runs_jacopp" / "a3_s0"); assert ck["method"] == "recon"
    tr_obs = load_split(find_dataset_dir(), "train")["obs_types"][:, 0, :500]
    with torch.no_grad():
        Q = np.concatenate([head(enc(torch.as_tensor(tr_obs[s:s + 200].astype(np.int64)))).numpy() for s in range(0, len(tr_obs), 200)]).astype(np.float64)
    np.save(D / "anchors_latent.npy", Q)
    pop = load_population(); A = pop["rank_policies"][np.flatnonzero(pop["split"] == 0)]; mask = np.asarray(get_symmetry().rank_legal_mask[1], bool)
    multi = mask.sum(1) > 1; tv = 0.5 * np.abs(Q - A)[:, multi].sum(2).mean(1)
    meta = {"run": "runs_jacopp/a3_s0", "step": int(ck["step"]), "stream": 0, "N": 500, "n_anchors": len(Q),
            "mean_infoset_TV_to_true": {"mean": float(tv.mean()), "median": float(np.median(tv)), "p90": float(np.percentile(tv, 90))},
            "illegal_mass_max": float(np.abs(np.where(mask[None], 0.0, Q)).max()), "wall_s": time.time() - t0}
    save_json(meta, D / "anchors_meta.json"); print(meta, flush=True)


def select():
    from .data.datasets import load_population, find_dataset_dir, load_split
    from .data.tokenizer import get_token_table
    from .baselines.likelihood import type_counts
    t0 = time.time(); pop = load_population(); tab = get_token_table(); val = load_split(find_dataset_dir(), "val")
    obs = val["obs_types"][:, 0]; ids = val["opp_ids"]; V = pop["V_oracle"][ids, 2]; G = pop["G"][ids]
    em = M.EMPool(4); lp = LPPool(4, audit=False)
    res = {"protocol": "150 validation opponents x stream 0; mean regret at eps 0.10; argmin over kappa (ties -> smaller kappa)",
           "K": {str(N): K_of(N) for N in N_BUDGETS}, "val_regret": {a: {} for a in ARMS}, "selected": {a: {} for a in ARMS},
           "em_s": {a: {} for a in ARMS}, "sc_all_pass": True}
    for arm in ARMS:
        for N in N_BUDGETS:
            c = type_counts(obs[:, :N], tab.n_types); regs, secs = {}, {}
            for kappa in KAPPAS:
                t = time.time(); g, eff, kept, wmax, it, good = em.ghat(arm, kappa, K_of(N), c); secs[str(kappa)] = time.time() - t
                res["sc_all_pass"] &= bool(good.all())
                X, _, ok = lp.solve(g, EPS); assert ok.all(); regs[str(kappa)] = float((V - (X * G).sum(1)).mean())
            sel = min(KAPPAS, key=lambda k: (regs[str(k)], k))
            res["val_regret"][arm][str(N)] = regs; res["selected"][arm][str(N)] = sel; res["em_s"][arm][str(N)] = secs
            print(f"{arm} N={N} K={K_of(N)}: val regret {json.dumps({k: round(v, 5) for k, v in regs.items()})} -> kappa {sel} "
                  f"(EM {sum(secs.values()):.0f}s; {time.time()-t0:.0f}s)", flush=True)
    # projected test EM time: per-history EM seconds at the selected kappa x 1600 test histories
    proj = sum(res["em_s"][a][str(N)][str(res["selected"][a][str(N)])] / len(obs) * 1600 for a in ARMS for N in N_BUDGETS)
    res["projected_test_em_s"] = proj; res["wall_s"] = time.time() - t0
    em.close(); lp.close(); save_json(res, D / "select.json"); print(f"selected {res['selected']}; projected test EM {proj/60:.1f} min", flush=True)


def test_histories():
    """(obs (1600, 500), G_true (1600, n0), V0, Veps, group labels, source index: test history index h = opp*8 + s or GEN index)."""
    from .data.datasets import load_population, find_dataset_dir, load_split
    pop = load_population(); test = load_split(find_dataset_dir(), "test"); S = test["obs_types"].shape[1]
    opp_pos = np.repeat(np.arange(len(test["opp_ids"])), ID_STREAMS); strm = np.tile(np.arange(ID_STREAMS), len(test["opp_ids"]))
    ids = test["opp_ids"][opp_pos]; src_id = opp_pos * S + strm
    F = np.load(OUT / "gen" / "families.npz"); fam_h = F["family"][F["hist_opp"]]; src_ood = np.flatnonzero(np.isin(fam_h, OOD))
    obs = np.concatenate([test["obs_types"][opp_pos, strm], F["obs"][src_ood]])
    G = np.concatenate([pop["G"][ids], F["G"][F["hist_opp"][src_ood]]])
    V0 = np.concatenate([pop["V_oracle"][ids, 0], F["V0"][F["hist_opp"][src_ood]]]); Ve = np.concatenate([pop["V_oracle"][ids, 2], F["Veps"][F["hist_opp"][src_ood]]])
    group = np.concatenate([np.full(len(ids), "ID"), fam_h[src_ood]]); opp = np.concatenate([opp_pos, 1000 + F["hist_opp"][src_ood]])
    return obs, G, V0, Ve, group, np.concatenate([src_id, src_ood]), opp


def test():
    from .data.tokenizer import get_token_table
    from .baselines.likelihood import type_counts
    t0 = time.time(); tab = get_token_table(); sel = json.loads((D / "select.json").read_text())["selected"]
    obs, G, V0, Ve, group, src, opp = test_histories(); H = len(obs); nN = len(N_BUDGETS); timing = {}
    em = M.EMPool(4); ghat = {a: np.zeros((H, nN, G.shape[1])) for a in ARMS}
    diag = {a: {k: np.zeros((H, nN)) for k in ("eff", "kept", "wmax")} for a in ARMS}; sc = True
    for arm in ARMS:
        t = time.time()
        for j, N in enumerate(N_BUDGETS):
            g, eff, kept, wmax, it, good = em.ghat(arm, sel[arm][str(N)], K_of(N), type_counts(obs[:, :N], tab.n_types))
            ghat[arm][:, j] = g; diag[arm]["eff"][:, j] = eff; diag[arm]["kept"][:, j] = kept; diag[arm]["wmax"][:, j] = wmax; sc &= bool(good.all())
            print(f"{arm} N={N}: EM done, median eff comps {np.median(eff):.2f} ({time.time()-t0:.0f}s)", flush=True)
        timing[f"em_{arm}_s"] = time.time() - t; np.save(D / f"ghat_{arm}.npy", ghat[arm].astype(np.float32))
    em.close(); lp = LPPool(4, audit=True); u = np.zeros((len(ARMS), H, nN)); ex = np.zeros_like(u); ok = np.zeros(u.shape, bool)
    for a, arm in enumerate(ARMS):
        t = time.time(); X, e, k = lp.solve(ghat[arm].reshape(H * nN, -1), EPS)
        u[a] = (X.reshape(H, nN, -1) * G[:, None]).sum(2); ex[a] = e.reshape(H, nN); ok[a] = k.reshape(H, nN)
        timing[f"lp_audit_{arm}_s"] = time.time() - t; print(f"{arm}: LP + audit done, max expl {ex[a].max():.3e}, LP ok {ok[a].all()} ({time.time()-t0:.0f}s)", flush=True)
    lp.close()
    np.savez_compressed(D / "solve.npz", arms=np.array(ARMS), u=u, expl=ex, ok=ok, group=group, src=src, opp=opp, V0=V0, Veps=Ve,
                        **{f"{k}_{a}": diag[a][k] for a in ARMS for k in diag[a]})
    meta = {"eps": EPS, "selected_kappa": sel, "K": {str(N): K_of(N) for N in N_BUDGETS}, "n_hist": {g: int((group == g).sum()) for g in np.unique(group)},
            "sc_all_pass": sc, "violations": int((ex > EPS + 1e-7).sum()), "lp_failures": int((~ok).sum()), "n_audits": int(ex.size),
            "max_expl": float(ex.max()), "timing_s": timing, "wall_s": time.time() - t0}
    save_json(meta, D / "test_meta.json"); print(meta, flush=True)


if __name__ == "__main__":
    {"prep": prep, "select": select, "test": test}[sys.argv[1]]()
