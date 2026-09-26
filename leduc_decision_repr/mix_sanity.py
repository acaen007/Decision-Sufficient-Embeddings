"""Blocking sanity checks for REPORT_LEDUC_MIXPRIOR.md, on validation histories (150 opponents, stream 0; never test):
  K    truncation: for each N, the regret of the bank posterior truncated to its top-K anchors (K in 16/64/128/256) vs the full
       bank; K(N) = the smallest K within 0.002 of the full bank (the spec's K = 16 loses 0.017 at N = 5).
  S-a  kappa = 1e4: MIX-BANK must reproduce BANK (q_k ~ a_k; weights ~ bank posterior on the kept anchors; deployed regret
       ~ full BANK's).  Run for both weight formulas ('map' = the EM objective, 'evidence'), with K(N).
  S-b  K = 1 with a flat anchor must reproduce TAB-EM (uniform prior, alpha = 1, i.e. kappa = 1 in this parameterization).
  S-c  weights sum to 1 and are finite.
Writes outputs/mixprior/sanity.json."""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, time
import numpy as np
from .common import OUT, N_BUDGETS, save_json
from .dc_common import LPPool
from . import mix_core as M

D = OUT / "mixprior"; K_GRID = (16, 64, 128, 256); K_TOL = 0.002; SA_TOL = 0.003


def main():
    from .data.datasets import load_population, find_dataset_dir, load_split
    from .game.leduc_tree import get_tree
    from .game.symmetry import get_symmetry
    from .game.sequence_form import get_sequence_form
    from .game.policy_utils import RankPolicyToG
    from .data.tokenizer import get_token_table
    from .baselines.likelihood import HandLikelihood, type_counts
    from .baselines.tabular_em import TabularEM
    D.mkdir(parents=True, exist_ok=True); t0 = time.time(); pop = load_population(); sym = get_symmetry(); tab = get_token_table()
    lik = HandLikelihood(tab, sym); mask = np.asarray(sym.rank_legal_mask[1], bool); r2g = RankPolicyToG(get_tree(), get_sequence_form(), sym)
    tr = np.flatnonzero(pop["split"] == 0); A = pop["rank_policies"][tr]; GA = pop["G"][tr]
    all_ll = lik.type_loglik(np.log(np.maximum(A, 1e-300)).reshape(len(A), -1))
    val = load_split(find_dataset_dir(), "val"); obs = val["obs_types"][:, 0]; ids = val["opp_ids"]
    V = pop["V_oracle"][ids, 2]; G = pop["G"][ids]; pool = LPPool(4, audit=False); em = M.EMPool(4)
    res = {"K_rule": f"smallest K in {K_GRID} with truncated-BANK val regret <= full BANK + {K_TOL}", "K": {}, "S-a": {}, "S-b": {}, "S-c": {}}
    def regret(Gh):
        X, _, ok = pool.solve(Gh, 0.10); assert ok.all(); return float((V - (X * G).sum(1)).mean())
    for N in N_BUDGETS:
        c = type_counts(obs[:, :N], tab.n_types); ll = c @ all_ll.T; post = np.exp(ll - ll.max(1, keepdims=True)); post /= post.sum(1, keepdims=True)
        cum = np.cumsum(-np.sort(-post, 1), 1); need = {str(q): float(np.median((cum < q).sum(1) + 1)) for q in (0.9, 0.99)}
        r_full = regret(post @ GA); trunc = {}
        for K in K_GRID:
            top = np.argsort(-ll, 1)[:, :K]; pt = np.take_along_axis(post, top, 1); pt /= pt.sum(1, keepdims=True)
            trunc[K] = regret((pt[:, :, None] * GA[top]).sum(1))
        KN = next((K for K in K_GRID if trunc[K] <= r_full + K_TOL), K_GRID[-1]); res["K"][str(N)] = KN
        top = np.argsort(-ll, 1)[:, :KN]; pt = np.take_along_axis(post, top, 1); ptn = pt / pt.sum(1, keepdims=True)
        entry = {"K": KN, "regret_BANK_full": r_full, "regret_BANK_topK": {str(k): v for k, v in trunc.items()}, "anchors_for_mass_median": need,
                 "kept_mass_median": float(np.median(pt.sum(1)))}
        for kind in ("map", "evidence"):
            g, eff, kept, wmax, it, good = em.ghat("MIX-BANK", 1e4, KN, c, kind=kind)
            _, w, topm, _, _ = M.mixture_ghat(lik, mask, r2g, all_ll, A, 1e4, c[:20], K=KN, kind=kind)   # weights of 20 histories vs the bank posterior
            assert np.array_equal(topm, top[:20])
            r = regret(g)
            entry[kind] = {"regret_MIX": r, "regret_MIX_minus_BANK_full": r - r_full, "mean_TV_w_vs_bankpost_topK_20h": float(0.5 * np.abs(w - ptn[:20]).sum(1).mean()),
                           "eff_components_median": float(np.median(eff)), "weights_finite_sum1": bool(good.all()), "em_iters_max": int(it)}
        entry["pass"] = bool(abs(entry["evidence"]["regret_MIX_minus_BANK_full"]) <= SA_TOL and entry["evidence"]["mean_TV_w_vs_bankpost_topK_20h"] <= 0.01)
        res["S-a"][str(N)] = entry; res["S-c"][str(N)] = entry["evidence"]["weights_finite_sum1"] and entry["map"]["weights_finite_sum1"]
        print(f"S-a N={N}: {json.dumps(entry)} ({time.time()-t0:.0f}s)", flush=True)
    # S-b: K = 1, flat anchor, kappa = 1  vs  TAB-EM (uniform prior, alpha = 1)
    uni = mask / mask.sum(1, keepdims=True); tem = TabularEM(lik, mask, uni, alpha=1.0, n_iter=200)
    for N in (5, 100, 500):
        c = type_counts(obs[:, :N], tab.n_types); Qt = tem.fit(c); Qm, _, _, _, _ = M.em_fit(lik, mask, np.repeat(uni[None], len(c), 0), 1.0, c)
        g, w, top, kept, it = M.mixture_ghat(lik, mask, r2g, lik.type_loglik(np.log(np.maximum(uni, 1e-300)).reshape(1, -1)), uni[None], 1.0, c, K=1)
        res["S-b"][str(N)] = {"max_abs_q_diff": float(np.abs(Qm - Qt).max()), "max_abs_g_diff": float(np.abs(g - r2g.g(Qt)).max()),
                              "regret_MIX_K1": regret(g), "regret_TABEM": regret(r2g.g(Qt)), "weights_all_one": bool(np.allclose(w, 1))}
        res["S-b"][str(N)]["pass"] = bool(res["S-b"][str(N)]["max_abs_g_diff"] < 1e-8)
        print(f"S-b N={N}: {res['S-b'][str(N)]}", flush=True)
    res["all_pass"] = {"S-a": all(e["pass"] for e in res["S-a"].values()), "S-b": all(e["pass"] for e in res["S-b"].values()), "S-c": all(res["S-c"].values())}
    pool.close(); em.close(); res["wall_s"] = time.time() - t0; save_json(res, D / "sanity.json"); print(res["all_pass"], f"{res['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
