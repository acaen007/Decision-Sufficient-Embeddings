"""T2.2: covariance gap of the bank posterior on held-out traces.
gap_g(H)   = || A (E[y|H] - y_{E[q|H]}) ||_2
gap_reg(H) = regret under E[g|H] of acting on A y_{E[q|H]} instead of E[g|H]  (eps = 0.10)."""
from __future__ import annotations

import time
import numpy as np

from .common import OUT, N_BUDGETS, save_json
from .game.leduc_tree import get_tree
from .game.sequence_form import get_sequence_form
from .game.safe_lp import get_solver, OpenSpielAuditor
from .game.symmetry import get_symmetry
from .game.policy_utils import RankPolicyToG
from .data.tokenizer import get_token_table
from .data.datasets import load_population, find_dataset_dir, load_split
from .baselines.likelihood import HandLikelihood, type_counts
from .baselines.bank_posterior import BankPosterior

OUT_DIR = OUT / "t2_covariance"


def main(eps=0.1):
    t0 = time.time(); OUT_DIR.mkdir(parents=True, exist_ok=True)
    tree, sf, sym, tab = get_tree(), get_sequence_form(), get_symmetry(), get_token_table()
    L = get_solver(); aud = OpenSpielAuditor(sf, L.v_star); pop = load_population()
    r2g = RankPolicyToG(tree, sf, sym); lik = HandLikelihood(tab, sym)
    train = np.flatnonzero(pop["split"] == 0)
    bank = BankPosterior(lik, pop["rank_policies"][train], pop["G"][train])
    Ybank = pop["Y"][train]; Qbank = pop["rank_policies"][train]
    data = load_split(find_dataset_dir(), "test"); obs = data["obs_types"]; n_opp, n_str, _ = obs.shape
    flat = obs.reshape(n_opp * n_str, -1); H = flat.shape[0]
    gap_g = np.zeros((H, len(N_BUDGETS))); gap_reg = np.zeros((H, len(N_BUDGETS))); post_ent = np.zeros((H, len(N_BUDGETS)))
    n_aud = 0; maxv = -1
    for j, N in enumerate(N_BUDGETS):
        counts = type_counts(flat[:, :N], tab.n_types)
        post = np.exp(bank.log_posterior(counts))                       # (H, K)
        Ey = post @ Ybank                                                # E[y | H]
        Eq = np.einsum("hk,kia->hia", post, Qbank)                       # E[q | H]
        y_Eq = r2g.realization(Eq)
        Eg = Ey @ sf.A.T; g_Eq = y_Eq @ sf.A.T
        gap_g[:, j] = np.linalg.norm(Eg - g_Eq, axis=1)
        post_ent[:, j] = -(post * np.log(np.maximum(post, 1e-300))).sum(1)
        for h in range(H):
            _, pol_a, xa = L.solve_safe(Eg[h], eps); _, pol_b, xb = L.solve_safe(g_Eq[h], eps)
            gap_reg[h, j] = Eg[h] @ xa - Eg[h] @ xb
            if h % 200 == 0:
                maxv = max(maxv, aud.exploitability_of_learner(pol_b) - eps); n_aud += 1
        print(f"N={N} done ({time.time()-t0:.0f}s): mean gap_g {gap_g[:, j].mean():.4f} gap_reg {gap_reg[:, j].mean():.4f}", flush=True)
    np.savez_compressed(OUT_DIR / "gaps.npz", gap_g=gap_g, gap_reg=gap_reg, post_entropy=post_ent, hist_opp=np.repeat(data["opp_ids"], n_str))
    save_json({"eps": eps, "mean_gap_g_by_N": gap_g.mean(0).tolist(), "mean_gap_reg_by_N": gap_reg.mean(0).tolist(),
               "audit_subsample_max_violation": float(maxv), "n_audited": n_aud, "runtime_s": time.time() - t0}, OUT_DIR / "gaps_summary.json")
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
