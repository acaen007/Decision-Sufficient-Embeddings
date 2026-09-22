"""Phase 3: generate the opponent population, split, decision vectors and the safe oracle."""
from __future__ import annotations

import pickle
import time

import numpy as np

from .common import OUT, EPSILONS, config_hash, save_json, environment_info, CODE_VERSION
from .game.leduc_tree import get_tree
from .game.sequence_form import get_sequence_form
from .game.safe_lp import get_solver, OpenSpielAuditor
from .game.symmetry import get_symmetry
from .data.opponents import build_population, FAMILIES

POP_CFG = {"code": CODE_VERSION, "base_seed": 2024, "counts": [413, 413, 412, 412],
           "split": [[300, 38, 75], [300, 38, 75], [300, 37, 75], [300, 37, 75]], "epsilons": EPSILONS,
           "blueprint_rule": "eps0_safe_lp_max_value_vs_uniform", "family_A": "lognormal_tau_v2", "infosets": "rank_level_144_tied", "blueprints": "suit_symmetrized"}
POP_DIR = OUT / f"population_{config_hash(POP_CFG)}"


def main():
    t0 = time.time()
    POP_DIR.mkdir(parents=True, exist_ok=True)
    T = get_tree(); S = get_sequence_form(); L = get_solver(); sym = get_symmetry()
    aud = OpenSpielAuditor(S, L.v_star)
    bp0, bp1 = L.nash_blueprint(0), L.nash_blueprint(1)        # suit-symmetric equilibria
    bp0_rank, bp1_rank = sym.reduce(0, bp0, check_tied=True), sym.reduce(1, bp1, check_tied=True)
    x_nash = S.behavioral_to_realization(0, bp0)
    print(f"v* = {L.v_star:.10f}; blueprint exploitability p0 = {L.exploitability(x_nash):.2e}, "
          f"openspiel = {aud.exploitability_of_learner(bp0):.2e}")
    pop = build_population(T, sym, bp1_rank, base_seed=POP_CFG["base_seed"])
    K = len(pop["policies"])
    Y = np.stack([S.behavioral_to_realization(1, q) for q in pop["policies"]])     # (K, n1)
    G = Y @ S.A.T                                                                   # (K, n0) = A y
    assert np.allclose(G[0], S.A @ Y[0])
    br_val = np.array([S.learner_br_value(y) for y in Y])                           # unrestricted BR
    nash_val = G @ x_nash
    # safe oracle
    V = np.zeros((K, len(EPSILONS))); X_or = np.zeros((K, len(EPSILONS), S.n_seq[0]))
    E_or = np.zeros((K, len(EPSILONS))); E_or_os = np.zeros((K, len(EPSILONS)))
    fails = 0
    for k in range(K):
        for j, eps in enumerate(EPSILONS):
            ok, pol, x = L.solve_safe(G[k], eps)
            fails += (not ok)
            X_or[k, j] = x; V[k, j] = x @ G[k]; E_or[k, j] = L.exploitability(x)
            E_or_os[k, j] = aud.exploitability_of_learner(pol)
        if k % 200 == 0:
            print(f"oracle {k}/{K}  t={time.time()-t0:.0f}s")
    viol = E_or_os - np.array(EPSILONS)[None]
    print(f"oracle LP failures: {fails}; max audited violation {viol.max():.2e}")
    np.savez_compressed(POP_DIR / "population.npz", policies=pop["policies"], rank_policies=pop["rank_policies"],
                        blueprint0_rank=bp0_rank, blueprint1_rank=bp1_rank, family_index=pop["family_index"],
                        split=pop["split"], within_index=pop["within_index"], Y=Y, G=G, br_val=br_val,
                        nash_val=nash_val, V_oracle=V, X_oracle=X_or, E_oracle=E_or, E_oracle_os=E_or_os,
                        x_nash=x_nash, blueprint0=bp0, blueprint1=bp1, v_star=L.v_star, epsilons=EPSILONS)
    with open(POP_DIR / "params.pkl", "wb") as f:
        pickle.dump({"params": pop["params"], "family": pop["family"].tolist(), "families": FAMILIES}, f)
    save_json({"config": POP_CFG, "env": environment_info(), "v_star": L.v_star, "K": K,
               "oracle_lp_failures": fails, "max_oracle_violation": float(viol.max()),
               "runtime_s": time.time() - t0}, POP_DIR / "meta.json")
    # summary
    gain = V - nash_val[:, None]
    print("\nfamily            n    exploit(BR-v*)      gain@eps=" + "  ".join(f"{e:.2f}" for e in EPSILONS))
    for fi, fam in enumerate(FAMILIES):
        m = pop["family_index"] == fi
        print(f"{fam:22s} {m.sum():4d}  {np.mean(br_val[m]-L.v_star):6.3f}±{np.std(br_val[m]-L.v_star):5.3f}   "
              + "  ".join(f"{gain[m, j].mean():.3f}" for j in range(len(EPSILONS))))
    print(f"ALL                    {K}  {np.mean(br_val-L.v_star):6.3f}            "
          + "  ".join(f"{gain[:, j].mean():.3f}" for j in range(len(EPSILONS))))
    print(f"saved to {POP_DIR}; runtime {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
