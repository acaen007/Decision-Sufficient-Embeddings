"""Phase 15-16: analyses, figures, and raw tables from the saved evaluation outputs."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .common import OUT, FIG, N_BUDGETS, EPSILONS, save_json, load_json
from .game.leduc_tree import get_tree, TERM_SHOWDOWN, TERM_FOLD_OPP, TERM_FOLD_SELF
from .game.sequence_form import get_sequence_form
from .game.symmetry import get_symmetry
from .game.policy_utils import infoset_reach_weights
from .data.tokenizer import get_token_table
from .data.datasets import load_population
from .analysis.metrics import EvalData, summarize
from .analysis.geometry import geometry_analysis, precondition_analysis
from .analysis import figures as F


def main(eval_dir: Path, fig_dir: Path):
    t0 = time.time()
    fig_dir.mkdir(parents=True, exist_ok=True)
    tree, sf, sym, tab = get_tree(), get_sequence_form(), get_symmetry(), get_token_table()
    pop = load_population()
    ed = EvalData(eval_dir, pop)
    print("methods with solve results:", ed.methods)
    summary = summarize(ed)
    save_json(summary, eval_dir / "summary.json")
    # raw per-opponent tables
    rows = []
    curves = ed.method_curves()
    for name, (R, Fr) in curves.items():
        U = None
        for i, opp in enumerate(ed.opp_ids):
            for j, N in enumerate(N_BUDGETS):
                for k, eps in enumerate(EPSILONS):
                    rows.append({"method": name, "opponent": int(opp), "family": int(ed.family[i]), "N": N, "eps": eps,
                                 "regret": float(R[i, j, k]), "frac": float(Fr[i, j, k]), "V_oracle": float(ed.V[i, k]),
                                 "u_nash": float(ed.u_nash[i]), "G_oracle": float(ed.G_oracle[i, k])})
    pd.DataFrame(rows).to_csv(eval_dir / "per_opponent_metrics.csv", index=False)
    # reach weights for the reach-weighted behavioral metric: blueprint learner vs uniform opponent
    x0 = pop["x_nash"]; y_u = sf.behavioral_to_realization(1, sf.uniform_policy(1))
    w_phys = infoset_reach_weights(tree, sf, x0, y_u, 1)
    w_rank = np.zeros(144)
    for I in range(468):
        w_rank[sym.rank_infoset_of[1][I]] += w_phys[I]
    geom, raw = geometry_analysis(ed, pop, sym, w_rank)
    save_json(geom, eval_dir / "geometry.json")
    np.savez_compressed(eval_dir / "geometry_raw.npz", **raw)
    bank_fm = ed.pred["BANK_POSTERIOR"]["family_mass"] if "BANK_POSTERIOR" in ed.pred else None
    pre = precondition_analysis(pop, sym, ed, bank_fm)
    save_json(pre, eval_dir / "precondition.json")
    # illustrative pairs: policy differences at the most different infosets
    ex = {}
    for key, info in geom["pairs_examples"].items():
        i, j = info["pair"]; a, b = ed.opp_ids[i], ed.opp_ids[j]
        qa, qb = pop["rank_policies"][a], pop["rank_policies"][b]
        diff = np.abs(qa - qb).sum(1)
        top = np.argsort(-diff)[:6]
        ex[key] = {**info, "opponents": [int(a), int(b)], "families": [int(pop["family_index"][a]), int(pop["family_index"][b])],
                   "exploitability": [float(pop["br_val"][a] - pop["v_star"]), float(pop["br_val"][b] - pop["v_star"])],
                   "V_oracle_eps0.1": [float(pop["V_oracle"][a, 2]), float(pop["V_oracle"][b, 2])],
                   "cross_values": {"u(x*(a),b)": float(pop["X_oracle"][a, 2] @ pop["G"][b]), "u(x*(b),a)": float(pop["X_oracle"][b, 2] @ pop["G"][a])},
                   "top_infoset_differences": [{"rank_infoset": sym.rank_keys[1][I], "q_a": qa[I].round(3).tolist(), "q_b": qb[I].round(3).tolist()} for I in top]}
    save_json(ex, eval_dir / "illustrative_pairs.json")
    # figures
    F.fig1_regret(summary, ed, fig_dir); F.fig2_frac(summary, fig_dir); F.fig3_thresholds(summary, fig_dir)
    F.fig4_safety(summary, ed, fig_dir); F.fig5_errors(summary, ed, fig_dir); F.fig6_family(summary, fig_dir)
    F.fig7_geometry(geom, raw, fig_dir); F.fig8_latent_dim(summary, ed, fig_dir); F.fig9_schematic(fig_dir)
    fold = [z for z in tree.terminals if tree.terminal_type[z] == TERM_FOLD_OPP and tree.round[z] == 2][0]
    fold1 = [z for z in tree.terminals if tree.terminal_type[z] == TERM_FOLD_SELF and tree.round[z] == 1][0]
    show = [z for z in tree.terminals if tree.terminal_type[z] == TERM_SHOWDOWN and len(tree.path_actions[z]) >= 9][0]
    text = F.fig10_tokenized_hands(tab, tree, fig_dir, [tab.obs_type_of_terminal[z] for z in (fold1, fold, show)])
    run_dirs = {m: info["run_dir"] for m, info in ed.meta["methods"].items() if info["kind"] == "neural"}
    if run_dirs:
        F.fig11_training_curves(run_dirs, fig_dir)
    (eval_dir / "tokenized_examples.txt").write_text(text)
    save_json({"runtime_s": time.time() - t0}, eval_dir / "analyze_meta.json")
    print("analysis done in", time.time() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", default=str(OUT / "eval" / "test"))
    ap.add_argument("--fig_dir", default=str(FIG))
    a = ap.parse_args()
    main(Path(a.eval_dir), Path(a.fig_dir))
