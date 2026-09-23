"""T4: validation-only selection of the SPO+ mixing weight lambda (and a sanity comparison of all seed-0 arms).
Exact-LP safe regret at eps = 0.10 on the 150 validation opponents (1 stream, N in {20, 100, 500}); nothing from the test split."""
import sys, json
import numpy as np, torch
from .common import OUT, N_BUDGETS, save_json
from .data.datasets import load_population, load_split, find_dataset_dir
from .game.safe_lp import get_solver
from .train import load_trained

EPS = 0.10; K = 2; NS = [20, 100, 500]


def val_regret(run_dir, L, pop, val):
    enc, head, ck = load_trained(run_dir); obs = val["obs_types"][:, 0]; ids = val["opp_ids"]
    out = {}
    with torch.no_grad():
        for N in NS:
            g_hat = head(enc(torch.as_tensor(obs[:, :N].astype(np.int64)))).numpy().astype(np.float64)
            reg = []
            for i in range(len(ids)):
                ok, pol, x = L.solve_safe(g_hat[i], EPS); reg.append(pop["V_oracle"][ids[i], K] - x @ pop["G"][ids[i]])
            out[N] = float(np.mean(reg))
    out["mean"] = float(np.mean([out[N] for N in NS])); return out


if __name__ == "__main__":
    runs = sys.argv[1:]
    L = get_solver(); pop = load_population(); val = load_split(find_dataset_dir(), "val")
    res = {r: val_regret(OUT / "runs_v3" / r, L, pop, val) for r in runs}
    for r, v in res.items(): print(r, {k: round(x, 4) for k, x in v.items()})
    save_json(res, OUT / "t4_val_select.json")
