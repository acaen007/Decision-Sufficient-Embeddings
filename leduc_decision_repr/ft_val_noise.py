"""How noisy was the FT validation signal?  Re-scores the selected checkpoints of the Phase 2 arms (control and SPO+,
seeds 0-2) on the fixed 450-point validation set (150 validation opponents x N in {20, 100, 500} x stream 0, eps 0.10),
per point, and bootstraps the paired SPO+ - control difference over validation opponents.  Validation only."""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import numpy as np
import torch
from .common import OUT, save_json
from .train import load_trained
from .data.datasets import load_population, find_dataset_dir, load_split
from .game.leduc_tree import get_tree
from .game.sequence_form import get_sequence_form
from .game.symmetry import get_symmetry
from .game.policy_utils import RankPolicyToG
from .models.torch_g import TorchRankPolicyToG
from .game.safe_lp import get_solver

VAL_N = (20, 100, 500); EPS, EPS_IDX = 0.10, 2


def per_point(run, val, pop, r2g, L):
    enc, head, ck = load_trained(run); enc.eval(); head.eval()
    obs = val["obs_types"][:, 0]; opp = val["opp_ids"]; R = np.zeros((len(opp), len(VAL_N)))
    with torch.no_grad():
        for j, N in enumerate(VAL_N):
            gh = r2g(head(enc(torch.as_tensor(obs[:, :N].astype(np.int64))))).numpy().astype(np.float64)
            for i, o in enumerate(opp):
                ok, pol, xd = L.solve_safe(gh[i], EPS)
                R[i, j] = pop["V_oracle"][o, EPS_IDX] - pop["G"][o] @ xd
    return R, int(ck["step"])


def main():
    torch.set_num_threads(1)
    pop = load_population(); val = load_split(find_dataset_dir(""), "val"); L = get_solver()
    r2g = TorchRankPolicyToG(RankPolicyToG(get_tree(), get_sequence_form(), get_symmetry()))
    ls = (OUT / "ft" / "LAMBDA_STAR").read_text().strip(); R_ = OUT / "runs_ft"; rng = np.random.default_rng(0); res = {}
    diffs = []
    for s in range(3):
        Rc, sc = per_point(R_ / f"jac_ctrl_s{s}", val, pop, r2g, L)
        Rs, ss = per_point(R_ / f"jac_spo{ls}_s{s}", val, pop, r2g, L)
        stored = [json.loads((R_ / r / "result.json").read_text())["best_val_regret"] for r in (f"jac_ctrl_s{s}", f"jac_spo{ls}_s{s}")]
        d = (Rs - Rc).mean(1)                                             # per validation opponent, averaged over the 3 N
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        res[f"s{s}"] = {"ctrl": float(Rc.mean()), "spo": float(Rs.mean()), "stored_ctrl_spo": stored, "steps_ctrl_spo": [sc, ss],
                        "diff": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
        diffs.append(d); print(s, res[f"s{s}"], flush=True)
    d = np.mean(diffs, 0); bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
    res["seed_mean"] = {"diff": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    print(res["seed_mean"]); save_json(res, OUT / "ft" / "val_noise.json")


if __name__ == "__main__":
    main()
