"""POST-HOC diagnostic for REPORT_LEDUC_MIXPRIOR.md (not pre-registered): how good a prior centre can a discrete bank be?
For every in-distribution test opponent (300):
  NN-TRUE     deploy on the single training opponent nearest in true policy (mean per-infoset TV): the best possible centre for
              a bank-anchored prior that has collapsed onto one anchor (what MIX-BANK does at large N; Fig 4).
  JAC-opp q_hat (seed 0, stream 0, N = 500) TV to the true policy, vs the nearest anchor's TV.
Exact eps-safe LP at eps = 0.10 + audit.  Writes outputs/mixprior/diag.json."""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import time
import numpy as np
from .common import OUT, save_json
from .dc_common import LPPool

D = OUT / "mixprior"


def main():
    import torch
    from .data.datasets import load_population, find_dataset_dir, load_split
    from .game.symmetry import get_symmetry
    from .train import load_trained
    t0 = time.time(); pop = load_population(); mask = np.asarray(get_symmetry().rank_legal_mask[1], bool); multi = mask.sum(1) > 1
    tr = np.flatnonzero(pop["split"] == 0); A = pop["rank_policies"][tr][:, multi]; test = load_split(find_dataset_dir(), "test"); ids = test["opp_ids"]
    Qt = pop["rank_policies"][ids][:, multi]
    tv = np.stack([0.5 * np.abs(A - q[None]).sum(2).mean(1) for q in Qt])            # (300, 1200) mean per-infoset TV
    nn = tv.argmin(1); tv_nn = tv[np.arange(len(ids)), nn]
    enc, head, ck = load_trained(OUT / "runs_jacopp" / "a3_s0")
    with torch.no_grad():
        qh = head(enc(torch.as_tensor(test["obs_types"][:, 0, :500].astype(np.int64)))).numpy()[:, multi]
    tv_net = 0.5 * np.abs(qh - Qt).sum(2).mean(1)
    pool = LPPool(4, audit=True); G = pop["G"][ids]; X, ex, ok = pool.solve(pop["G"][tr][nn], 0.10); pool.close()
    reg_nn = pop["V_oracle"][ids, 2] - (X * G).sum(1)
    res = {"n_opp": len(ids), "TV_nearest_train_anchor": {"mean": float(tv_nn.mean()), "median": float(np.median(tv_nn))},
           "TV_JACopp_qhat_N500_stream0": {"mean": float(tv_net.mean()), "median": float(np.median(tv_net))},
           "frac_opp_net_closer_than_nearest_anchor": float((tv_net < tv_nn).mean()),
           "regret_NN_TRUE": {"mean": float(reg_nn.mean()), "sem": float(reg_nn.std(ddof=1) / np.sqrt(len(ids)))},
           "audit": {"violations": int((ex > 0.10 + 1e-7).sum()), "max_expl": float(ex.max()), "lp_ok": bool(ok.all())}, "wall_s": time.time() - t0}
    save_json(res, D / "diag.json"); print(res, flush=True)


if __name__ == "__main__":
    main()
