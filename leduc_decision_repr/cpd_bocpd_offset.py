"""Post-hoc robustness check (declared in REPORT_LEDUC_CHANGEDETECT.md deviations): BOCPD with its change-point grid offset by
2 hands, so that the true switch after hand 200 falls between grid points (starts allowed at 0 and at hands 2, 7, 12, ... 0-based).
Same selected hazard, predictive, pruning and mixture deployment as cpd_bocpd; the posterior is snapshotted at the fine
checkpoints, which now fall inside blocks.  Test SWITCH processes only.  Writes outputs/cpd/bocpd_offset.npz."""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, time
import numpy as np
from .cpd_eval import D, Pred, H_LEN, BLOCK, CKPTS, log
from .cpd_bocpd import TOP, REL_PRUNE, MIX_MIN, lse, deploy

OFFSET = 2


def run_bocpd_bounds(pred, O, H, bounds, Ts):
    """Blocks [bounds[k], bounds[k+1]); segment starts allowed at block starts; posterior over starts snapshotted after
    hands < T for each T in Ts.  Returns {T: (n, n_bounds) posterior}."""
    n = len(O); nb_ = len(bounds) - 1; LW = np.full((n, nb_), -np.inf); LW[:, 0] = 0.0; snaps = {}; t0 = time.time()
    for k in range(nb_):
        e0, e1 = bounds[k], bounds[k + 1]; L = e1 - e0
        if k > 0:
            tot = lse(LW[:, :k], 1); LW[:, :k] += L * np.log1p(-H); LW[:, k] = np.log1p(-(1 - H) ** L) + tot   # hazard mass for a block of L hands
        nxt = O[:, e0:e1].astype(np.int64); per = np.zeros((n, k + 1, L))
        hh, ii = np.nonzero(np.isfinite(LW[:, :k]))
        if len(hh):
            q = pred.q(O, np.stack([hh, np.array(bounds)[ii], np.full(len(hh), e0)], 1))
            per[hh, ii] = np.take_along_axis(pred.type_ll(q), nxt[hh], 1)
        per[:, k] = pred.pop_ll[nxt]; cum = np.cumsum(per, 2)
        for T in Ts:
            if e0 < T <= e1 and T != e1:                                              # snapshot inside the block (after hands < T)
                x = np.where(np.isfinite(LW[:, :k + 1]), LW[:, :k + 1] + cum[:, :, T - e0 - 1], -np.inf); x -= lse(x, 1)[:, None]
                snaps[T] = np.zeros((n, nb_)); snaps[T][:, :k + 1] = np.exp(x)
        LW[:, :k + 1] = np.where(np.isfinite(LW[:, :k + 1]), LW[:, :k + 1] + cum[:, :, -1], -np.inf)
        mx = LW.max(1, keepdims=True); LW[LW < mx + REL_PRUNE] = -np.inf
        if k + 1 > TOP:
            kth = -np.sort(-LW, axis=1)[:, TOP - 1:TOP]; LW[LW < kth] = -np.inf
        LW -= lse(LW, 1)[:, None]
        if e1 in Ts:
            snaps[e1] = np.exp(LW)
        if k % 20 == 19:
            log(f"  bocpd-offset hand {e1}/{H_LEN} ({time.time() - t0:.0f}s)")
    return snaps


def main():
    P = np.load(D / "processes.npz", allow_pickle=True); pred = Pred(); sel = json.loads((D / "bocpd_calib.json").read_text())["selected_H"]
    Osw = P["SW_obs"].reshape(-1, H_LEN); Ts = CKPTS["SW"]; bounds = [0] + list(range(OFFSET, H_LEN, BLOCK)) + [H_LEN]
    t = time.time(); snaps = run_bocpd_bounds(pred, Osw, sel, bounds, Ts); mix = {}
    for h in range(len(Osw)):
        for j, T in enumerate(Ts):
            w = snaps[T][h]; keep = np.flatnonzero(w >= MIX_MIN); ww = w[keep] / w[keep].sum()
            assert all(bounds[i] < T for i in keep)
            mix[(h, j)] = [(int(bounds[i]), float(x)) for i, x in zip(keep, ww)]
    res, n_seg, n_dep, u, ex, ok = deploy(pred, P, {"SW": Osw}, {("BOCPD-OFFSET2", "SW"): mix}, audit=True)
    U, E, K = res[("BOCPD-OFFSET2", "SW")]
    np.savez_compressed(D / "bocpd_offset.npz", u=U, expl=E, ok=K, offset=OFFSET, H=sel)
    log(f"offset BOCPD done: {n_dep} deployed from {n_seg} segment estimates, max expl-eps {np.nanmax(E) - 0.10:.2e}, {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
