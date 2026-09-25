"""BOCPD arm of the change-detection study (pre-registered conditional arm, REPORT_LEDUC_CHANGEDETECT.md section 0; run because
CUSUM's ratio fell in the inconclusive band).

Adams-MacKay run-length posterior with change points on the 5-hand grid:
  * the run-length predictive is the JAC-opp amortized posterior predictive on the candidate segment, rebuilt at each block
    boundary; a segment that starts at the current block uses the exact population prior predictive;
  * the per-block hazard is 1 - (1 - H)^5 at each grid point;
  * starts are pruned to the 20 most probable, and starts below 1e-10 of the most probable are dropped.
It deploys the posterior mixture over run lengths of the g of PRIOR-EM fitted on each candidate segment.  This is exact because
g is linear in the realization plan; starts below 1e-3 posterior mass are dropped from the mixture and the rest renormalized.
The posterior is taken over which segment produced the hands seen so far, and it is scored against the current opponent.
The hazard is calibrated on the CAL processes over {1/100, 1/250, 1/1000}, with the same objective and constraint as CUSUM.
Writes outputs/cpd/{bocpd_calib.json, bocpd_test.npz, bocpd_meta.json}."""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, time
import numpy as np
from .common import save_json
from .cpd_eval import D, Pred, H_LEN, BLOCK, KEY, CKPTS, cur_values, ghat, solve_lps, log

HAZ = [1 / 100, 1 / 250, 1 / 1000]; TOP = 20; REL_PRUNE = np.log(1e-10); MIX_MIN = 1e-3; NB = H_LEN // BLOCK


def hname(H):
    return f"BOCPD[H=1/{round(1 / H)}]"


def lse(x, axis):
    m = np.max(x, axis=axis, keepdims=True); m = np.where(np.isfinite(m), m, 0.0)
    return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis)


def run_bocpd(pred, O, hazards, tag):
    """Returns post (nH, n, NB+1, NB+1) float32: the posterior at boundary e = 5b over segment starts 5i (i < b),
    computed from hands 0..e-1."""
    n = len(O); nH = len(hazards); Hs = np.asarray(hazards)
    lhb = np.log1p(-(1 - Hs) ** BLOCK); l1m = BLOCK * np.log1p(-Hs)
    LW = np.full((nH, n, NB + 1), -np.inf); LW[:, :, 0] = 0.0; post = np.zeros((nH, n, NB + 1, NB + 1), np.float32)
    n_enc = 0; t0 = time.time()
    for b in range(NB):
        e = b * BLOCK
        if b > 0:
            post[:, :, b, :b] = np.exp(LW[:, :, :b])
            tot = lse(LW[:, :, :b], 2); LW[:, :, :b] += l1m[:, None, None]; LW[:, :, b] = lhb[:, None] + tot
        nxt = O[:, e:e + BLOCK].astype(np.int64); nb = nxt.shape[1]
        alive = np.isfinite(LW[:, :, :b]).any(0)                               # (n, b): starts that need a segment model
        hh, ii = np.nonzero(alive); LLf = np.zeros((n, b + 1))
        if len(hh):
            q = pred.q(O, np.stack([hh, ii * BLOCK, np.full(len(hh), e)], 1)); n_enc += len(hh)
            LLf[hh, ii] = np.take_along_axis(pred.type_ll(q), nxt[hh], 1).sum(1)
        LLf[:, b] = pred.pop_ll[nxt].sum(1)
        LW[:, :, :b + 1] = np.where(np.isfinite(LW[:, :, :b + 1]), LW[:, :, :b + 1] + LLf[None], -np.inf)
        # prune (top 20 and relative mass) and normalize
        mx = LW.max(2, keepdims=True); LW[LW < mx + REL_PRUNE] = -np.inf
        if b + 1 > TOP:
            kth = -np.sort(-LW, axis=2)[:, :, TOP - 1:TOP]; LW[LW < kth] = -np.inf
        LW -= lse(LW, 2)[:, :, None]
        if b % 20 == 19:
            log(f"  bocpd[{tag}] hand {e + nb}/{H_LEN}: {n_enc} encodings, mean alive starts {np.isfinite(LW).sum(2).mean():.2f} ({time.time() - t0:.0f}s)")
    post[:, :, NB, :NB] = np.exp(LW[:, :, :NB])
    return post


def mixtures(post_k, T_list):
    """post_k (n, NB+1, NB+1) -> per (h, j): list of (start, weight) with weight >= MIX_MIN, renormalized."""
    out = {}
    for h in range(post_k.shape[0]):
        for j, T in enumerate(T_list):
            w = post_k[h, T // BLOCK, :T // BLOCK].astype(np.float64); keep = np.flatnonzero(w >= MIX_MIN); w = w[keep] / w[keep].sum()
            out[(h, j)] = [(int(i * BLOCK), float(x)) for i, x in zip(keep, w)]
    return out


def deploy(pred, P, Os, mix_by, audit):
    """mix_by: {(method, set): {(h, j): [(start, w), ...]}} -> {(method, set): (U, E, K)} via mixture g and exact LPs."""
    uniq = {}
    for (m, s), mix in mix_by.items():
        for (h, j), lst in mix.items():
            T = CKPTS[s][j]
            for st, _ in lst:
                uniq.setdefault((s, h, T, st), len(uniq))
    keys = list(uniq); gs = np.zeros((len(keys), 1093))
    for s in {k[0] for k in keys}:
        rows = [(i, h, T, st) for i, (ss, h, T, st) in enumerate(keys) if ss == s]; arr = np.array(rows, dtype=np.int64)
        log(f"  {s}: {len(arr)} unique segment estimates"); gs[arr[:, 0]] = ghat(pred, Os[s], np.stack([arr[:, 1], arr[:, 2], np.zeros(len(arr), np.int64), arr[:, 3]], 1))
    items = []; Gm = []; Gc = []
    for (m, s), mix in mix_by.items():
        Gt, _, _ = cur_values(P, s, CKPTS[s])
        for (h, j), lst in mix.items():
            T = CKPTS[s][j]; Gm.append(sum(w * gs[uniq[(s, h, T, st)]] for st, w in lst)); Gc.append(Gt[h // 2, j]); items.append((m, s, h, j))
    u, ex, ok = solve_lps(np.array(Gm), np.array(Gc), audit)
    res = {}
    for (m, s, h, j), uu, ee, kk in zip(items, u, ex, ok):
        if (m, s) not in res:
            n = len(Os[s]); nT = len(CKPTS[s]); res[(m, s)] = (np.full((n, nT), np.nan), np.full((n, nT), np.nan), np.zeros((n, nT), bool))
        res[(m, s)][0][h, j], res[(m, s)][1][h, j], res[(m, s)][2][h, j] = uu, ee, kk
    return res, len(keys), len(items), u, ex, ok


def frac_mean(P, s, U):
    U = U.reshape(-1, 2, U.shape[1]).mean(1); _, V0, Ve = cur_values(P, s, CKPTS[s]); keep = (Ve - V0) >= 0.01
    return float(np.mean(np.where(keep, U - V0, 0).sum(0) / np.where(keep, Ve - V0, 0).sum(0)))


def main():
    P = np.load(D / "processes.npz", allow_pickle=True); pred = Pred(); wall = {}
    cal = json.loads((D / "calib.json").read_text())
    # ---------------- calibration of the hazard (CAL processes only)
    if not (D / "bocpd_calib.json").exists():
        t = time.time(); Osw = P["CSW_obs"].reshape(-1, H_LEN); Ost = P["CST_obs"].reshape(-1, H_LEN); nsw = len(Osw)
        post = run_bocpd(pred, np.concatenate([Osw, Ost]), HAZ, "CAL"); Os = {"CSW": Osw, "CST": Ost}; mix_by = {}
        for k, H in enumerate(HAZ):
            mix_by[(hname(H), "CSW")] = mixtures(post[k, :nsw], CKPTS["CSW"]); mix_by[(hname(H), "CST")] = mixtures(post[k, nsw:], CKPTS["CST"])
        res, n_seg, n_dep, _, _, ok = deploy(pred, P, Os, mix_by, audit=False)
        F = {hname(H): {s: frac_mean(P, s, res[(hname(H), s)][0]) for s in ("CSW", "CST")} for H in HAZ}
        base = cal["F"]["PRIOR-EM"]["CST"]; cost = {m: base - F[m]["CST"] for m in F}; okH = [H for H in HAZ if cost[hname(H)] <= 0.02]
        sel = max(okH, key=lambda H: F[hname(H)]["CSW"]) if okH else min(HAZ, key=lambda H: cost[hname(H)])
        out = {"F": F, "stationary_cost": cost, "qualifying": [hname(H) for H in okH], "selected_H": sel, "selected_name": hname(sel),
               "n_segment_estimates": n_seg, "n_deployments": n_dep, "lp_failures": int((~ok).sum()), "wall_s": time.time() - t,
               "cusum_selected": cal["selected_name"], "cusum_F": cal["F"][cal["selected_name"]]}
        save_json(out, D / "bocpd_calib.json"); log(f"BOCPD calibration: selected {hname(sel)}  F {F[hname(sel)]}  cost {cost[hname(sel)]:+.4f}")
    bc = json.loads((D / "bocpd_calib.json").read_text()); sel = bc["selected_H"]
    # ---------------- test
    if not (D / "bocpd_test.npz").exists():
        t = time.time(); Osw = P["SW_obs"].reshape(-1, H_LEN); Ost = P["ST_obs"].reshape(-1, H_LEN); Odr = P["DR_obs"].reshape(-1, H_LEN); nsw = len(Osw)
        post = run_bocpd(pred, np.concatenate([Osw, Ost]), HAZ, "TEST"); post_dr = run_bocpd(pred, Odr, [sel], "DRIFT")
        Os = {"SW": Osw, "ST": Ost, "DR": Odr}; mix_by = {}; si = HAZ.index(sel)
        for k, H in enumerate(HAZ):
            mix_by[(hname(H), "SW")] = mixtures(post[k, :nsw], CKPTS["SW"]); mix_by[(hname(H), "ST")] = mixtures(post[k, nsw:], CKPTS["ST"])
        mix_by[("BOCPD-PRIOR-EM", "SW")] = mix_by[(hname(sel), "SW")]; mix_by[("BOCPD-PRIOR-EM", "ST")] = mix_by[(hname(sel), "ST")]
        mix_by[("BOCPD-PRIOR-EM", "DR")] = mixtures(post_dr[0], CKPTS["DR"])
        res, n_seg, n_dep, u, ex, ok = deploy(pred, P, Os, mix_by, audit=True)
        arrs = {}
        for (m, s), (U, E, K) in res.items():
            arrs[f"{s}::{m}::u"] = U; arrs[f"{s}::{m}::expl"] = E; arrs[f"{s}::{m}::ok"] = K
        # soft detection: posterior mass on segments starting at or after hand 195 (0-based), at every boundary, selected hazard
        grid = np.arange(NB + 1) * BLOCK
        arrs["SW_mass_post_switch"] = post[si, :nsw][:, :, grid[:NB + 1] >= 195].sum(2)
        arrs["SW_map_start"] = grid[post[si, :nsw].argmax(2)]; arrs["ST_map_start"] = grid[post[si, nsw:].argmax(2)]
        arrs["DR_map_start"] = grid[post_dr[0].argmax(2)]
        arrs["SW_n_mix"] = np.array([[len(mix_by[("BOCPD-PRIOR-EM", "SW")][(h, j)]) for j in range(len(CKPTS["SW"]))] for h in range(nsw)])
        np.savez_compressed(D / "bocpd_test.npz", **arrs, u_all=u, expl_all=ex, ok_all=ok)
        save_json({"selected_H": sel, "selected_name": hname(sel), "n_segment_estimates": n_seg, "n_deployments": n_dep, "wall_s": time.time() - t},
                  D / "bocpd_meta.json")
        log(f"BOCPD test done: {n_dep} deployed strategies from {n_seg} segment estimates")


if __name__ == "__main__":
    main()
