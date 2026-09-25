"""Shared pieces for the decision-compression study (REPORT_LEDUC_DECISION_COMPRESSION.md): opponent sets, exact
hand-type distributions, pooled fractions, and a parallel exact-LP + OpenSpiel-audit pool."""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import time, multiprocessing as mp
import numpy as np
from .common import OUT

D = OUT / "dcomp"; EPS_LIST = [0.05, 0.10, 0.20]; EPS_IDX = {0.0: 0, 0.05: 1, 0.10: 2, 0.20: 3}
OOD_FAMS = ["NEAR", "FAR-ARCH", "FAR-CFR", "FAR-EXPL"]


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.0f}s] {msg}", flush=True)


def load_sets():
    """Opponent sets: rank policies Q, true g, V0, and V_eps by eps (OOD: eps = 0.10 only)."""
    from .data.datasets import load_population
    pop = load_population(); S = {}
    for name, sid in (("train", 0), ("val", 1), ("test", 2)):
        ids = np.flatnonzero(pop["split"] == sid)
        S[name] = {"ids": ids, "Q": pop["rank_policies"][ids], "G": pop["G"][ids], "V0": pop["V_oracle"][ids, 0],
                   "V": {e: pop["V_oracle"][ids, EPS_IDX[e]] for e in EPS_LIST}, "X": {e: pop["X_oracle"][ids, EPS_IDX[e]] for e in EPS_LIST},
                   "family": pop["family_index"][ids]}
    GF = np.load(OUT / "gen" / "families.npz", allow_pickle=True); m = np.isin(GF["family"], OOD_FAMS); ids = np.flatnonzero(m)
    S["ood"] = {"ids": ids, "Q": GF["rank_policies"][ids], "G": GF["G"][ids], "V0": GF["V0"][ids], "V": {0.10: GF["Veps"][ids]},
                "family": GF["family"][ids]}
    return S, pop


class HandDist:
    """Exact distribution over the reachable observation types of one hand (learner plays its blueprint):
    p_q(o) = C(o) * L(o | q), with C recovered once from an exact tree traversal."""
    def __init__(self, pop):
        from .game.leduc_tree import get_tree
        from .game.symmetry import get_symmetry
        from .data.tokenizer import get_token_table
        from .baselines.likelihood import HandLikelihood
        from .cpd_eval import type_dist
        tree, self.sym, tab = get_tree(), get_symmetry(), get_token_table(); self.lik = HandLikelihood(tab, self.sym)
        mask = self.sym.rank_legal_mask[1]; uni = mask / mask.sum(1, keepdims=True)
        pu = type_dist(tree, tab, pop["blueprint0"], self.sym.expand(1, uni)); self.reach = pu > 0
        lu = self.lik.type_loglik(np.log(np.maximum(uni, 1e-300)).reshape(1, -1))[0]
        self.logC = (np.log(pu[self.reach]) - lu[self.reach])
        q1 = pop["rank_policies"][7]; p1 = type_dist(tree, tab, pop["blueprint0"], self.sym.expand(1, q1))
        assert np.abs(self(q1[None])[0] - p1[self.reach]).max() < 1e-12

    def __call__(self, Q):
        ll = self.lik.type_loglik(np.log(np.maximum(Q, 1e-300)).reshape(len(Q), -1))[:, self.reach]
        P = np.exp(ll + self.logC); return P / P.sum(1, keepdims=True)


def kl_rows(P, Qd):
    return (P * (np.log(np.maximum(P, 1e-300)) - np.log(np.maximum(Qd, 1e-300)))).sum(1)


def js_matrix(P):
    """Pairwise Jensen-Shannon divergence (nats) for rows of P."""
    n = len(P); out = np.zeros((n, n)); lp = np.log(np.maximum(P, 1e-300)); H = (P * lp).sum(1)
    for i in range(n):
        M = 0.5 * (P[i][None] + P); lM = np.log(np.maximum(M, 1e-300))
        out[i] = 0.5 * (H[i] - (P[i][None] * lM).sum(1)) + 0.5 * (H - (P * lM).sum(1))
    return out


def policy_kl(Q, Qh, legal):
    """Mean over infosets of KL(q_I || qh_I) per opponent."""
    t = np.where(legal[None], Q * (np.log(np.maximum(Q, 1e-300)) - np.log(np.maximum(Qh, 1e-300))), 0.0)
    return t.sum(2).mean(1)


def fraction(u, V0, Ve):
    hd = Ve - V0; keep = hd >= 0.01
    return float((u - V0)[keep].sum() / hd[keep].sum())


def boot_fraction(u, V0, Ve, B=2000, seed=0):
    rng = np.random.default_rng(seed); hd = Ve - V0; keep = np.flatnonzero(hd >= 0.01); num = (u - V0)[keep]; den = hd[keep]
    ii = rng.integers(0, len(keep), (B, len(keep))); bs = num[ii].sum(1) / den[ii].sum(1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


# ------------------------------------------------------------------ exact LP + audit pool
_W = {}


def _init(audit):
    from .game.safe_lp import get_solver, OpenSpielAuditor
    from .game.sequence_form import get_sequence_form
    _W["sf"] = get_sequence_form(); _W["L"] = get_solver(); _W["aud"] = OpenSpielAuditor(_W["sf"], _W["L"].v_star) if audit else None


def _chunk(args):
    kind, arr, eps = args; L, aud, sf = _W["L"], _W["aud"], _W["sf"]; n = len(arr)
    X = np.zeros((n, arr.shape[1])); ex = np.full(n, np.nan); ok = np.ones(n, bool)
    for i in range(n):
        if kind == "solve":
            okk, pol, x = L.solve_safe(np.asarray(arr[i], dtype=np.float64), eps); ok[i] = okk; X[i] = x
        else:                                                           # audit a given realization plan
            x = np.asarray(arr[i], dtype=np.float64); pol = sf.realization_to_behavioral(0, x); X[i] = x
        if aud is not None:
            ex[i] = aud.exploitability_of_learner(pol)
    return X, ex, ok


class LPPool:
    def __init__(self, workers=4, audit=True):
        self.pool = mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(audit,)); self.workers = workers

    def run(self, kind, arr, eps=0.10, chunk=25):
        if len(arr) == 0:
            return np.zeros((0, 1093)), np.zeros(0), np.zeros(0, bool)
        jobs = [(kind, arr[i:i + chunk], eps) for i in range(0, len(arr), chunk)]
        res = self.pool.map(_chunk, jobs)
        return np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res]), np.concatenate([r[2] for r in res])

    def solve(self, G_hat, eps):
        return self.run("solve", G_hat, eps)

    def audit(self, X):
        return self.run("audit", X)

    def close(self):
        self.pool.close(); self.pool.join()
