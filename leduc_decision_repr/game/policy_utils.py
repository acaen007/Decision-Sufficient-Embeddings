"""Policy-level utilities: structured infoset features, distances, participation ratio."""
from __future__ import annotations

import numpy as np

from .leduc_tree import LeducTree, get_tree, NUM_ACTIONS, MAX_POT, MAX_CONTRIB


INFOSET_FEATURE_NAMES = [
    "rank_J", "rank_Q", "rank_K",
    "round2",
    "pub_none", "pub_J", "pub_Q", "pub_K",
    "paired", "rank_above_public", "rank_below_public",
    "facing_raise", "to_call_norm", "raises_this_round_1", "raises_this_round_2",
    "pot_norm", "own_contrib_norm", "opp_contrib_norm",
    "n_raises_prev_round", "first_to_act_this_round",
    "legal_fold", "legal_raise",
    "hand_strength",
]


def infoset_features(tree: LeducTree, p: int) -> np.ndarray:
    """Structured features of every infoset of player p (m, F). Values roughly in [-1, 1]."""
    tab = tree.infosets[p]
    m = len(tab)
    F = np.zeros((m, len(INFOSET_FEATURE_NAMES)), dtype=np.float64)
    for I in range(m):
        r = tab.private_rank[I]
        pub = tab.public_rank[I]
        F[I, r] = 1.0
        F[I, 3] = float(tab.round[I] == 2)
        F[I, 4 + (pub + 1)] = 1.0            # pub_none / pub_J / pub_Q / pub_K
        if pub >= 0:
            F[I, 8] = float(r == pub)
            F[I, 9] = float(r > pub)
            F[I, 10] = float(r < pub)
        F[I, 11] = float(tab.facing_raise[I])
        F[I, 12] = tab.to_call[I] / 4.0
        F[I, 13] = float(tab.raises_this_round[I] == 1)
        F[I, 14] = float(tab.raises_this_round[I] == 2)
        F[I, 15] = tab.pot[I] / MAX_POT
        F[I, 16] = tab.own_contrib[I] / MAX_CONTRIB
        F[I, 17] = tab.opp_contrib[I] / MAX_CONTRIB
        F[I, 18] = tab.n_raises_prev_round[I] / 2.0
        F[I, 19] = float(tab.n_actions_this_round[I] == 0)
        F[I, 20] = float(tab.legal_mask[I, 0])
        F[I, 21] = float(tab.legal_mask[I, 2])
        # crude hand strength in [-1, 1]: pair best, then rank; before flop just rank
        if pub >= 0 and r == pub:
            strength = 1.0
        else:
            strength = (r - 1.0) * 0.6
        F[I, 22] = strength
    return F


def rank_infoset_features(tree: LeducTree, sym, p: int) -> np.ndarray:
    """Features of the rank-level infosets (n_rank, F): representative physical member."""
    phys = infoset_features(tree, p)
    return np.stack([phys[m[0]] for m in sym.rank_members[p]])


def participation_ratio(X: np.ndarray, center: bool = True) -> float:
    """PR = (sum lambda)^2 / sum lambda^2 of the covariance eigenvalues (linear diagnostic)."""
    X = np.asarray(X, dtype=np.float64)
    if center:
        X = X - X.mean(0, keepdims=True)
    if X.shape[0] < 2:
        return float("nan")
    s = np.linalg.svd(X, compute_uv=False)
    lam = s ** 2
    if lam.sum() <= 0:
        return 0.0
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def pca_variance_fractions(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    lam = s ** 2
    return lam / lam.sum()


def dims_for_variance(X: np.ndarray, fracs=(0.9, 0.95, 0.99)) -> dict:
    v = np.cumsum(pca_variance_fractions(X))
    return {f"dims_{int(f*100)}": int(np.searchsorted(v, f) + 1) for f in fracs}


def behavioral_distance_matrix(policies: np.ndarray, legal_mask: np.ndarray,
                               weights: np.ndarray | None = None) -> np.ndarray:
    """Pairwise L2 distance between policy tables over legal entries (optionally infoset-weighted).

    policies: (K, m, 3); weights: (m,) infoset weights (uniform if None), normalized to sum 1.
    Returns (K, K) distances  d(q, q') = sqrt( sum_I w_I * ||q(I) - q'(I)||^2 ).
    """
    K, m, _ = policies.shape
    w = np.ones(m) / m if weights is None else weights / weights.sum()
    X = (policies * legal_mask[None]) * np.sqrt(w)[None, :, None]
    X = X.reshape(K, -1)
    sq = (X ** 2).sum(1)
    D2 = sq[:, None] + sq[None, :] - 2 * X @ X.T
    return np.sqrt(np.maximum(D2, 0.0))


def infoset_reach_weights(tree: LeducTree, sf, x0: np.ndarray, y1: np.ndarray, p: int) -> np.ndarray:
    """Probability of reaching each infoset of player p under (x0, y1) realization plans + chance."""
    T = tree
    w = np.zeros(T.n_infosets[p])
    for i in range(T.n_nodes):
        if T.node_type[i] == 1 and T.player[i] == p:
            s0, s1 = T.seq_before[i]
            w[T.infoset_of_node[i]] += T.reach_chance[i] * x0[s0] * y1[s1]
    return w


class RankPolicyToG:
    """Vectorized map from rank-level opponent policies (n, 144, 3) to decision vectors g = A y.

    Each physical opponent sequence s is the product of the opponent's (rank infoset, action)
    probabilities along its prefix (at most 4 factors).  Index tables are exposed so the same
    computation can be reproduced differentiably in torch.
    """

    def __init__(self, tree: LeducTree, sf, sym):
        T = tree
        n1 = T.n_seq[1]
        self.max_len = 4
        self.idx = np.zeros((n1, self.max_len), dtype=np.int64)   # flat index into (144*3), or -1
        self.valid = np.zeros((n1, self.max_len), dtype=bool)
        for s in range(1, n1):
            chain = []
            cur = s
            while cur != 0:
                I = T.seq_infoset[1][cur]; a = T.seq_action[1][cur]
                chain.append(int(sym.rank_infoset_of[1][I]) * NUM_ACTIONS + int(a))
                cur = T.infosets[1].parent_seq[I]
            chain = chain[::-1]
            assert len(chain) <= self.max_len
            self.idx[s, : len(chain)] = chain
            self.valid[s, : len(chain)] = True
        self.A_T = np.ascontiguousarray(sf.A.T)          # (n1, n0)
        self.n1 = n1

    def realization(self, rank_policies: np.ndarray) -> np.ndarray:
        """(n, 144, 3) -> (n, n1) opponent realization plans."""
        Q = rank_policies.reshape(rank_policies.shape[0], -1)          # (n, 432)
        fac = np.where(self.valid[None], Q[:, self.idx], 1.0)          # (n, n1, 4)
        return fac.prod(axis=2)

    def g(self, rank_policies: np.ndarray) -> np.ndarray:
        return self.realization(rank_policies) @ self.A_T
