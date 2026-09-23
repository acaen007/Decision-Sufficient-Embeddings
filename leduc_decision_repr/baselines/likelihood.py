"""Hidden-card-marginalized likelihood structure of observed hands.

For an observation type t (player-0 view of a completed hand) and a candidate opponent rank r
that is consistent with card removal and with any showdown reveal, the opponent's visited
(rank infoset, action) pairs are fixed.  The likelihood of the hand under an opponent policy q is

    p(t | q) = sum_r  P(r | our rank, public rank)  *  prod_{(I,a) in pairs(t, r)} q(I, a)

up to factors (our own action probabilities, chance probabilities) that do not depend on q.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ..game.leduc_tree import NUM_RANKS, NUM_ACTIONS, TERM_SHOWDOWN, card_rank
from ..data.tokenizer import HandTokenTable, EV_ACTION, EV_SHOWDOWN, EV_PUBLIC_CARD, CAT_FIELDS

CI = {n: i for i, n in enumerate(CAT_FIELDS)}


class HandLikelihood:
    def __init__(self, tab: HandTokenTable, sym):
        self.tab = tab
        self.sym = sym
        self.n_rank = sym.n_rank_infosets[1]
        self.dim = self.n_rank * NUM_ACTIONS
        rows, cols, pair_type, pair_rank, pair_logprior = [], [], [], [], []
        pair = 0
        for t in range(tab.n_types):
            cat = tab.cat[t, : tab.length[t]]
            our = int(cat[0, CI["our_rank"]]) - 1
            pub = -1
            reveal = None
            acts = []
            opp_positions = []       # (index into acts, action, public rank at that time)
            for j in range(len(cat)):
                ev = cat[j, CI["event_type"]]
                if ev == EV_PUBLIC_CARD:
                    pub = int(cat[j, CI["public_rank"]]) - 1
                elif ev == EV_ACTION:
                    a = int(cat[j, CI["action_type"]]) - 1
                    if cat[j, CI["actor"]] == 2:            # opponent
                        opp_positions.append((len(acts), a, pub))
                    acts.append(a)
                elif ev == EV_SHOWDOWN:
                    reveal = int(cat[j, CI["opp_rank"]]) - 1
            counts = np.array([2 - (r == our) - (r == pub) for r in range(NUM_RANKS)], dtype=float)
            total = counts.sum()
            for r in range(NUM_RANKS):
                if counts[r] <= 0:
                    continue
                if reveal is not None and r != reveal:
                    continue
                for (pos, a, pub_at) in opp_positions:
                    key = (r, pub_at, tuple(acts[:pos]))
                    I = sym.rank_key_to_index[1][key]
                    rows.append(pair); cols.append(I * NUM_ACTIONS + a)
                pair_type.append(t); pair_rank.append(r); pair_logprior.append(np.log(counts[r] / total))
                pair += 1
        self.n_pairs = pair
        self.M = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(pair, self.dim))
        self.pair_type = np.array(pair_type)
        self.pair_rank = np.array(pair_rank)
        self.pair_logprior = np.array(pair_logprior)
        # type -> pairs aggregation matrix (T, n_pairs)
        self.S = sp.csr_matrix((np.ones(pair), (self.pair_type, np.arange(pair))), shape=(tab.n_types, pair))

    def pair_loglik(self, log_q_flat: np.ndarray) -> np.ndarray:
        """log_q_flat: (..., dim) -> (..., n_pairs) log-likelihood of each pair (without prior)."""
        return (self.M @ log_q_flat.reshape(-1, self.dim).T).T.reshape(*log_q_flat.shape[:-1], self.n_pairs)

    def type_loglik(self, log_q_flat: np.ndarray) -> np.ndarray:
        """(..., dim) -> (..., T) log p(type | q) including the card prior."""
        ll = self.pair_loglik(log_q_flat) + self.pair_logprior
        out = np.full(ll.shape[:-1] + (self.tab.n_types,), -np.inf)
        # logsumexp within type groups
        m = np.full(ll.shape[:-1] + (self.tab.n_types,), -np.inf)
        np.maximum.at(m.reshape(-1, self.tab.n_types).T, self.pair_type, ll.reshape(-1, self.n_pairs).T)
        mm = m[..., self.pair_type]
        e = np.exp(ll - mm)
        s = (self.S @ e.reshape(-1, self.n_pairs).T).T.reshape(ll.shape[:-1] + (self.tab.n_types,))
        return m + np.log(np.maximum(s, 1e-300))


def type_counts(obs_types: np.ndarray, n_types: int) -> np.ndarray:
    """(n_hist, N) observation-type ids -> (n_hist, T) counts."""
    n = obs_types.shape[0]
    out = np.zeros((n, n_types), dtype=np.float64)
    for h in range(n):
        out[h] = np.bincount(obs_types[h].astype(np.int64), minlength=n_types)
    return out
