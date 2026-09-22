"""Tabular EM estimate of the opponent's rank-level policy from observed hands.

Hidden variable: the opponent's private rank in each hand (never revealed after a fold).
E-step: posterior over consistent ranks given current q; M-step: Dirichlet-MAP update
q(I, a) = (expected counts + alpha * prior(I, a)) / (expected visits + alpha) over legal
actions.  `prior` is uniform over legal actions ("uniform") or the opponent Nash policy
("nash"); alpha is the pseudo-count mass per infoset.  Vectorized over histories.
"""
from __future__ import annotations

import numpy as np

from ..game.leduc_tree import NUM_ACTIONS
from .likelihood import HandLikelihood


class TabularEM:
    def __init__(self, lik: HandLikelihood, legal_mask: np.ndarray, prior_policy: np.ndarray,
                 alpha: float = 1.0, n_iter: int = 200, tol: float = 1e-7):
        self.lik = lik
        self.mask = legal_mask.astype(bool)                 # (144, 3)
        self.prior = prior_policy * self.mask
        self.prior = self.prior / self.prior.sum(1, keepdims=True)
        self.alpha = alpha
        self.n_iter = n_iter
        self.tol = tol

    def fit(self, counts: np.ndarray, return_loglik: bool = False):
        """counts: (n_hist, T) type counts -> (n_hist, 144, 3) policy estimates."""
        lik = self.lik
        n = counts.shape[0]
        Q = np.broadcast_to(self.prior, (n,) + self.prior.shape).copy()
        c_pair = counts[:, lik.pair_type]                    # (n, n_pairs)
        alpha_prior = self.alpha * self.prior                # (144, 3)
        logliks = []
        prev = None
        for it in range(self.n_iter):
            logq = np.log(np.maximum(Q, 1e-300)).reshape(n, -1)
            ll = lik.pair_loglik(logq) + lik.pair_logprior[None]                       # (n, n_pairs)
            # normalize within type groups
            m = np.full((n, lik.tab.n_types), -np.inf)
            np.maximum.at(m.T, lik.pair_type, ll.T)
            w = np.exp(ll - m[:, lik.pair_type])
            z = (lik.S @ w.T).T                                                          # (n, T)
            w = w / np.maximum(z[:, lik.pair_type], 1e-300)
            if return_loglik:
                logliks.append((counts * (m + np.log(np.maximum(z, 1e-300)))).sum(1))
            exp_counts = (lik.M.T @ (w * c_pair).T).T.reshape(n, -1, NUM_ACTIONS)     # (n, 144, 3)
            num = exp_counts + alpha_prior[None]
            den = num.sum(2, keepdims=True)
            Qn = np.where(self.mask[None], num / den, 0.0)
            delta = np.abs(Qn - Q).max()
            Q = Qn
            if delta < self.tol:
                break
        if return_loglik:
            return Q, np.array(logliks)
        return Q
