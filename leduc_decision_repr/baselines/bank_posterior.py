"""Train-bank posterior baseline: exact Bayesian averaging over the training opponents.

p(q_k | H) ∝ p(H | q_k) p(q_k) with a uniform prior over the training bank; p(H | q_k) is the
exact hidden-card-marginalized likelihood (factors independent of q_k cancel).  The decision
vector is the posterior mean  g_bar = sum_k p(q_k | H) g(q_k).
"""
from __future__ import annotations

import numpy as np

from .likelihood import HandLikelihood


class BankPosterior:
    def __init__(self, lik: HandLikelihood, bank_rank_policies: np.ndarray, bank_G: np.ndarray):
        self.lik = lik
        K = bank_rank_policies.shape[0]
        logq = np.log(np.maximum(bank_rank_policies.reshape(K, -1), 1e-300))
        self.type_ll = lik.type_loglik(logq)                 # (K, T)
        self.bank_G = bank_G                                 # (K, n0)
        self.K = K

    def log_posterior(self, counts: np.ndarray) -> np.ndarray:
        """counts: (n_hist, T) -> (n_hist, K) log posterior (normalized)."""
        ll = counts @ self.type_ll.T                         # (n_hist, K)
        ll = ll - ll.max(1, keepdims=True)
        return ll - np.log(np.exp(ll).sum(1, keepdims=True))

    def g_bar(self, counts: np.ndarray):
        post = np.exp(self.log_posterior(counts))
        return post @ self.bank_G, post
