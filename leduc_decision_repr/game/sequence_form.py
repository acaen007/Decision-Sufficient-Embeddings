"""Deterministic perfect-recall sequence form for OpenSpiel Leduc.

Sequences of player p: id 0 = empty sequence, id k>0 = (infoset I, action a).
Realization plan x (player 0) / y (player 1) satisfy

    E x = e,  x >= 0        F y = f,  y >= 0

where row 0 of E is x[0] = 1 and row 1+I is  x[parent_seq(I)] - sum_a x[seq(I,a)] = 0.
The payoff matrix A[s0, s1] = sum over terminals z with last sequences (s0, s1) of
chance(z) * u_0(z), so that  u_0(x, y) = x^T A y   and   g(q) = A y_q.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .leduc_tree import LeducTree, get_tree, NUM_ACTIONS, PLAYER, TERMINAL


class SequenceForm:
    def __init__(self, tree: LeducTree | None = None):
        self.tree = tree or get_tree()
        T = self.tree
        self.n_seq = list(T.n_seq)
        self.n_infosets = list(T.n_infosets)
        self.E, self.e = self._constraints(0)
        self.F, self.f = self._constraints(1)
        self.A = self._payoff()
        self.A_csr = sp.csr_matrix(self.A)
        self.AT_csr = sp.csr_matrix(self.A.T)
        # children infosets of each sequence (for tree best response)
        self.child_infosets = []
        for p in range(2):
            ch = [[] for _ in range(self.n_seq[p])]
            for I in range(self.n_infosets[p]):
                ch[T.infosets[p].parent_seq[I]].append(I)
            self.child_infosets.append(ch)

    # ------------------------------------------------------------------ constraints
    def _constraints(self, p: int):
        T = self.tree
        m, n = T.n_infosets[p], T.n_seq[p]
        rows, cols, vals = [0], [0], [1.0]
        for I in range(m):
            rows.append(1 + I); cols.append(T.infosets[p].parent_seq[I]); vals.append(1.0)
            for a in np.flatnonzero(T.infosets[p].legal_mask[I]):
                rows.append(1 + I); cols.append(T.seq_index[p][(I, int(a))]); vals.append(-1.0)
        E = sp.csr_matrix((vals, (rows, cols)), shape=(1 + m, n))
        e = np.zeros(1 + m); e[0] = 1.0
        return E, e

    def _payoff(self):
        T = self.tree
        A = np.zeros((self.n_seq[0], self.n_seq[1]))
        for z in T.terminals:
            s0, s1 = T.seq_before[z]
            A[s0, s1] += T.reach_chance[z] * T.returns0[z]
        return A

    # ------------------------------------------------------------------ conversions
    def behavioral_to_realization(self, p: int, policy: np.ndarray) -> np.ndarray:
        """policy: (n_infosets_p, 3) behavioral probabilities (illegal entries ignored)."""
        T = self.tree
        tab = T.infosets[p]
        x = np.zeros(self.n_seq[p])
        x[0] = 1.0
        for I in T.infoset_order[p]:
            par = x[tab.parent_seq[I]]
            for a in np.flatnonzero(tab.legal_mask[I]):
                x[T.seq_index[p][(I, int(a))]] = par * policy[I, a]
        return x

    def realization_to_behavioral(self, p: int, x: np.ndarray, unreachable="uniform") -> np.ndarray:
        T = self.tree
        tab = T.infosets[p]
        pol = np.zeros((self.n_infosets[p], NUM_ACTIONS))
        for I in range(self.n_infosets[p]):
            legal = np.flatnonzero(tab.legal_mask[I])
            vals = np.array([x[T.seq_index[p][(I, int(a))]] for a in legal])
            vals = np.maximum(vals, 0.0)
            tot = vals.sum()
            if tot > 1e-12:
                pol[I, legal] = vals / tot
            else:
                pol[I, legal] = 1.0 / len(legal)
        return pol

    def normalize_policy(self, p: int, policy: np.ndarray) -> np.ndarray:
        """Mask illegal actions and renormalize rows."""
        mask = self.tree.infosets[p].legal_mask
        pol = np.where(mask, np.maximum(policy, 0.0), 0.0)
        s = pol.sum(1, keepdims=True)
        uni = mask / mask.sum(1, keepdims=True)
        return np.where(s > 1e-300, pol / np.maximum(s, 1e-300), uni)

    def check_realization(self, p: int, x: np.ndarray, tol=1e-9) -> float:
        M, v = (self.E, self.e) if p == 0 else (self.F, self.f)
        return float(np.max(np.abs(M @ x - v)))

    # ------------------------------------------------------------------ values
    def value(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(x @ (self.A @ y))

    def g_of_y(self, y: np.ndarray) -> np.ndarray:
        return self.A @ y

    def best_response_value(self, p: int, c: np.ndarray, maximize: bool) -> float:
        """Exact best response over player p's sequence tree given sequence utilities c.

        Returns the optimal value of sum_s c_s y_s over realization plans of player p,
        maximizing or minimizing.  Used for security values: sec(x) = min over y of
        (A^T x)^T y  (player 1 minimizes learner utility).
        """
        T = self.tree
        tab = T.infosets[p]
        V = np.array(c, dtype=np.float64).copy()
        best = np.max if maximize else np.min
        for I in T.infoset_order[p][::-1]:
            legal = np.flatnonzero(tab.legal_mask[I])
            vals = [V[T.seq_index[p][(I, int(a))]] for a in legal]
            V[tab.parent_seq[I]] += best(vals)
        return float(V[0])

    def best_response_policy(self, p: int, c: np.ndarray, maximize: bool):
        """Pure best-response behavioral policy (and its value) of player p to sequence utilities c."""
        T = self.tree
        tab = T.infosets[p]
        V = np.array(c, dtype=np.float64).copy()
        pol = np.zeros((self.n_infosets[p], NUM_ACTIONS))
        for I in T.infoset_order[p][::-1]:
            legal = np.flatnonzero(tab.legal_mask[I])
            vals = np.array([V[T.seq_index[p][(I, int(a))]] for a in legal])
            k = int(np.argmax(vals) if maximize else np.argmin(vals))
            pol[I, legal[k]] = 1.0
            V[tab.parent_seq[I]] += vals[k]
        return pol, float(V[0])

    def security(self, x: np.ndarray) -> float:
        """min_y x^T A y over opponent realization plans (learner's guaranteed value)."""
        return self.best_response_value(1, self.A.T @ x, maximize=False)

    def learner_br_value(self, y: np.ndarray) -> float:
        """max_x x^T A y (unrestricted best response value against opponent y)."""
        return self.best_response_value(0, self.A @ y, maximize=True)

    def uniform_policy(self, p: int) -> np.ndarray:
        mask = self.tree.infosets[p].legal_mask
        return mask / mask.sum(1, keepdims=True)


_SF = None


def get_sequence_form() -> SequenceForm:
    global _SF
    if _SF is None:
        _SF = SequenceForm(get_tree())
    return _SF
