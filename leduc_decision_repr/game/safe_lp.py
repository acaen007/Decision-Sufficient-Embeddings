"""Exact game-value LP and epsilon-safe response LP in sequence form (HiGHS via highspy).

For the "self" player with realization constraints  Ms z = ms, z >= 0,  opponent constraints
Mo y = mo, y >= 0,  and payoff matrix P (self utility = z^T P y):

    security(z) = min_y z^T P y  s.t. Mo y = mo, y >= 0
                = max_v mo^T v   s.t. Mo^T v <= P^T z          (LP duality)

Since mo = (1, 0, ..., 0), mo^T v = v_0.  The epsilon-safe LP is therefore

    max_{z, v}  g^T z
    s.t.        Ms z = ms,  z >= 0
                Mo^T v - P^T z <= 0
                v_0 >= v_star - eps                       (v free otherwise)

and the game-value LP is the same with objective  max v_0  and no bound on v_0.

Player 0 (learner):  Ms = E, Mo = F, P = A.
Player 1 (opponent): Ms = F, Mo = E, P = -A^T.

Sign convention check (see tests): for fixed z the optimal v_0 equals the exact tree
best-response value computed independently in `SequenceForm.best_response_value`.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import highspy

from .sequence_form import SequenceForm, get_sequence_form

INF = highspy.kHighsInf


class SequenceLP:
    def __init__(self, Ms, ms, Mo, mo, P, primal_tol=1e-9, dual_tol=1e-9):
        Ms = sp.csr_matrix(Ms); Mo = sp.csr_matrix(Mo)
        self.n_self = Ms.shape[1]
        self.n_opp = Mo.shape[1]
        self.m_opp = Mo.shape[0]          # 1 + opponent infosets = number of dual variables
        n = self.n_self + self.m_opp
        assert abs(mo[0] - 1.0) < 1e-12 and np.all(mo[1:] == 0)
        # constraint matrix rows: [Ms 0] (eq)  ;  [-P^T  Mo^T] (<= 0)
        top = sp.hstack([Ms, sp.csr_matrix((Ms.shape[0], self.m_opp))])
        bot = sp.hstack([-sp.csr_matrix(P).T, Mo.T])
        M = sp.vstack([top, bot]).tocsc()
        self.n_rows = M.shape[0]
        lp = highspy.HighsLp()
        lp.num_col_ = n
        lp.num_row_ = self.n_rows
        lp.col_cost_ = np.zeros(n)
        lp.col_lower_ = np.concatenate([np.zeros(self.n_self), -INF * np.ones(self.m_opp)])
        lp.col_upper_ = np.concatenate([np.ones(self.n_self), INF * np.ones(self.m_opp)])
        lp.row_lower_ = np.concatenate([np.asarray(ms, float), -INF * np.ones(self.n_opp)])
        lp.row_upper_ = np.concatenate([np.asarray(ms, float), np.zeros(self.n_opp)])
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = M.indptr.astype(np.int32)
        lp.a_matrix_.index_ = M.indices.astype(np.int32)
        lp.a_matrix_.value_ = M.data.astype(np.float64)
        lp.sense_ = highspy.ObjSense.kMinimize
        self.h = highspy.Highs()
        self.h.setOptionValue("output_flag", False)
        self.h.setOptionValue("solver", "simplex")
        self.h.setOptionValue("simplex_strategy", 1)        # dual simplex
        self.h.setOptionValue("primal_feasibility_tolerance", primal_tol)
        self.h.setOptionValue("dual_feasibility_tolerance", dual_tol)
        self.h.setOptionValue("threads", 1)
        self.h.passModel(lp)
        self.v0_col = self.n_self
        self.self_cols = np.arange(self.n_self, dtype=np.int32)
        self.n_fail = 0

    def _set_objective(self, g_self, v0_cost):
        cost = np.zeros(self.n_self + self.m_opp)
        cost[: self.n_self] = -np.asarray(g_self, float)
        cost[self.v0_col] = v0_cost
        self.h.changeColsCost(len(cost), np.arange(len(cost), dtype=np.int32), cost)

    def _run(self):
        self.h.run()
        status = self.h.getModelStatus()
        ok = status == highspy.HighsModelStatus.kOptimal
        if not ok:
            self.n_fail += 1
        sol = self.h.getSolution()
        return ok, np.array(sol.col_value)

    def solve_value(self):
        """Game value for self and an equilibrium realization plan (max v_0)."""
        self.h.changeColBounds(self.v0_col, -INF, INF)
        self._set_objective(np.zeros(self.n_self), -1.0)
        ok, col = self._run()
        assert ok, f"game value LP failed: {self.h.getModelStatus()}"
        z = col[: self.n_self]
        return float(col[self.v0_col]), z

    def solve_safe(self, g, eps, v_star):
        """max g^T z subject to security(z) >= v_star - eps.  Returns (ok, z, v0)."""
        self.h.changeColBounds(self.v0_col, float(v_star - eps), INF)
        self._set_objective(g, 0.0)
        ok, col = self._run()
        return ok, col[: self.n_self], float(col[self.v0_col])

    def solve_unrestricted(self, g):
        """max g^T z without security constraint (unrestricted best response, as LP)."""
        self.h.changeColBounds(self.v0_col, -INF, INF)
        self._set_objective(g, 0.0)
        ok, col = self._run()
        return ok, col[: self.n_self]


class LeducSafeSolver:
    """Learner-side (player 0) safe solver + equilibrium blueprints, with deployed-policy audit."""

    def __init__(self, sf: SequenceForm | None = None):
        self.sf = sf or get_sequence_form()
        S = self.sf
        self.lp0 = SequenceLP(S.E, S.e, S.F, S.f, S.A)
        self.lp1 = SequenceLP(S.F, S.f, S.E, S.e, -S.A.T)
        self.v_star, _ = self.lp0.solve_value()
        v1, _ = self.lp1.solve_value()
        self.v_star_p1 = v1
        assert abs(self.v_star + v1) < 1e-7, (self.v_star, v1)

    # ---- equilibrium selection: eps=0 safe LP maximizing value vs uniform opponent
    def nash_blueprint(self, p: int, symmetrize: bool = True):
        """Equilibrium selection rule: eps=0 safe LP maximizing value vs the uniform opponent,
        then (optionally) suit-symmetrized by group averaging in realization space."""
        S = self.sf
        if p == 0:
            y_u = S.behavioral_to_realization(1, S.uniform_policy(1))
            g = S.A @ y_u
            ok, x, v0 = self.lp0.solve_safe(g, 0.0, self.v_star)
            assert ok
            pol = S.realization_to_behavioral(0, x)
        else:
            x_u = S.behavioral_to_realization(0, S.uniform_policy(0))
            g = -(S.A.T @ x_u)
            ok, y, v0 = self.lp1.solve_safe(g, 0.0, self.v_star_p1)
            assert ok
            pol = S.realization_to_behavioral(1, y)
        if symmetrize:
            from .symmetry import get_symmetry
            pol = get_symmetry().symmetrize_policy(S, p, pol)
        return pol

    def solve_safe(self, g_hat: np.ndarray, eps: float):
        """Return (ok, deployed behavioral policy, deployed realization plan)."""
        ok, x, _ = self.lp0.solve_safe(g_hat, eps, self.v_star)
        pol = self.sf.realization_to_behavioral(0, x)
        x_dep = self.sf.behavioral_to_realization(0, pol)
        return ok, pol, x_dep

    def exploitability(self, x: np.ndarray) -> float:
        """e(x) = v* - min_y x^T A y  (fast exact tree best response)."""
        return self.v_star - self.sf.security(x)


# ---------------------------------------------------------------------- OpenSpiel audit
class OpenSpielAuditor:
    """Independent exploitability audit using OpenSpiel's C++ TabularBestResponse."""

    def __init__(self, sf: SequenceForm | None = None, v_star: float | None = None):
        import pyspiel
        self.sf = sf or get_sequence_form()
        self.game = self.sf.tree.game
        self.pyspiel = pyspiel
        self.v_star = v_star
        self.keys = [self.sf.tree.infosets[p].keys for p in range(2)]
        self.masks = [self.sf.tree.infosets[p].legal_mask for p in range(2)]
        self._br = {}   # cached TabularBestResponse objects (reused via set_policy)

    def tabular_policy(self, p: int, policy: np.ndarray):
        d = {}
        keys, mask = self.keys[p], self.masks[p]
        for I, key in enumerate(keys):
            d[key] = [(int(a), float(policy[I, a])) for a in np.flatnonzero(mask[I])]
        return self.pyspiel.TabularPolicy(d)

    def best_response_value(self, p_responder: int, policy_of_other: np.ndarray) -> float:
        tp = self.tabular_policy(1 - p_responder, policy_of_other)
        if p_responder not in self._br:
            self._br[p_responder] = self.pyspiel.TabularBestResponse(self.game, p_responder, tp)
        else:
            self._br[p_responder].set_policy(tp)
        return float(self._br[p_responder].value(""))

    def exploitability_of_learner(self, policy0: np.ndarray) -> float:
        """e(x) = v* - min_y u0 = v* + max_y u1 (opponent best response value)."""
        return self.v_star + self.best_response_value(1, policy0)

    def learner_br_value(self, policy1: np.ndarray) -> float:
        return self.best_response_value(0, policy1)

    def expected_value(self, policy0: np.ndarray, policy1: np.ndarray) -> float:
        """Expected utility of player 0 via explicit OpenSpiel tree walk (independent of A)."""
        g = self.game
        idx0 = self.sf.tree.infosets[0].index
        idx1 = self.sf.tree.infosets[1].index

        def walk(state, prob):
            if state.is_terminal():
                return prob * state.returns()[0]
            tot = 0.0
            if state.is_chance_node():
                for a, pr in state.chance_outcomes():
                    tot += walk(state.child(a), prob * pr)
                return tot
            p = state.current_player()
            key = state.information_state_string(p)
            pol = policy0[idx0[key]] if p == 0 else policy1[idx1[key]]
            for a in state.legal_actions():
                if pol[a] > 0:
                    tot += walk(state.child(a), prob * pol[a])
            return tot

        return walk(g.new_initial_state(), 1.0)


_SOLVER = None


def get_solver() -> LeducSafeSolver:
    global _SOLVER
    if _SOLVER is None:
        _SOLVER = LeducSafeSolver(get_sequence_form())
    return _SOLVER
