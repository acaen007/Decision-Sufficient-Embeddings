"""Differentiable regularized epsilon-safe response (training-time only).

Solves, for a predicted decision vector g_hat and safety budget eps,

    max_{x, v}  g_hat^T x - tau/2 ||x||^2 - tau_v/2 ||v||^2
    s.t.        E x = e,  x >= 0,  F^T v - A^T x <= 0,  v_0 >= v* - eps

i.e. the exact epsilon-safe LP of `safe_lp.py` with a strictly concave quadratic regularizer
on x (tau) and a negligible regularizer on the dual-side variables v (tau_v << tau) so that the
optimum is unique and the KKT system is non-singular.  The feasible set is *identical* to the LP's,
so every regularized solution is exactly epsilon-safe; only the objective is smoothed.

Forward: Clarabel (interior point).  Backward: implicit differentiation of the KKT conditions on
the active set (OptNet-style), solved with a sparse LU.  For the loss  L = <w, x*(g_hat)>  the
gradient is  dL/dg_hat = P_x (H^{-1} w~ - H^{-1} G_a^T K^{-1} G_a H^{-1} w~),  with
K = G_a H^{-1} G_a^T over the active constraints G_a, H = diag(tau I, tau_v I), w~ = (w, 0).

The deployed strategy at evaluation time never comes from this solver: `safe_lp.py` is used.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import clarabel

from .sequence_form import SequenceForm, get_sequence_form


class SafeQP:
    def __init__(self, sf: SequenceForm | None = None, v_star: float | None = None, tol: float = 1e-8,
                 tau_v_ratio: float = 1e-4, kkt_shift: float = 1e-9):
        self.sf = sf or get_sequence_form()
        S = self.sf
        if v_star is None:
            from .safe_lp import get_solver
            v_star = get_solver().v_star
        self.v_star = float(v_star)
        self.n0, self.n1 = S.n_seq[0], S.n_seq[1]
        self.m1 = S.F.shape[0]                      # 1 + opponent infosets = number of v variables
        self.nz = self.n0 + self.m1
        self.tol = tol
        self.tau_v_ratio = tau_v_ratio
        self.kkt_shift = kkt_shift
        # equality block [E 0] z = e
        self.Eq = sp.hstack([S.E, sp.csr_matrix((S.E.shape[0], self.m1))]).tocsc()
        # inequality block  G z + s = h, s >= 0:  [-A^T F^T] z <= 0 ;  -x <= 0 ;  -v_0 <= -(v*-eps)
        safety = sp.hstack([-S.A_csr.T, S.F.T])
        nonneg = sp.hstack([-sp.eye(self.n0), sp.csr_matrix((self.n0, self.m1))])
        bound = sp.csr_matrix(([-1.0], ([0], [self.n0])), shape=(1, self.nz))
        self.G = sp.vstack([safety, nonneg, bound]).tocsc()
        self.n_eq = self.Eq.shape[0]; self.n_in = self.G.shape[0]
        self.Acone = sp.vstack([self.Eq, self.G]).tocsc()
        self.cones = [clarabel.ZeroConeT(self.n_eq), clarabel.NonnegativeConeT(self.n_in)]
        self.b_base = np.concatenate([S.e, np.zeros(self.n_in)])
        self.i_safety = np.arange(0, self.n1)
        self.i_nonneg = np.arange(self.n1, self.n1 + self.n0)
        self.i_bound = self.n1 + self.n0
        self.n_fail = 0

    # ------------------------------------------------------------------ forward
    def solve(self, g: np.ndarray, eps: float, tau: float, tau_v: float | None = None):
        """Returns dict with x (n0,), v (m1,), lam (n_in,), s (n_in,), status, iterations."""
        tau_v = tau * self.tau_v_ratio if tau_v is None else tau_v
        P = sp.diags(np.concatenate([tau * np.ones(self.n0), tau_v * np.ones(self.m1)])).tocsc()
        q = np.concatenate([-np.asarray(g, dtype=np.float64), np.zeros(self.m1)])
        b = self.b_base.copy(); b[-1] = -(self.v_star - eps)
        st = clarabel.DefaultSettings(); st.verbose = False
        st.tol_gap_abs = self.tol; st.tol_gap_rel = self.tol; st.tol_feas = self.tol
        st.max_iter = 200
        solver = clarabel.DefaultSolver(P, q, self.Acone, b, self.cones, st)
        sol = solver.solve()
        status = str(sol.status)
        ok = status in ("Solved", "AlmostSolved")
        if not ok:
            self.n_fail += 1
        z = np.array(sol.x); lam = np.array(sol.z)[self.n_eq:]; s = np.array(sol.s)[self.n_eq:]
        return {"x": z[: self.n0], "v": z[self.n0:], "z": z, "lam": lam, "s": s, "status": status,
                "iterations": int(sol.iterations), "ok": ok, "tau": tau, "tau_v": tau_v}

    # ------------------------------------------------------------------ backward
    def active_set(self, sol, rel: float = 1.0):
        """Inequality rows deemed active: dual larger than slack (interior-point heuristic)."""
        return sol["lam"] > rel * sol["s"]

    def grad(self, sol, w: np.ndarray, active=None):
        """d<w, x*>/d g_hat for the fixed active set (w has shape (n0,))."""
        tau, tau_v = sol["tau"], sol["tau_v"]
        act = self.active_set(sol) if active is None else active
        Ga = sp.vstack([self.Eq, self.G[act]]).tocsc()
        hinv = np.concatenate([np.ones(self.n0) / tau, np.ones(self.m1) / tau_v])
        wt = np.concatenate([np.asarray(w, dtype=np.float64), np.zeros(self.m1)])
        Hinv_w = hinv * wt
        K = (Ga @ sp.diags(hinv) @ Ga.T).tocsc()
        K = K + sp.eye(K.shape[0]) * (self.kkt_shift * max(1.0 / tau, 1.0))
        rhs = Ga @ Hinv_w
        try:
            lu = spla.splu(K)
            mu = lu.solve(rhs)
        except RuntimeError:
            mu = spla.lsqr(K, rhs, atol=1e-12, btol=1e-12)[0]
        dz_dq_T_w = -(Hinv_w - hinv * (Ga.T @ mu))          # dL/dq  (q = (-g, 0))
        return -dz_dq_T_w[: self.n0], int(act.sum())        # dL/dg_hat

    # ------------------------------------------------------------------ diagnostics
    def policy_stats(self, x: np.ndarray):
        """Behavioral determinism / entropy of the realization plan x (reachable infosets only)."""
        S = self.sf
        pol = S.realization_to_behavioral(0, x)
        tab = S.tree.infosets[0]
        reach = x[tab.parent_seq]
        m = reach > 1e-6
        p = np.where(tab.legal_mask, np.maximum(pol, 1e-300), 1.0)
        ent = -(np.where(tab.legal_mask, pol * np.log(p), 0.0)).sum(1)
        det = pol.max(1) > 0.999
        return {"mean_entropy_reachable": float(ent[m].mean()), "frac_deterministic_reachable": float(det[m].mean()),
                "n_reachable": int(m.sum()), "frac_x_zero": float((x < 1e-8).mean())}


# ---------------------------------------------------------------------- torch layer
def make_torch_layer(qp: SafeQP):
    import torch

    class SafeQPFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, g_hat, eps_vec, tau):
            g_np = g_hat.detach().cpu().numpy().astype(np.float64)
            B = g_np.shape[0]
            xs = np.zeros((B, qp.n0)); sols = []
            for i in range(B):
                sol = qp.solve(g_np[i], float(eps_vec[i]), float(tau))
                xs[i] = sol["x"]; sols.append(sol)
            ctx.sols = sols
            ctx.n_active = [int(qp.active_set(s).sum()) for s in sols]
            return torch.as_tensor(xs, dtype=g_hat.dtype)

        @staticmethod
        def backward(ctx, grad_out):
            w = grad_out.detach().cpu().numpy().astype(np.float64)
            grads = np.zeros((len(ctx.sols), qp.n0))
            for i, sol in enumerate(ctx.sols):
                if sol["ok"]:
                    grads[i], _ = qp.grad(sol, w[i])
            return torch.as_tensor(grads, dtype=grad_out.dtype), None, None

    return SafeQPFunction
