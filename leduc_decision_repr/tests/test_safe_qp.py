"""Differentiable regularized safe QP: safety, LP limit, gradient, torch layer."""
import numpy as np
import torch

from leduc_decision_repr.game.sequence_form import get_sequence_form
from leduc_decision_repr.game.safe_lp import get_solver, OpenSpielAuditor
from leduc_decision_repr.game.safe_qp import SafeQP, make_torch_layer

S = get_sequence_form(); L = get_solver(); QP = SafeQP(S, L.v_star); AUD = OpenSpielAuditor(S, L.v_star)
RNG = np.random.default_rng(5)


def random_g():
    q = S.normalize_policy(1, RNG.dirichlet(np.ones(3), size=S.n_infosets[1]))
    return S.A @ S.behavioral_to_realization(1, q)


def test_regularized_solution_is_exactly_safe_and_feasible():
    for _ in range(4):
        g = random_g()
        for eps in [0.05, 0.1, 0.2]:
            for tau in [0.1, 0.01, 1e-4]:
                sol = QP.solve(g, eps, tau)
                assert sol["ok"]
                x = sol["x"]
                assert np.abs(S.E @ x - S.e).max() < 1e-8 and x.min() > -1e-8
                pol = S.realization_to_behavioral(0, x)
                xd = S.behavioral_to_realization(0, pol)
                assert L.exploitability(xd) <= eps + 1e-6
                assert AUD.exploitability_of_learner(pol) <= eps + 1e-6


def test_lp_limit_and_monotone_in_tau():
    for _ in range(3):
        g = random_g(); eps = 0.1
        ok, xl, _ = L.lp0.solve_safe(g, eps, L.v_star)
        prev = -np.inf
        for tau in [0.1, 0.01, 1e-3, 1e-4]:
            x = QP.solve(g, eps, tau)["x"]
            val = g @ x
            assert val >= prev - 1e-6            # objective value increases as tau decreases
            assert val <= g @ xl + 1e-6          # never above the LP optimum
            prev = val
        assert abs(prev - g @ xl) < 2e-3         # tau = 1e-4 essentially reaches the LP


def test_gradient_matches_finite_differences():
    good = 0; total = 0
    for _ in range(3):
        g = random_g(); w = -random_g(); tau = 0.02
        sol = QP.solve(g, 0.1, tau); grad, _ = QP.grad(sol, w)
        for _ in range(3):
            d = RNG.normal(size=g.shape); d /= np.linalg.norm(d); h = 1e-5
            fd = (w @ QP.solve(g + h * d, 0.1, tau)["x"] - w @ QP.solve(g - h * d, 0.1, tau)["x"]) / (2 * h)
            an = grad @ d
            total += 1; good += abs(an - fd) <= 0.05 * max(abs(fd), 1e-3)
    assert good >= 7, (good, total)              # a few FD probes may straddle an active-set kink


def test_torch_layer_backward():
    F = make_torch_layer(QP)
    g = torch.tensor(np.stack([random_g(), random_g()]), dtype=torch.float64, requires_grad=True)
    x = F.apply(g, np.array([0.1, 0.2]), 0.02)
    assert x.shape == (2, S.n_seq[0])
    w = torch.as_tensor(np.stack([-random_g(), -random_g()]), dtype=torch.float64)
    (x * w).sum().backward()
    assert g.grad is not None and torch.isfinite(g.grad).all() and g.grad.abs().sum() > 0
    # matches the numpy gradient
    sol = QP.solve(g.detach().numpy()[0], 0.1, 0.02); gr, _ = QP.grad(sol, w.numpy()[0])
    assert np.allclose(g.grad.numpy()[0], gr, atol=1e-8)
