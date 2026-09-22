"""Game-theoretic test suite (task section 21, items 1-7)."""
import numpy as np
import pytest
import pyspiel
from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import exploitability as os_exploit

from leduc_decision_repr.game.leduc_tree import get_tree, TERMINAL, PLAYER, CHANCE
from leduc_decision_repr.game.sequence_form import get_sequence_form
from leduc_decision_repr.game.safe_lp import get_solver, OpenSpielAuditor

T = get_tree()
S = get_sequence_form()
L = get_solver()
AUD = OpenSpielAuditor(S, L.v_star)
RNG = np.random.default_rng(12345)


def random_policy(p, rng=RNG, alpha=1.0):
    return S.normalize_policy(p, rng.dirichlet(alpha * np.ones(3), size=S.n_infosets[p]))


def os_tabular(p, pol):
    tp = policy_lib.TabularPolicy(T.game)
    for I, key in enumerate(T.infosets[p].keys):
        tp.action_probability_array[tp.state_lookup[key]] = pol[I]
    return tp


# ---------------------------------------------------------------- tree sanity
def test_tree_matches_openspiel_public_state():
    mask = T.node_type != TERMINAL
    assert np.all(T.pot[mask] == T.os_pot[mask])
    assert np.all((100 - T.contrib)[mask] == T.os_money[mask])
    assert T.n_nodes == 9457 and T.n_terminals == 5520
    assert T.n_infosets == [468, 468]


def test_tree_action_deltas_match_openspiel_money():
    for i in np.flatnonzero(T.node_type == PLAYER)[::7]:
        st = T.state_of_node(i)
        for a in T.legal[i]:
            child = st.child(a)
            s = str(child)
            import re
            m = re.search(r"Money \(player_0 player_1\): (\d+) (\d+)", s)
            money_after = (int(m.group(1)), int(m.group(2)))
            p = T.player[i]
            c = T.child_table[i, a]
            if T.node_type[c] == TERMINAL:
                # OpenSpiel reports post-settlement money at terminals; check payoff instead:
                # the loser pays exactly its own contribution, ties pay nothing.
                u = T.returns0[c]
                c0, c1 = T.contrib[c]
                assert u in (c1, -c0, 0.0), (u, c0, c1)
                if a == 0:   # fold: folder loses its own contribution
                    assert u == (-c0 if p == 0 else c1)
                continue
            delta_os = (100 - money_after[p]) - T.contrib[i, p]
            assert delta_os == T.node_action_delta(i, a)


# ---------------------------------------------------------------- 1. realization constraints
def test_realization_constraints_random_policies():
    for _ in range(50):
        for p in range(2):
            pol = random_policy(p)
            x = S.behavioral_to_realization(p, pol)
            assert S.check_realization(p, x) < 1e-12
            assert np.all(x >= 0) and abs(x[0] - 1) < 1e-12


# ---------------------------------------------------------------- 2. behavioral <-> realization
def test_behavioral_realization_roundtrip():
    for _ in range(30):
        for p in range(2):
            pol = random_policy(p)
            x = S.behavioral_to_realization(p, pol)
            back = S.realization_to_behavioral(p, x)
            # all infosets are reachable under a fully mixed policy
            assert np.abs(back - pol).max() < 1e-10
    # a deterministic policy leaves infosets unreachable; they must come back uniform
    pol = np.zeros((S.n_infosets[0], 3)); pol[:, 1] = 1.0
    x = S.behavioral_to_realization(0, pol)
    back = S.realization_to_behavioral(0, x)
    mask = T.infosets[0].legal_mask
    unreachable = x[[T.seq_index[0][(I, 1)] for I in range(S.n_infosets[0])]] == 0
    uni = mask / mask.sum(1, keepdims=True)
    assert np.allclose(back[unreachable], uni[unreachable])
    assert np.allclose(back[~unreachable], pol[~unreachable])


# ---------------------------------------------------------------- 3. payoff matrix
def test_payoff_matrix_matches_openspiel_tree_walk():
    for _ in range(12):
        p0, p1 = random_policy(0), random_policy(1)
        x, y = S.behavioral_to_realization(0, p0), S.behavioral_to_realization(1, p1)
        v_seq = S.value(x, y)
        v_walk = AUD.expected_value(p0, p1)
        assert abs(v_seq - v_walk) < 1e-10
    # also against OpenSpiel's own expected-game-score style check via deterministic policies
    for _ in range(5):
        p0 = np.zeros((S.n_infosets[0], 3)); p1 = np.zeros((S.n_infosets[1], 3))
        for I in range(S.n_infosets[0]):
            p0[I, RNG.choice(np.flatnonzero(T.infosets[0].legal_mask[I]))] = 1
        for I in range(S.n_infosets[1]):
            p1[I, RNG.choice(np.flatnonzero(T.infosets[1].legal_mask[I]))] = 1
        x, y = S.behavioral_to_realization(0, p0), S.behavioral_to_realization(1, p1)
        assert abs(S.value(x, y) - AUD.expected_value(p0, p1)) < 1e-10


def test_reach_probabilities_sum_to_one():
    x = S.behavioral_to_realization(0, random_policy(0))
    y = S.behavioral_to_realization(1, random_policy(1))
    tot = sum(T.reach_chance[z] * x[T.seq_before[z, 0]] * y[T.seq_before[z, 1]] for z in T.terminals)
    assert abs(tot - 1.0) < 1e-12


# ---------------------------------------------------------------- 4. decision-vector identity
def test_decision_vector_identity():
    for _ in range(20):
        x = S.behavioral_to_realization(0, random_policy(0))
        y = S.behavioral_to_realization(1, random_policy(1))
        g = S.g_of_y(y)
        assert abs(x @ g - S.value(x, y)) < 1e-12


# ---------------------------------------------------------------- 5. game value
def test_game_value_matches_openspiel():
    # OpenSpiel's own sequence-form LP (cvxpy/ECOS) and the literature value -0.085606...
    from open_spiel.python.algorithms import sequence_form_lp
    v0, v1, _, _ = sequence_form_lp.solve_zero_sum_game(T.game)
    assert abs(L.v_star - v0) < 1e-6
    assert abs(L.v_star + 0.0856064) < 1e-6
    # equilibrium blueprints have (numerically) zero exploitability
    bp0, bp1 = L.nash_blueprint(0), L.nash_blueprint(1)
    x, y = S.behavioral_to_realization(0, bp0), S.behavioral_to_realization(1, bp1)
    assert L.exploitability(x) < 1e-9
    assert S.learner_br_value(y) - L.v_star < 1e-9
    # CFR converges to the same value (independent algorithm)
    from open_spiel.python.algorithms import cfr
    solver = cfr.CFRPlusSolver(T.game)
    for _ in range(300):
        solver.evaluate_and_update_policy()
    avg = solver.average_policy()
    conv = os_exploit.nash_conv(T.game, avg)
    assert conv < 0.02
    # value of the CFR average policy pair is within nash_conv of v*
    pol0 = np.zeros((S.n_infosets[0], 3)); pol1 = np.zeros((S.n_infosets[1], 3))
    for I, key in enumerate(T.infosets[0].keys):
        pol0[I] = avg.action_probability_array[avg.state_lookup[key]]
    for I, key in enumerate(T.infosets[1].keys):
        pol1[I] = avg.action_probability_array[avg.state_lookup[key]]
    v = S.value(S.behavioral_to_realization(0, pol0), S.behavioral_to_realization(1, pol1))
    assert abs(v - L.v_star) < conv


# ---------------------------------------------------------------- 6. exploitability
def test_best_response_matches_openspiel():
    for _ in range(10):
        p0 = random_policy(0)
        x = S.behavioral_to_realization(0, p0)
        mine = L.exploitability(x)
        os_val = AUD.exploitability_of_learner(p0)
        assert abs(mine - os_val) < 1e-9
        # python exploitability (both players, uniform for the other side): BR value of p1
        tp = os_tabular(0, p0)
        br_py = os_exploit.best_response(T.game, tp, 1)["best_response_value"]
        assert abs(L.v_star + br_py - mine) < 1e-9
    for _ in range(5):
        p1 = random_policy(1)
        y = S.behavioral_to_realization(1, p1)
        assert abs(S.learner_br_value(y) - AUD.learner_br_value(p1)) < 1e-9


def test_dual_security_matches_tree_best_response():
    """For fixed x the dual LP's optimal v_0 must equal min_y x^T A y (sign convention check)."""
    import highspy
    from leduc_decision_repr.game.safe_lp import SequenceLP
    for _ in range(4):
        pol = random_policy(0)
        x = S.behavioral_to_realization(0, pol)
        # solve the dual LP directly: max v_0 s.t. F^T v <= A^T x
        h = highspy.Highs(); h.setOptionValue("output_flag", False)
        c = S.A.T @ x
        m = S.F.shape[0]; n = S.F.shape[1]
        lp = highspy.HighsLp(); lp.num_col_ = m; lp.num_row_ = n
        cost = np.zeros(m); cost[0] = -1.0; lp.col_cost_ = cost
        lp.col_lower_ = -highspy.kHighsInf * np.ones(m); lp.col_upper_ = highspy.kHighsInf * np.ones(m)
        lp.row_lower_ = -highspy.kHighsInf * np.ones(n); lp.row_upper_ = c
        FT = S.F.T.tocsc()
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = FT.indptr.astype(np.int32); lp.a_matrix_.index_ = FT.indices.astype(np.int32)
        lp.a_matrix_.value_ = FT.data.astype(np.float64)
        h.passModel(lp); h.run()
        v0 = h.getSolution().col_value[0]
        assert abs(v0 - S.security(x)) < 1e-7


# ---------------------------------------------------------------- 7. safe LP
def test_safe_lp_properties():
    bp0 = L.nash_blueprint(0)
    x_nash = S.behavioral_to_realization(0, bp0)
    for k in range(8):
        alpha = [0.3, 1.0, 3.0, 10.0][k % 4]
        q = random_policy(1, alpha=alpha)
        y = S.behavioral_to_realization(1, q)
        g = S.g_of_y(y)
        br_val = S.learner_br_value(y)
        nash_val = x_nash @ g
        prev = -np.inf
        for eps in [0.0, 0.05, 0.1, 0.2, 1.0, 5.0, 50.0]:
            ok, pol, x = L.solve_safe(g, eps)
            assert ok
            e_fast = L.exploitability(x)
            e_os = AUD.exploitability_of_learner(pol)
            assert e_fast <= eps + 1e-7, (eps, e_fast)
            assert e_os <= eps + 1e-7, (eps, e_os)
            assert abs(e_fast - e_os) < 1e-9
            val = x @ g
            assert val >= nash_val - 1e-7             # at least Nash value (Nash is feasible)
            assert val <= br_val + 1e-7               # never above unrestricted BR
            assert val >= prev - 1e-7                 # monotone in eps
            prev = val
            if eps == 0.0:
                assert e_fast < 1e-7                  # minimax safe
        # large eps: constraint not binding -> reaches unrestricted BR value
        assert abs(val - br_val) < 1e-7
        # exploitable value exists for a non-Nash opponent -> strictly above Nash at eps=0.2
        ok, pol, x = L.solve_safe(g, 0.2)
        if br_val - nash_val > 0.05:
            assert x @ g > nash_val + 1e-6
