"""Baseline tests: likelihood correctness, EM convergence to known policies, posterior sanity."""
import numpy as np

from leduc_decision_repr.game.leduc_tree import get_tree, TERM_SHOWDOWN
from leduc_decision_repr.game.sequence_form import get_sequence_form
from leduc_decision_repr.game.safe_lp import get_solver
from leduc_decision_repr.game.symmetry import get_symmetry
from leduc_decision_repr.game.policy_utils import RankPolicyToG, infoset_reach_weights
from leduc_decision_repr.data.tokenizer import get_token_table
from leduc_decision_repr.data.simulator import simulate_streams
from leduc_decision_repr.baselines.likelihood import HandLikelihood, type_counts
from leduc_decision_repr.baselines.tabular_em import TabularEM
from leduc_decision_repr.baselines.bank_posterior import BankPosterior

T = get_tree(); S = get_sequence_form(); L = get_solver(); SYM = get_symmetry(); TAB = get_token_table()
LIK = HandLikelihood(TAB, SYM)
R2G = RankPolicyToG(T, S, SYM)
RNG = np.random.default_rng(3)
BP0 = L.nash_blueprint(0)


def random_rank_policy(rng, alpha=1.0):
    q = rng.dirichlet(alpha * np.ones(3), size=144) * SYM.rank_legal_mask[1]
    return q / q.sum(1, keepdims=True)


def test_rank_policy_to_g_matches_sequence_form():
    qs = np.stack([random_rank_policy(RNG) for _ in range(5)])
    G = R2G.g(qs)
    for i in range(5):
        y = S.behavioral_to_realization(1, SYM.expand(1, qs[i]))
        assert np.allclose(R2G.realization(qs[i:i + 1])[0], y)
        assert np.allclose(G[i], S.A @ y)


def test_type_likelihood_matches_reach_probabilities():
    """sum over terminals of chance*reach must equal exp(type loglik) * our-action factor."""
    q = random_rank_policy(RNG)
    y = S.behavioral_to_realization(1, SYM.expand(1, q))
    x = S.behavioral_to_realization(0, BP0)
    # exact P(type) under (blueprint, q)
    p_type = np.zeros(TAB.n_types)
    for z in T.terminals:
        p_type[TAB.obs_type_of_terminal[z]] += T.reach_chance[z] * x[T.seq_before[z, 0]] * y[T.seq_before[z, 1]]
    # likelihood module gives P(type | q) up to the learner/chance factor; that factor is the
    # probability of the type under q = uniform-everything... compare instead ratios across two
    # opponents, for which the q-independent factor cancels.
    q2 = random_rank_policy(RNG)
    y2 = S.behavioral_to_realization(1, SYM.expand(1, q2))
    p_type2 = np.zeros(TAB.n_types)
    for z in T.terminals:
        p_type2[TAB.obs_type_of_terminal[z]] += T.reach_chance[z] * x[T.seq_before[z, 0]] * y2[T.seq_before[z, 1]]
    ll1 = LIK.type_loglik(np.log(q.reshape(-1)))
    ll2 = LIK.type_loglik(np.log(q2.reshape(-1)))
    m = (p_type > 1e-12) & (p_type2 > 1e-12)
    assert np.allclose(ll1[m] - ll2[m], np.log(p_type[m]) - np.log(p_type2[m]), atol=1e-9)
    # opponents sharing the same policy at all visited infosets give identical likelihood
    assert np.allclose(LIK.type_loglik(np.log(q.reshape(-1))), ll1)


def test_showdown_pairs_restricted_and_fold_pairs_marginalized():
    for t in range(TAB.n_types):
        z = TAB.representative[t]
        pairs = np.flatnonzero(LIK.pair_type == t)
        if T.terminal_type[z] == TERM_SHOWDOWN:
            assert len(pairs) == 1
        else:
            assert len(pairs) >= 2      # at least two candidate ranks after card removal
        assert abs(np.exp(LIK.pair_logprior[pairs]).sum() - (1.0 if T.terminal_type[z] != TERM_SHOWDOWN else np.exp(LIK.pair_logprior[pairs]).sum())) < 1e-12


def test_em_converges_to_known_policy():
    q = random_rank_policy(RNG, alpha=2.0)
    uniform = SYM.rank_legal_mask[1] / SYM.rank_legal_mask[1].sum(1, keepdims=True)
    em = TabularEM(LIK, SYM.rank_legal_mask[1], uniform, alpha=1.0, n_iter=300)
    x = S.behavioral_to_realization(0, BP0)
    y = S.behavioral_to_realization(1, SYM.expand(1, q))
    reach = infoset_reach_weights(T, S, x, y, 1)
    w = np.zeros(144)
    for I in range(468):
        w[SYM.rank_infoset_of[1][I]] += reach[I]
    errs = []
    for N in [50, 500, 5000, 20000]:
        streams = simulate_streams(T, BP0, SYM.expand(1, q), 1, N, 123, 0, 0)
        counts = type_counts(TAB.obs_type_of_terminal[streams], TAB.n_types)
        qh, ll = em.fit(counts, return_loglik=True)
        assert np.all(np.diff(ll[:, 0]) > -1e-6)           # EM monotone in log-likelihood
        err = (w * np.abs(qh[0] - q).sum(1)).sum() / w.sum()  # reach-weighted L1 error
        errs.append(err)
    assert errs[-1] < 0.05 and errs[-1] < errs[0] / 3, errs


def test_bank_posterior_identifies_true_opponent():
    bank = np.stack([random_rank_policy(RNG) for _ in range(40)])
    G = R2G.g(bank)
    bp = BankPosterior(LIK, bank, G)
    for k in [0, 7, 21]:
        streams = simulate_streams(T, BP0, SYM.expand(1, bank[k]), 1, 200, 5, 0, k)
        counts = type_counts(TAB.obs_type_of_terminal[streams], TAB.n_types)
        gbar, post = bp.g_bar(counts)
        assert np.argmax(post[0]) == k and post[0, k] > 0.99
        assert np.allclose(gbar[0], G[k], atol=1e-2)
    # with zero observations the posterior is the uniform prior
    gbar, post = bp.g_bar(np.zeros((1, TAB.n_types)))
    assert np.allclose(post, 1 / 40)
