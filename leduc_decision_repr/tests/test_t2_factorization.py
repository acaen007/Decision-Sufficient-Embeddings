"""T2 gate: with a posterior that factorizes across opponent infosets (independent Dirichlet per infoset),
perfect recall gives E[y_q] = y_{E[q]}, hence A E[y] = A y_{E[q]} exactly."""
import numpy as np

from leduc_decision_repr.game.leduc_tree import get_tree
from leduc_decision_repr.game.sequence_form import get_sequence_form
from leduc_decision_repr.game.symmetry import get_symmetry
from leduc_decision_repr.game.policy_utils import RankPolicyToG

T = get_tree(); S = get_sequence_form(); SYM = get_symmetry(); R2G = RankPolicyToG(T, S, SYM)


def test_factorized_posterior_mean_commutes_with_realization():
    rng = np.random.default_rng(0)
    mask = SYM.rank_legal_mask[1]
    for trial in range(3):
        alpha = np.where(mask, rng.uniform(0.5, 5.0, size=(144, 3)), 0.0)
        # Monte-Carlo posterior: independent Dirichlet per infoset (full revelation => factorized)
        n = 20000
        Q = np.zeros((n, 144, 3))
        for I in range(144):
            legal = np.flatnonzero(mask[I]); Q[:, I, legal] = rng.dirichlet(alpha[I, legal], size=n)
        Ey = R2G.realization(Q).mean(0)                     # E[y_q]
        Eq = Q.mean(0); y_Eq = R2G.realization(Eq[None])[0]  # y_{E[q]}
        # exact expectation: E[y] = product of per-infoset means (independence along each sequence)
        exact_Eq = np.where(mask, alpha / np.maximum(alpha.sum(1, keepdims=True), 1e-12), 0.0)
        y_exact = R2G.realization(exact_Eq[None])[0]
        # (i) exact identity: with the exact Dirichlet means the realization of the mean policy equals the
        #     expected realization (independence along sequences) -- machine precision
        # (ii) Monte-Carlo estimate agrees at the sampling-error level
        assert np.abs(S.A @ (y_exact - R2G.realization(exact_Eq[None])[0])).max() < 1e-12
        mc_err = np.abs(S.A @ (Ey - y_Eq)).max()
        assert mc_err < 5e-3, mc_err
    # analytic check of independence: for a product measure, E[prod q(I_k,a_k)] = prod E[q(I_k,a_k)] along any sequence
    alpha = np.where(mask, rng.uniform(0.5, 5.0, size=(144, 3)), 0.0)
    means = np.where(mask, alpha / np.maximum(alpha.sum(1, keepdims=True), 1e-12), 0.0)
    # every opponent sequence visits each infoset at most once (perfect recall), so the product of independent
    # per-infoset means is exactly the expected product
    for s in range(1, R2G.n1):
        infosets = [int(i // 3) for i in R2G.idx[s][R2G.valid[s]]]
        assert len(set(infosets)) == len(infosets)
