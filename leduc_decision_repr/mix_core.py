"""Mixture-prior Bayesian opponent modelling (REPORT_LEDUC_MIXPRIOR.md): q ~ sum_k pi_k Dir(kappa * a_k).
Per history: score every anchor by its exact hand likelihood, keep the top K, run tabular EM with a Dirichlet prior centred
on each kept anchor, weight the components by an approximate marginal likelihood, and deploy on the belief average of g."""
import numpy as np
from scipy.special import gammaln


def em_fit(lik, mask, A, kappa, counts, n_iter=200, tol=1e-7):
    """Rows: anchor policies A (n, 144, 3) (floored at 1e-6, renormalized), concentration kappa, type counts (n, T).
    M-step q = (expected counts + kappa * a) / sum: the same update as fit_em_with_row_prior.  Returns the fitted Q, the
    expected completed-data counts at Q and the observed-data log-likelihood log p(H | Q) (up to q-independent factors)."""
    n = counts.shape[0]
    prior = np.where(mask[None], np.maximum(A, 1e-6), 0.0); prior = prior / prior.sum(2, keepdims=True)
    Q = prior.copy(); c_pair = counts[:, lik.pair_type]; ap = kappa * prior
    def estep(Q):                                                     # pair log-likelihoods are bounded (q >= ~1e-8), so plain exp is safe
        e = np.exp(lik.pair_loglik(np.log(np.maximum(Q, 1e-300)).reshape(n, -1)) + lik.pair_logprior[None])
        z = (lik.S @ e.T).T; w = e / np.maximum(z[:, lik.pair_type], 1e-300)
        ec = (lik.M.T @ (w * c_pair).T).T.reshape(n, -1, 3)
        obs_ll = np.where(counts > 0, counts * np.log(np.maximum(z, 1e-300)), 0.0).sum(1)
        return ec, obs_ll
    for it in range(n_iter):
        ec, _ = estep(Q)
        num = ec + ap; Qn = np.where(mask[None], num / num.sum(2, keepdims=True), 0.0)
        delta = np.abs(Qn - Q).max(); Q = Qn
        if delta < tol:
            break
    ec, obs_ll = estep(Q)
    return Q, ec, obs_ll, prior, it + 1


def log_B(alpha, mask):
    """Per row: sum over infosets of log B(alpha_I) over legal actions (single-action infosets contribute 0)."""
    a = np.where(mask[None], alpha, 1.0)
    return (np.where(mask[None], gammaln(a), 0.0).sum(2) - gammaln(np.where(mask[None], alpha, 0.0).sum(2))).sum(1)


def log_weight(kind, Q, ec, obs_ll, prior, kappa, mask):
    """Unnormalized log component weight (uniform pi_k cancels).
    'map':      EM objective log p(H | q_k) + log Dir(q_k | kappa a_k + 1)  (pseudo-count parameterization, the density the
                M-step maximizes).
    'evidence': expected-count Dirichlet-multinomial evidence, log p(H | q_k) - sum n_hat log q_k + sum_I log B(kappa a_I + n_hat_I)
                - log B(kappa a_I): the completed-data term of the EM bound replaced by its exact Dirichlet-multinomial marginal at
                the expected counts.  Tends to log p(H | a_k) as kappa -> infinity and is exact when no card is hidden."""
    lq = np.log(np.maximum(Q, 1e-300))
    if kind == "map":
        al = kappa * prior + 1.0
        dens = -log_B(np.where(mask[None], al, 1.0), mask) + np.where(mask[None], (al - 1.0) * lq, 0.0).sum((1, 2))
        return obs_ll + dens
    al = np.where(mask[None], kappa * prior, 1.0)
    return obs_ll - np.where(mask[None], ec * lq, 0.0).sum((1, 2)) + log_B(al + np.where(mask[None], ec, 0.0), mask) - log_B(al, mask)


def mixture_ghat(lik, mask, r2g, anchors_ll, anchors, kappa, counts, K=16, kind="evidence"):
    """counts (n, T).  anchors_ll (n_anchor, T) type log-likelihoods of the anchors.  Returns g_hat (n, n0), weights (n, K),
    component indices (n, K), bank-posterior mass of the kept anchors (n,) and EM iterations."""
    n = counts.shape[0]; ll = counts @ anchors_ll.T
    top = np.argsort(-ll, axis=1)[:, :K]
    post = np.exp(ll - ll.max(1, keepdims=True)); post /= post.sum(1, keepdims=True); kept = np.take_along_axis(post, top, 1).sum(1)
    rows_c = np.repeat(counts, K, 0); rows_a = anchors[top.ravel()]
    Q, ec, obs_ll, prior, iters = em_fit(lik, mask, rows_a, kappa, rows_c)
    lw = log_weight(kind, Q, ec, obs_ll, prior, kappa, mask).reshape(n, K)
    w = np.exp(lw - lw.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
    G = r2g.g(Q).reshape(n, K, -1)
    return (w[:, :, None] * G).sum(1), w, top, kept, iters


# ------------------------------------------------------------------ parallel mixture fits (spawn pool, 1 thread per worker)
_W = {}


def anchor_sets():
    """{'MIX-BANK': true training policies (1200,144,3), 'MIX-LATENT': JAC-opp seed-0 q_hat of each training opponent (if built)}."""
    from .common import OUT
    from .data.datasets import load_population
    pop = load_population(); tr = np.flatnonzero(pop["split"] == 0); out = {"MIX-BANK": pop["rank_policies"][tr]}
    p = OUT / "mixprior" / "anchors_latent.npy"
    if p.exists():
        out["MIX-LATENT"] = np.load(p)
    return out


def _init():
    from .game.leduc_tree import get_tree
    from .game.symmetry import get_symmetry
    from .game.sequence_form import get_sequence_form
    from .game.policy_utils import RankPolicyToG
    from .data.tokenizer import get_token_table
    from .baselines.likelihood import HandLikelihood
    sym = get_symmetry(); lik = HandLikelihood(get_token_table(), sym); anchors = anchor_sets()
    _W.update(lik=lik, mask=np.asarray(sym.rank_legal_mask[1], bool), r2g=RankPolicyToG(get_tree(), get_sequence_form(), sym), anchors=anchors,
              ll={k: lik.type_loglik(np.log(np.maximum(A, 1e-300)).reshape(len(A), -1)) for k, A in anchors.items()})


def _task(args):
    arm, kappa, K, kind, counts = args
    g, w, top, kept, it = mixture_ghat(_W["lik"], _W["mask"], _W["r2g"], _W["ll"][arm], _W["anchors"][arm], kappa, counts, K=K, kind=kind)
    good = np.isfinite(w).all(1) & (np.abs(w.sum(1) - 1) < 1e-9)
    return g, 1.0 / (w ** 2).sum(1), kept, w.max(1), it, good


class EMPool:
    def __init__(self, workers=4):
        import multiprocessing as mp
        self.pool = mp.get_context("spawn").Pool(workers, initializer=_init)

    def ghat(self, arm, kappa, K, counts, kind="evidence", rows=3200):
        """Returns g_hat (n, n0), effective components 1/sum w^2 (n,), kept bank-likelihood mass (n,), max weight (n,),
        max EM iterations over chunks, and a per-history S-c flag (weights finite and summing to 1)."""
        step = max(1, rows // K); parts = self.pool.map(_task, [(arm, kappa, K, kind, counts[s:s + step]) for s in range(0, len(counts), step)])
        cat = lambda i: np.concatenate([p[i] for p in parts])
        return cat(0), cat(1), cat(2), cat(3), max(p[4] for p in parts), cat(5)

    def close(self):
        self.pool.close(); self.pool.join()
