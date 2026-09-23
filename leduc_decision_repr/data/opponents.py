"""Opponent population generator: four families with shared statistical structure.

All opponents are behavioral policies of player 1 (the opponent) defined on the 144 rank-level
infosets (private rank, public rank, betting history) and expanded to the 468 physical OpenSpiel
infosets by tying suit copies.  Generator parameters and per-opponent seeds are recorded.

Families
--------
NASH_LOGIT_PERTURB   logits = log(floor(nash)) + tau * (W^T phi(I) + eps_I); coherent
                     feature-driven perturbation plus per-infoset noise, strength tau.
NASH_RANDOM_MIX      q = (1 - eta) * nash + eta * Dirichlet(1) per infoset.
STRUCTURED_CORRELATED logits = trait-driven + random feature coefficients on shared infoset
                     features (rank, public rank, pairing, round, facing aggression, to-call,
                     raise count, pot/contributions, legal actions) + small per-infoset noise.
UNSTRUCTURED_DIRICHLET independent Dirichlet(alpha) draw per infoset (control family).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from ..game.leduc_tree import LeducTree, get_tree, NUM_ACTIONS, FOLD, CALL, RAISE
from ..game.policy_utils import rank_infoset_features, INFOSET_FEATURE_NAMES
from ..game.symmetry import SuitSymmetry

FAMILIES = ["NASH_LOGIT_PERTURB", "NASH_RANDOM_MIX", "STRUCTURED_CORRELATED", "UNSTRUCTURED_DIRICHLET"]


def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    z = np.where(mask, logits, -np.inf)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z) * mask
    return p / p.sum(1, keepdims=True)


def floor_policy(pol: np.ndarray, mask: np.ndarray, floor: float) -> np.ndarray:
    p = np.where(mask, np.maximum(pol, floor), 0.0)
    return p / p.sum(1, keepdims=True)


class OpponentGenerator:
    def __init__(self, tree: LeducTree, sym: SuitSymmetry, nash_opp_rank: np.ndarray, base_seed: int = 2024):
        """nash_opp_rank: (n_rank, 3) suit-symmetric Nash policy of the opponent at rank level."""
        self.tree = tree
        self.sym = sym
        self.mask = sym.rank_legal_mask[1]
        self.phi = rank_infoset_features(tree, sym, 1)      # (m, F), m = 144 rank infosets
        self.nash = nash_opp_rank
        self.base_seed = base_seed
        self.m, self.F = self.phi.shape
        self.phi_norm = float(np.sqrt(np.mean(np.sum(self.phi ** 2, axis=1))))
        self.feat_idx = {n: i for i, n in enumerate(INFOSET_FEATURE_NAMES)}

    def _rng(self, family_idx: int, k: int):
        return np.random.default_rng([self.base_seed, family_idx, k])

    # ------------------------------------------------------------------ families
    def nash_logit_perturb(self, k: int):
        rng = self._rng(0, k)
        # tau is the logit-scale strength of the perturbation (log-uniform so that near-Nash
        # opponents are well represented); the coherent feature term is normalized so that
        # W^T phi(I) has unit variance on average over infosets.
        tau = float(np.exp(rng.uniform(np.log(0.15), np.log(2.0))))
        W = rng.normal(0, 1.0, size=(self.F, NUM_ACTIONS))
        noise_scale = float(rng.uniform(0.2, 1.0))
        eps = rng.normal(0, noise_scale, size=(self.m, NUM_ACTIONS))
        base = np.log(floor_policy(self.nash, self.mask, 0.02) + 1e-300)
        logits = base + tau * (self.phi @ W / self.phi_norm + eps)
        q = masked_softmax(logits, self.mask)
        return q, {"tau": tau, "noise_scale": noise_scale, "W": W.tolist()}

    def nash_random_mix(self, k: int):
        rng = self._rng(1, k)
        eta = float(rng.uniform(0.05, 0.85))
        rand = masked_softmax(np.log(rng.dirichlet(np.ones(NUM_ACTIONS), size=self.m) + 1e-300), self.mask)
        q = (1 - eta) * self.nash + eta * rand
        q = q * self.mask
        q = q / q.sum(1, keepdims=True)
        return q, {"eta": eta}

    def structured_correlated(self, k: int):
        rng = self._rng(2, k)
        fi = self.feat_idx
        traits = rng.normal(0, 1.0, size=8)
        aggression, passivity, strength_sens, bluff, overfold, overcall, r2_shift, tight = traits
        W = rng.normal(0, 0.6, size=(self.F, NUM_ACTIONS))          # random feature coefficients
        b = np.zeros(NUM_ACTIONS)
        # trait-driven coherent structure
        b[RAISE] += 1.2 * aggression - 0.8 * passivity
        b[CALL] += 0.8 * passivity
        b[FOLD] += 1.0 * overfold - 0.8 * overcall + 0.6 * tight
        W[fi["hand_strength"], RAISE] += 1.5 * strength_sens + 0.5
        W[fi["hand_strength"], FOLD] -= 1.5 * strength_sens + 0.5
        W[fi["rank_J"], RAISE] += 1.2 * bluff
        W[fi["rank_J"], FOLD] -= 0.8 * bluff
        W[fi["facing_raise"], FOLD] += 1.0 * overfold + 0.4 * tight
        W[fi["facing_raise"], CALL] += 0.8 * overcall
        W[fi["round2"], RAISE] += 0.8 * r2_shift
        W[fi["round2"], FOLD] -= 0.5 * r2_shift
        W[fi["paired"], RAISE] += 1.0 + 0.5 * strength_sens
        W[fi["raises_this_round_2"], RAISE] -= 0.5 * tight
        eps = rng.normal(0, 0.3, size=(self.m, NUM_ACTIONS))
        logits = b[None, :] + self.phi @ W + eps
        q = masked_softmax(logits, self.mask)
        return q, {"traits": traits.tolist(), "W": W.tolist(), "b": b.tolist()}

    def unstructured_dirichlet(self, k: int):
        rng = self._rng(3, k)
        alpha = float(np.exp(rng.uniform(np.log(0.3), np.log(3.0))))
        q = np.zeros((self.m, NUM_ACTIONS))
        for I in range(self.m):
            legal = np.flatnonzero(self.mask[I])
            q[I, legal] = rng.dirichlet(alpha * np.ones(len(legal)))
        return q, {"alpha": alpha}

    def generate(self, family: str, k: int):
        fn = {
            "NASH_LOGIT_PERTURB": self.nash_logit_perturb,
            "NASH_RANDOM_MIX": self.nash_random_mix,
            "STRUCTURED_CORRELATED": self.structured_correlated,
            "UNSTRUCTURED_DIRICHLET": self.unstructured_dirichlet,
        }[family]
        q, params = fn(k)
        assert np.allclose(q.sum(1), 1.0) and np.all(q[~self.mask] == 0)
        return q, params


def build_population(tree, sym, nash_opp_rank, counts_per_family=(413, 413, 412, 412),
                     split_per_family=((300, 38, 75), (300, 38, 75), (300, 37, 75), (300, 37, 75)),
                     base_seed=2024):
    """Generate the population and a stratified train/val/test split.

    Returns dict with rank_policies (K, 144, 3), policies (K, 468, 3) (physical, tied), family (K,),
    family_index, split (K,) in {0: train, 1: val, 2: test}, params (list of dicts), within index.
    """
    gen = OpponentGenerator(tree, sym, nash_opp_rank, base_seed)
    policies, fam, fam_idx, split, params, within = [], [], [], [], [], []
    for fi, family in enumerate(FAMILIES):
        n_tr, n_va, n_te = split_per_family[fi]
        assert n_tr + n_va + n_te == counts_per_family[fi]
        rng = np.random.default_rng([base_seed, 999, fi])
        perm = rng.permutation(counts_per_family[fi])
        labels = np.zeros(counts_per_family[fi], dtype=np.int8)
        labels[perm[n_tr:n_tr + n_va]] = 1
        labels[perm[n_tr + n_va:]] = 2
        for k in range(counts_per_family[fi]):
            q, prm = gen.generate(family, k)
            policies.append(q); fam.append(family); fam_idx.append(fi)
            split.append(labels[k]); params.append(prm); within.append(k)
    rank_policies = np.array(policies)
    return {
        "rank_policies": rank_policies, "policies": np.stack([sym.expand(1, q) for q in rank_policies]),
        "family": np.array(fam), "family_index": np.array(fam_idx),
        "split": np.array(split), "params": params, "within_index": np.array(within),
        "base_seed": base_seed, "families": FAMILIES,
    }
