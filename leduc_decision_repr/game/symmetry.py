"""Suit symmetry: rank-level information sets and suit-permutation symmetrization.

OpenSpiel's Leduc information sets distinguish physical cards (two copies per rank).  Suit
identity is strategically irrelevant, so opponent policies are defined on *rank infosets*
(private rank, public rank or none, betting history) and tied across their physical copies.
The suit-permutation group (swap the two copies of any rank: 8 elements) acts on realization
plans; averaging an equilibrium plan over the group yields a suit-symmetric equilibrium
(the equilibrium set of a zero-sum game is convex and invariant under the group).
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

import numpy as np

from .leduc_tree import LeducTree, get_tree, NUM_ACTIONS, card_rank, PLAYER


class SuitSymmetry:
    def __init__(self, tree: LeducTree | None = None):
        self.tree = tree or get_tree()
        T = self.tree
        self.phys_key: List[Dict[int, Tuple]] = [{}, {}]      # infoset -> (priv card, pub card, history)
        self.key_to_infoset: List[Dict[Tuple, int]] = [{}, {}]
        self.rank_key_to_index: List[Dict[Tuple, int]] = [{}, {}]
        self.rank_keys: List[List[Tuple]] = [[], []]
        self.rank_infoset_of: List[np.ndarray] = []            # physical infoset -> rank infoset
        self.rank_members: List[List[List[int]]] = []          # rank infoset -> physical infosets
        for p in range(2):
            tab = T.infosets[p]
            rank_of = np.zeros(len(tab), dtype=np.int32)
            members: List[List[int]] = []
            for I in range(len(tab)):
                node = tab.example_node[I]
                priv = int(T.card[node, p]); pub = int(T.card[node, 2])
                acts = []
                n = 0
                for a in T.path_actions[node]:
                    if T.node_type[n] == PLAYER:
                        acts.append(int(a))
                    n = T.child_table[n, a]
                key = (priv, pub, tuple(acts))
                self.phys_key[p][I] = key
                self.key_to_infoset[p][key] = I
                rkey = (card_rank(priv), card_rank(pub) if pub >= 0 else -1, tuple(acts))
                if rkey not in self.rank_key_to_index[p]:
                    self.rank_key_to_index[p][rkey] = len(members)
                    self.rank_keys[p].append(rkey)
                    members.append([])
                r = self.rank_key_to_index[p][rkey]
                rank_of[I] = r
                members[r].append(I)
            self.rank_infoset_of.append(rank_of)
            self.rank_members.append(members)
        self.n_rank_infosets = [len(self.rank_members[p]) for p in range(2)]
        self.rank_legal_mask = []
        for p in range(2):
            mask = np.stack([T.infosets[p].legal_mask[m[0]] for m in self.rank_members[p]])
            for r, m in enumerate(self.rank_members[p]):
                for I in m:
                    assert np.array_equal(T.infosets[p].legal_mask[I], mask[r])
            self.rank_legal_mask.append(mask)
        self.perms = []
        for bits in itertools.product([0, 1], repeat=3):
            perm = list(range(6))
            for rnk, b in enumerate(bits):
                if b:
                    perm[2 * rnk], perm[2 * rnk + 1] = 2 * rnk + 1, 2 * rnk
            self.perms.append(perm)
        self.seq_perm = [[], []]
        for p in range(2):
            for perm in self.perms:
                sp = np.zeros(T.n_seq[p], dtype=np.int32)
                for s in range(1, T.n_seq[p]):
                    I = T.seq_infoset[p][s]; a = T.seq_action[p][s]
                    priv, pub, acts = self.phys_key[p][I]
                    img = (perm[priv], perm[pub] if pub >= 0 else -1, acts)
                    J = self.key_to_infoset[p][img]
                    sp[s] = T.seq_index[p][(J, int(a))]
                self.seq_perm[p].append(sp)

    def expand(self, p: int, rank_policy: np.ndarray) -> np.ndarray:
        """(n_rank, 3) -> (n_phys, 3) tied policy."""
        return rank_policy[self.rank_infoset_of[p]]

    def reduce(self, p: int, phys_policy: np.ndarray, check_tied: bool = False) -> np.ndarray:
        out = np.stack([phys_policy[m[0]] for m in self.rank_members[p]])
        if check_tied:
            assert np.allclose(self.expand(p, out), phys_policy)
        return out

    def symmetrize_realization(self, p: int, x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x)
        for sp in self.seq_perm[p]:
            out[sp] += x
        return out / len(self.perms)

    def symmetrize_policy(self, sf, p: int, policy_phys: np.ndarray) -> np.ndarray:
        """Group-average in realization space, then return the tied behavioral policy."""
        x = sf.behavioral_to_realization(p, policy_phys)
        xs = self.symmetrize_realization(p, x)
        pol = sf.realization_to_behavioral(p, xs)
        return self.expand(p, self.reduce(p, pol))


_SYM = None


def get_symmetry() -> SuitSymmetry:
    global _SYM
    if _SYM is None:
        _SYM = SuitSymmetry(get_tree())
    return _SYM
