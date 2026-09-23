"""Table-driven, vectorized Leduc hand simulator.

A hand is fully described by the terminal node it reaches in the enumerated tree.  For
one opponent all hands of all streams are simulated simultaneously; the uniforms used by
stream s are drawn from an RNG seeded by (base_seed, split, opponent_id, s) only, so each
stream is a deterministic function of its seed and prefixes are automatically consistent.
"""
from __future__ import annotations

import numpy as np

from ..game.leduc_tree import LeducTree, CHANCE, PLAYER, TERMINAL, NUM_CARDS

MAX_STEPS = 12   # 3 chance + at most 8 player actions; loop exits early when all terminal


def node_probability_table(tree: LeducTree, policy0: np.ndarray, policy1: np.ndarray) -> np.ndarray:
    """(n_nodes, 6) transition probabilities over child slots for the given policies."""
    n = tree.n_nodes
    P = np.zeros((n, NUM_CARDS))
    for i in range(n):
        if tree.node_type[i] == CHANCE:
            valid = tree.child_table[i] >= 0
            P[i, valid] = 1.0 / valid.sum()
        elif tree.node_type[i] == PLAYER:
            pol = policy0 if tree.player[i] == 0 else policy1
            P[i, :3] = pol[tree.infoset_of_node[i]]
    return P


def stream_seed(base_seed: int, split_id: int, opponent_id: int, stream_id: int):
    return [int(base_seed), int(split_id), int(opponent_id), int(stream_id)]


def simulate_streams(tree: LeducTree, policy0: np.ndarray, policy1: np.ndarray, n_streams: int,
                     n_hands: int, base_seed: int, split_id: int, opponent_id: int) -> np.ndarray:
    """Return (n_streams, n_hands) terminal node indices."""
    P = node_probability_table(tree, policy0, policy1)
    cum = np.cumsum(P, axis=1)
    U = np.stack([np.random.default_rng(stream_seed(base_seed, split_id, opponent_id, s))
                  .random((n_hands, MAX_STEPS)) for s in range(n_streams)])       # (S, H, steps)
    U = U.reshape(-1, MAX_STEPS)
    nodes = np.zeros(U.shape[0], dtype=np.int64)
    for step in range(MAX_STEPS):
        active = tree.node_type[nodes] != TERMINAL
        if not active.any():
            break
        idx = np.flatnonzero(active)
        u = U[idx, step]
        c = cum[nodes[idx]]
        a = (u[:, None] >= c).sum(1)
        a = np.minimum(a, NUM_CARDS - 1)
        # guard against cumulative rounding: make sure the chosen slot is a valid child
        child = tree.child_table[nodes[idx], a]
        bad = child < 0
        if bad.any():
            # fall back to the last valid slot with positive probability
            for j in np.flatnonzero(bad):
                valid = np.flatnonzero(P[nodes[idx[j]]] > 0)
                child[j] = tree.child_table[nodes[idx[j]], valid[-1]]
        nodes[idx] = child
    assert np.all(tree.node_type[nodes] == TERMINAL)
    return nodes.reshape(n_streams, n_hands)


def simulate_streams_reference(tree: LeducTree, policy0, policy1, n_streams, n_hands, base_seed,
                               split_id, opponent_id) -> np.ndarray:
    """Slow scalar reference implementation with identical random-number consumption."""
    P = node_probability_table(tree, policy0, policy1)
    cum = np.cumsum(P, axis=1)
    out = np.zeros((n_streams, n_hands), dtype=np.int64)
    for s in range(n_streams):
        U = np.random.default_rng(stream_seed(base_seed, split_id, opponent_id, s)).random((n_hands, MAX_STEPS))
        for h in range(n_hands):
            node = 0; step = 0
            while tree.node_type[node] != TERMINAL:
                a = min(int((U[h, step] >= cum[node]).sum()), NUM_CARDS - 1)
                node = int(tree.child_table[node, a]); step += 1
            out[s, h] = node
    return out
