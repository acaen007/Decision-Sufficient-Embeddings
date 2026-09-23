"""Enumeration of the OpenSpiel two-player Leduc poker game tree into flat arrays.

Everything downstream (sequence form, simulator, tokenizer, EM baseline) is table-driven
from this enumeration.  OpenSpiel is the source of truth for legality, chance
probabilities, information-set identity and terminal utilities; the public betting state
(pot, contributions, amount to call, raise counts) is re-derived from the action history
using Leduc's rules and cross-checked against OpenSpiel's state strings (see tests).

Conventions
-----------
* players: 0 = learner (acts first in each round), 1 = opponent.
* cards: OpenSpiel card ids 0..5; rank = card // 2 (0=J, 1=Q, 2=K).
* actions: 0 = FOLD, 1 = CALL (= check when nothing to call), 2 = RAISE.
* node types: CHANCE = 0, PLAYER = 1, TERMINAL = 2.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pyspiel

FOLD, CALL, RAISE = 0, 1, 2
CHANCE, PLAYER, TERMINAL = 0, 1, 2
NUM_ACTIONS = 3
NUM_CARDS = 6
NUM_RANKS = 3
ANTE = 1
RAISE_SIZE = {1: 2, 2: 4}
MAX_RAISES = 2
START_MONEY = 100
MAX_CONTRIB = ANTE + MAX_RAISES * RAISE_SIZE[1] + MAX_RAISES * RAISE_SIZE[2]  # 13
MAX_POT = 2 * MAX_CONTRIB  # 26

# terminal types
TERM_NONE, TERM_FOLD_SELF, TERM_FOLD_OPP, TERM_SHOWDOWN = 0, 1, 2, 3


def card_rank(card: int) -> int:
    return card // 2


@dataclass
class InfosetTable:
    """Per-player information-set table."""
    keys: List[str] = field(default_factory=list)
    index: Dict[str, int] = field(default_factory=dict)
    legal_mask: List[np.ndarray] = field(default_factory=list)   # (3,) bool
    parent_seq: List[int] = field(default_factory=list)          # sequence id leading to infoset
    depth: List[int] = field(default_factory=list)               # number of own actions before
    # structured public/private features visible to the acting player
    private_rank: List[int] = field(default_factory=list)
    round: List[int] = field(default_factory=list)               # 1 or 2
    public_rank: List[int] = field(default_factory=list)         # -1 if none
    pot: List[int] = field(default_factory=list)
    own_contrib: List[int] = field(default_factory=list)
    opp_contrib: List[int] = field(default_factory=list)
    to_call: List[int] = field(default_factory=list)
    raises_this_round: List[int] = field(default_factory=list)
    facing_raise: List[int] = field(default_factory=list)
    n_actions_this_round: List[int] = field(default_factory=list)
    n_raises_prev_round: List[int] = field(default_factory=list)
    example_node: List[int] = field(default_factory=list)

    def __len__(self):
        return len(self.keys)


class LeducTree:
    """Flat-array enumeration of the Leduc game tree."""

    def __init__(self, game_string: str = "leduc_poker"):
        self.game_string = game_string
        self.game = pyspiel.load_game(game_string)
        assert self.game.num_players() == 2
        assert self.game.num_distinct_actions() == NUM_ACTIONS
        self._enumerate()

    # ------------------------------------------------------------------ enumeration
    def _enumerate(self):
        g = self.game
        # per node lists
        self.parent: List[int] = []
        self.action_from_parent: List[int] = []
        self.node_type: List[int] = []
        self.player: List[int] = []          # acting player for PLAYER nodes else -1
        self.chance_prob: List[float] = []   # prob of action_from_parent if parent is chance
        self.reach_chance: List[float] = []  # product of chance probs on the path
        self.children: List[Dict[int, int]] = []
        self.legal: List[List[int]] = []
        self.depth: List[int] = []
        self.returns0: List[float] = []      # utility of player 0 at terminal else 0
        self.terminal_type: List[int] = []   # TERM_* from player 0's perspective
        self.infoset_of_node: List[int] = [] # infoset index for acting player else -1
        self.seq_before: List[Tuple[int, int]] = []   # last sequence of (p0, p1) before node
        # private/public card info (internal only)
        self.card: List[Tuple[int, int, int]] = []    # (card0, card1, public) with -1 for undealt
        # public betting state before the node's action
        self.round: List[int] = []
        self.pot: List[int] = []
        self.contrib: List[Tuple[int, int]] = []
        self.raises_this_round: List[int] = []
        self.n_raises_round1: List[int] = []
        self.n_actions_this_round: List[int] = []
        self.actions_round: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
        self.os_pot: List[int] = []                    # OpenSpiel's reported pot (for tests)
        self.os_money: List[Tuple[int, int]] = []      # OpenSpiel's reported money (for tests)
        self.infosets = [InfosetTable(), InfosetTable()]
        # sequence tables: seq 0 = empty; others = (infoset, action)
        self.seq_infoset = [[-1], [-1]]
        self.seq_action = [[-1], [-1]]
        self.seq_index: List[Dict[Tuple[int, int], int]] = [{}, {}]

        root = g.new_initial_state()
        pot_re = re.compile(r"Pot: (\d+)")
        money_re = re.compile(r"Money \(player_0 player_1\): (\d+) (\d+)")

        def add_node(state, parent, a, cprob, reach, seq0, seq1, cards, rnd, contrib,
                     raises_rnd, raises_r1, nact_rnd, acts_r1, acts_r2, depth):
            idx = len(self.parent)
            self.parent.append(parent)
            self.action_from_parent.append(a)
            self.chance_prob.append(cprob)
            self.reach_chance.append(reach)
            self.children.append({})
            self.depth.append(depth)
            self.seq_before.append((seq0, seq1))
            self.card.append(cards)
            self.round.append(rnd)
            self.pot.append(contrib[0] + contrib[1])
            self.contrib.append(contrib)
            self.raises_this_round.append(raises_rnd)
            self.n_raises_round1.append(raises_r1)
            self.n_actions_this_round.append(nact_rnd)
            self.actions_round.append((tuple(acts_r1), tuple(acts_r2)))
            s = str(state)
            m = pot_re.search(s)
            self.os_pot.append(int(m.group(1)) if m else -1)
            m = money_re.search(s)
            self.os_money.append((int(m.group(1)), int(m.group(2))) if m else (-1, -1))
            if state.is_terminal():
                self.node_type.append(TERMINAL)
                self.player.append(-1)
                self.legal.append([])
                r = state.returns()
                self.returns0.append(float(r[0]))
                self.infoset_of_node.append(-1)
                # terminal type: fold if last action was FOLD
                if a == FOLD:
                    folder = self.player[parent]
                    self.terminal_type.append(TERM_FOLD_SELF if folder == 0 else TERM_FOLD_OPP)
                else:
                    self.terminal_type.append(TERM_SHOWDOWN)
                return idx
            self.returns0.append(0.0)
            self.terminal_type.append(TERM_NONE)
            if state.is_chance_node():
                self.node_type.append(CHANCE)
                self.player.append(-1)
                self.legal.append([o for o, _ in state.chance_outcomes()])
                self.infoset_of_node.append(-1)
                return idx
            p = state.current_player()
            self.node_type.append(PLAYER)
            self.player.append(p)
            la = list(state.legal_actions())
            self.legal.append(la)
            key = state.information_state_string(p)
            tab = self.infosets[p]
            parent_seq = seq0 if p == 0 else seq1
            own_depth = len(acts_r1) + len(acts_r2)
            if key not in tab.index:
                I = len(tab.keys)
                tab.index[key] = I
                tab.keys.append(key)
                mask = np.zeros(NUM_ACTIONS, dtype=bool)
                mask[la] = True
                tab.legal_mask.append(mask)
                tab.parent_seq.append(parent_seq)
                tab.depth.append(sum(1 for _ in self._own_actions(acts_r1, acts_r2, p)))
                tab.private_rank.append(card_rank(cards[p]))
                tab.round.append(rnd)
                tab.public_rank.append(card_rank(cards[2]) if cards[2] >= 0 else -1)
                tab.pot.append(contrib[0] + contrib[1])
                tab.own_contrib.append(contrib[p])
                tab.opp_contrib.append(contrib[1 - p])
                tab.to_call.append(contrib[1 - p] - contrib[p])
                tab.raises_this_round.append(raises_rnd)
                tab.facing_raise.append(int(contrib[1 - p] > contrib[p]))
                tab.n_actions_this_round.append(nact_rnd)
                tab.n_raises_prev_round.append(raises_r1 if rnd == 2 else 0)
                tab.example_node.append(idx)
                for act in la:
                    sid = len(self.seq_infoset[p])
                    self.seq_infoset[p].append(I)
                    self.seq_action[p].append(act)
                    self.seq_index[p][(I, act)] = sid
            else:
                I = tab.index[key]
                # perfect recall + consistency checks
                assert tab.parent_seq[I] == parent_seq, "perfect recall violated"
                assert list(np.flatnonzero(tab.legal_mask[I])) == la
            self.infoset_of_node.append(I)
            return idx

        def rec(state, parent, a, cprob, reach, seq0, seq1, cards, rnd, contrib,
                raises_rnd, raises_r1, nact_rnd, acts_r1, acts_r2, depth):
            idx = add_node(state, parent, a, cprob, reach, seq0, seq1, cards, rnd, contrib,
                           raises_rnd, raises_r1, nact_rnd, acts_r1, acts_r2, depth)
            if state.is_terminal():
                return idx
            if state.is_chance_node():
                for o, pr in state.chance_outcomes():
                    child = state.child(o)
                    ncards = list(cards)
                    if cards[0] < 0:
                        ncards[0] = o
                    elif cards[1] < 0:
                        ncards[1] = o
                    else:
                        ncards[2] = o
                    # dealing the public card starts round 2
                    nrnd = 2 if ncards[2] >= 0 else 1
                    cidx = rec(child, idx, o, pr, reach * pr, seq0, seq1, tuple(ncards), nrnd,
                               contrib, 0 if nrnd == 2 and rnd == 1 else raises_rnd,
                               raises_r1 if nrnd == 2 else raises_r1,
                               0 if nrnd == 2 and rnd == 1 else nact_rnd, acts_r1, acts_r2, depth + 1)
                    self.children[idx][o] = cidx
                return idx
            p = state.current_player()
            I = self.infoset_of_node[idx]
            for act in state.legal_actions():
                child = state.child(act)
                sid = self.seq_index[p][(I, act)]
                nseq0, nseq1 = (sid, seq1) if p == 0 else (seq0, sid)
                ncontrib = list(contrib)
                to_call = contrib[1 - p] - contrib[p]
                nraises = raises_rnd
                if act == CALL:
                    ncontrib[p] += to_call
                elif act == RAISE:
                    ncontrib[p] += to_call + RAISE_SIZE[rnd]
                    nraises += 1
                nacts_r1 = list(acts_r1)
                nacts_r2 = list(acts_r2)
                (nacts_r1 if rnd == 1 else nacts_r2).append(act)
                nr1 = nraises if rnd == 1 else raises_r1
                cidx = rec(child, idx, act, 1.0, reach, nseq0, nseq1, cards, rnd, tuple(ncontrib),
                           nraises, nr1, nact_rnd + 1, nacts_r1, nacts_r2, depth + 1)
                self.children[idx][act] = cidx
            return idx

        rec(root, -1, -1, 1.0, 1.0, 0, 0, (-1, -1, -1), 1, (ANTE, ANTE), 0, 0, 0, [], [], 0)
        self._finalize()

    @staticmethod
    def _own_actions(acts_r1, acts_r2, p):
        # player 0 acts at even positions of each round, player 1 at odd positions
        for rnd_acts in (acts_r1, acts_r2):
            for i, a in enumerate(rnd_acts):
                if i % 2 == p:
                    yield a

    def _finalize(self):
        n = len(self.parent)
        self.n_nodes = n
        self.parent = np.array(self.parent, dtype=np.int32)
        self.action_from_parent = np.array(self.action_from_parent, dtype=np.int32)
        self.node_type = np.array(self.node_type, dtype=np.int8)
        self.player = np.array(self.player, dtype=np.int8)
        self.chance_prob = np.array(self.chance_prob, dtype=np.float64)
        self.reach_chance = np.array(self.reach_chance, dtype=np.float64)
        self.depth = np.array(self.depth, dtype=np.int32)
        self.returns0 = np.array(self.returns0, dtype=np.float64)
        self.terminal_type = np.array(self.terminal_type, dtype=np.int8)
        self.infoset_of_node = np.array(self.infoset_of_node, dtype=np.int32)
        self.seq_before = np.array(self.seq_before, dtype=np.int32)      # (n, 2)
        self.card = np.array(self.card, dtype=np.int8)                  # (n, 3)
        self.round = np.array(self.round, dtype=np.int8)
        self.pot = np.array(self.pot, dtype=np.int32)
        self.contrib = np.array(self.contrib, dtype=np.int32)          # (n, 2)
        self.raises_this_round = np.array(self.raises_this_round, dtype=np.int8)
        self.n_raises_round1 = np.array(self.n_raises_round1, dtype=np.int8)
        self.n_actions_this_round = np.array(self.n_actions_this_round, dtype=np.int8)
        self.os_pot = np.array(self.os_pot, dtype=np.int32)
        self.os_money = np.array(self.os_money, dtype=np.int32)
        self.terminals = np.flatnonzero(self.node_type == TERMINAL)
        self.n_terminals = len(self.terminals)
        # child table: (n, 6) with -1 for missing (6 = max chance outcomes)
        self.child_table = -np.ones((n, NUM_CARDS), dtype=np.int32)
        for i, ch in enumerate(self.children):
            for a, c in ch.items():
                self.child_table[i, a] = c
        self.n_infosets = [len(self.infosets[0]), len(self.infosets[1])]
        self.n_seq = [len(self.seq_infoset[0]), len(self.seq_infoset[1])]
        for p in range(2):
            self.seq_infoset[p] = np.array(self.seq_infoset[p], dtype=np.int32)
            self.seq_action[p] = np.array(self.seq_action[p], dtype=np.int32)
            tab = self.infosets[p]
            tab.legal_mask = np.array(tab.legal_mask, dtype=bool)      # (m, 3)
            tab.parent_seq = np.array(tab.parent_seq, dtype=np.int32)
            for name in ("depth", "private_rank", "round", "public_rank", "pot", "own_contrib",
                         "opp_contrib", "to_call", "raises_this_round", "facing_raise",
                         "n_actions_this_round", "n_raises_prev_round", "example_node"):
                setattr(tab, name, np.array(getattr(tab, name), dtype=np.int32))
        # infoset processing order: by depth (parents before children)
        self.infoset_order = [np.argsort(self.infosets[p].depth, kind="stable") for p in range(2)]
        # nodes of each infoset
        self.infoset_nodes = [[[] for _ in range(self.n_infosets[p])] for p in range(2)]
        for i in range(n):
            if self.node_type[i] == PLAYER:
                self.infoset_nodes[self.player[i]][self.infoset_of_node[i]].append(i)
        # per-node full action path (for tokenizer / simulator)
        self.path_actions: List[List[int]] = [[] for _ in range(n)]
        for i in range(1, n):
            self.path_actions[i] = self.path_actions[self.parent[i]] + [int(self.action_from_parent[i])]

    # ------------------------------------------------------------------ helpers
    def node_action_delta(self, node: int, action: int) -> int:
        """Contribution added by the acting player when taking `action` at `node`."""
        child = self.child_table[node, action]
        p = self.player[node]
        return int(self.contrib[child, p] - self.contrib[node, p])

    def state_of_node(self, node: int) -> pyspiel.State:
        s = self.game.new_initial_state()
        for a in self.path_actions[node]:
            s.apply_action(a)
        return s

    def infoset_key_to_index(self, p: int, key: str) -> int:
        return self.infosets[p].index[key]


_TREE = None


def get_tree() -> LeducTree:
    global _TREE
    if _TREE is None:
        _TREE = LeducTree()
    return _TREE
