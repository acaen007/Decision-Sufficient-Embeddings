"""Structured event tokenization of a completed Leduc hand from the learner's (player 0) view.

The ONLY function that touches the game tree's private-card array is `observe_terminal`,
and it reads the opponent's card only when the terminal is a showdown.  Everything the
model sees is derived from the `HandObservation` record produced there.

Event tokens (one token per game event) carry categorical and numeric fields:

categorical (index 0 = NONE / padding)
    event_type   : PAD, HAND_START, ACTION, PUBLIC_CARD, SHOWDOWN, HAND_END
    actor        : NONE, SELF, OPPONENT, CHANCE
    round        : NONE, 1, 2
    action_type  : NONE, FOLD, CHECK_CALL, RAISE
    our_rank     : NONE, J, Q, K              (in every real event of the hand)
    public_rank  : NONE, J, Q, K              (from the PUBLIC_CARD event onwards)
    opp_rank     : NONE, J, Q, K              (ONLY in a SHOWDOWN event)
    terminal_type: NONE, FOLD_SELF, FOLD_OPP, SHOWDOWN   (only in HAND_END)
    legal_cat    : NONE, {CALL,RAISE}, {FOLD,CALL,RAISE}, {FOLD,CALL}   (ACTION: actor's legal set)
numeric (public state BEFORE the event's action; normalized)
    pot_before, amount_to_call, our_contrib, opp_contrib, raises_this_round,
    contribution_delta (actual chips added by the action), payoff (HAND_END only)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..game.leduc_tree import (LeducTree, CHANCE, PLAYER, TERMINAL, FOLD, CALL, RAISE, card_rank,
                               TERM_SHOWDOWN, TERM_FOLD_SELF, TERM_FOLD_OPP, MAX_POT, MAX_CONTRIB,
                               RAISE_SIZE)

EV_PAD, EV_HAND_START, EV_ACTION, EV_PUBLIC_CARD, EV_SHOWDOWN, EV_HAND_END = range(6)
ACTOR_NONE, ACTOR_SELF, ACTOR_OPP, ACTOR_CHANCE = range(4)
CAT_FIELDS = ["event_type", "actor", "round", "action_type", "our_rank", "public_rank", "opp_rank",
              "terminal_type", "legal_cat"]
CAT_VOCAB = [6, 4, 3, 4, 4, 4, 4, 4, 4]
NUM_FIELDS = ["pot_before", "amount_to_call", "our_contrib", "opp_contrib", "raises_this_round",
              "contribution_delta", "payoff"]
NUM_SCALE = np.array([MAX_POT, 4.0, MAX_CONTRIB, MAX_CONTRIB, 2.0, 8.0, MAX_CONTRIB])
MAX_EVENTS = 12
EVENT_NAMES = ["PAD", "HAND_START", "ACTION", "PUBLIC_CARD", "SHOWDOWN", "HAND_END"]
ACTOR_NAMES = ["-", "SELF", "OPP", "CHANCE"]
ACTION_NAMES = ["-", "FOLD", "CHECK/CALL", "RAISE"]
RANK_NAMES = ["-", "J", "Q", "K"]
TERM_NAMES = ["-", "FOLD_SELF", "FOLD_OPP", "SHOWDOWN"]
LEGAL_NAMES = ["-", "{C,R}", "{F,C,R}", "{F,C}"]


def legal_category(mask) -> int:
    key = tuple(bool(v) for v in mask)
    return {(False, True, True): 1, (True, True, True): 2, (True, True, False): 3}[key]


@dataclass
class ActionObs:
    actor: int              # 0 = self, 1 = opponent
    action: int
    round: int
    pot_before: int
    to_call: int
    contrib_before: Tuple[int, int]
    raises_before: int
    delta: int
    legal_mask: Tuple[bool, bool, bool]


@dataclass
class HandObservation:
    """Everything player 0 observed in one completed hand. No hidden information."""
    our_rank: int
    public_rank: Optional[int]
    actions: List[ActionObs]
    public_card_event_index: Optional[int]     # position in `actions` before which the card came
    terminal_type: int
    revealed_opp_rank: Optional[int]           # only at showdown
    payoff: float
    final_contrib: Tuple[int, int]


def observe_terminal(tree: LeducTree, z: int, reveal_all: bool = False) -> HandObservation:
    """Extract player 0's observation of the hand that ended at terminal node z.

    reveal_all=True is the T2 censoring toggle: the opponent's private rank is revealed at the end of
    EVERY hand (also after folds); it is emitted in a SHOWDOWN-type token before HAND_END."""
    assert tree.node_type[z] == TERMINAL
    path = tree.path_actions[z]
    node = 0
    n_chance = 0
    our_rank = None; public_rank = None; pub_idx = None
    actions: List[ActionObs] = []
    for a in path:
        if tree.node_type[node] == CHANCE:
            n_chance += 1
            if n_chance == 1:
                our_rank = card_rank(a)          # our private card: observed
            elif n_chance == 2:
                pass                            # opponent's private card: NOT observed here
            else:
                public_rank = card_rank(a)
                pub_idx = len(actions)
        else:
            p = int(tree.player[node])
            contrib = (int(tree.contrib[node, 0]), int(tree.contrib[node, 1]))
            actions.append(ActionObs(
                actor=p, action=int(a), round=int(tree.round[node]),
                pot_before=int(tree.pot[node]), to_call=contrib[1 - p] - contrib[p],
                contrib_before=contrib, raises_before=int(tree.raises_this_round[node]),
                delta=tree.node_action_delta(node, int(a)),
                legal_mask=tuple(bool(v) for v in tree.infosets[p].legal_mask[tree.infoset_of_node[node]])))
        node = int(tree.child_table[node, a])
    assert node == z
    tt = int(tree.terminal_type[z])
    revealed = card_rank(int(tree.card[z, 1])) if (tt == TERM_SHOWDOWN or reveal_all) else None   # gated read
    return HandObservation(our_rank=our_rank, public_rank=public_rank, actions=actions,
                           public_card_event_index=pub_idx, terminal_type=tt, revealed_opp_rank=revealed,
                           payoff=float(tree.returns0[z]),
                           final_contrib=(int(tree.contrib[z, 0]), int(tree.contrib[z, 1])))


def tokenize(obs: HandObservation):
    """Return (cat (MAX_EVENTS, 9) int64, num (MAX_EVENTS, 7) float32, n_events)."""
    cat = np.zeros((MAX_EVENTS, len(CAT_FIELDS)), dtype=np.int64)
    num = np.zeros((MAX_EVENTS, len(NUM_FIELDS)), dtype=np.float32)
    events = []
    our = obs.our_rank + 1
    pub = 0

    def add(event_type, actor=ACTOR_NONE, rnd=0, action=0, opp_rank=0, term=0, legal=0,
            pot=0, to_call=0, c_self=0, c_opp=0, raises=0, delta=0, payoff=0.0):
        events.append(([event_type, actor, rnd, action, our, pub, opp_rank, term, legal],
                       [pot, to_call, c_self, c_opp, raises, delta, payoff]))

    add(EV_HAND_START, rnd=1, pot=2, c_self=1, c_opp=1)
    for i, ao in enumerate(obs.actions):
        if obs.public_card_event_index == i:
            pub = obs.public_rank + 1
            add(EV_PUBLIC_CARD, actor=ACTOR_CHANCE, rnd=2, pot=ao.pot_before,
                c_self=ao.contrib_before[0], c_opp=ao.contrib_before[1])
        add(EV_ACTION, actor=ACTOR_SELF if ao.actor == 0 else ACTOR_OPP, rnd=ao.round,
            action=ao.action + 1, legal=legal_category(ao.legal_mask), pot=ao.pot_before,
            to_call=ao.to_call, c_self=ao.contrib_before[0], c_opp=ao.contrib_before[1],
            raises=ao.raises_before, delta=ao.delta)
    if obs.public_card_event_index == len(obs.actions):
        # public card dealt with no subsequent action cannot happen in Leduc, kept for safety
        pub = obs.public_rank + 1
        add(EV_PUBLIC_CARD, actor=ACTOR_CHANCE, rnd=2, pot=sum(obs.final_contrib),
            c_self=obs.final_contrib[0], c_opp=obs.final_contrib[1])
    final_pot = sum(obs.final_contrib)
    rnd_final = 2 if obs.public_rank is not None else 1
    if obs.revealed_opp_rank is not None:
        add(EV_SHOWDOWN, actor=ACTOR_CHANCE, rnd=rnd_final, opp_rank=obs.revealed_opp_rank + 1,
            pot=final_pot, c_self=obs.final_contrib[0], c_opp=obs.final_contrib[1])
    add(EV_HAND_END, rnd=rnd_final, term=obs.terminal_type, pot=final_pot,
        c_self=obs.final_contrib[0], c_opp=obs.final_contrib[1], payoff=obs.payoff)
    assert len(events) <= MAX_EVENTS
    for j, (c, n) in enumerate(events):
        cat[j] = c
        num[j] = np.array(n, dtype=np.float32) / NUM_SCALE
    return cat, num, len(events)


class HandTokenTable:
    """Tokenization of every terminal history, deduplicated into observation types."""

    def __init__(self, tree: LeducTree, reveal_all: bool = False):
        self.tree = tree
        self.reveal_all = reveal_all
        keys = {}
        self.obs_type_of_terminal = np.full(tree.n_nodes, -1, dtype=np.int32)
        cats, nums, lens, reps = [], [], [], []
        for z in tree.terminals:
            cat, num, n = tokenize(observe_terminal(tree, z, reveal_all))
            key = (cat.tobytes(), num.tobytes())
            if key not in keys:
                keys[key] = len(cats)
                cats.append(cat); nums.append(num); lens.append(n); reps.append(int(z))
            self.obs_type_of_terminal[z] = keys[key]
        self.cat = np.stack(cats)          # (T, MAX_EVENTS, 9)
        self.num = np.stack(nums)          # (T, MAX_EVENTS, 7)
        self.length = np.array(lens)       # (T,)
        self.representative = np.array(reps)
        self.n_types = len(cats)

    def render(self, z: int | None = None, obs_type: int | None = None) -> str:
        t = self.obs_type_of_terminal[z] if z is not None else obs_type
        return render_tokens(self.cat[t], self.num[t], self.length[t])


def render_tokens(cat, num, n) -> str:
    lines = [f"{'#':>2} {'event':11s} {'actor':6s} {'rnd':3s} {'action':10s} {'our':3s} {'pub':3s} {'opp':3s} "
             f"{'term':9s} {'legal':7s} | {'pot':>4s} {'call':>4s} {'c_us':>4s} {'c_op':>4s} {'rais':>4s} {'delta':>5s} {'pay':>5s}"]
    raw = num * NUM_SCALE
    for j in range(n):
        c = cat[j]; r = raw[j]
        lines.append(f"{j:>2} {EVENT_NAMES[c[0]]:11s} {ACTOR_NAMES[c[1]]:6s} {c[2]:<3d} {ACTION_NAMES[c[3]]:10s} "
                     f"{RANK_NAMES[c[4]]:3s} {RANK_NAMES[c[5]]:3s} {RANK_NAMES[c[6]]:3s} {TERM_NAMES[c[7]]:9s} "
                     f"{LEGAL_NAMES[c[8]]:7s} | {r[0]:4.0f} {r[1]:4.0f} {r[2]:4.0f} {r[3]:4.0f} {r[4]:4.0f} {r[5]:5.0f} {r[6]:5.1f}")
    return "\n".join(lines)


_TABLE = {}


def get_token_table(reveal_all: bool = False) -> HandTokenTable:
    if reveal_all not in _TABLE:
        from ..game.leduc_tree import get_tree
        _TABLE[reveal_all] = HandTokenTable(get_tree(), reveal_all)
    return _TABLE[reveal_all]
