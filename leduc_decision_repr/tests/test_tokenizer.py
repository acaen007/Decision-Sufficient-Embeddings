"""Tokenizer / simulator test suite (task section 14, items 1-8; 9-10 live in test_models.py)."""
import re
import numpy as np
import pytest

from leduc_decision_repr.game.leduc_tree import (get_tree, TERMINAL, PLAYER, CHANCE, TERM_SHOWDOWN,
                                                  TERM_FOLD_OPP, TERM_FOLD_SELF, card_rank, ANTE, RAISE_SIZE)
from leduc_decision_repr.data.tokenizer import (get_token_table, observe_terminal, tokenize, HandTokenTable,
                                                 EV_ACTION, EV_SHOWDOWN, EV_HAND_END, EV_PUBLIC_CARD, EV_HAND_START,
                                                 EV_PAD, NUM_SCALE, CAT_FIELDS, NUM_FIELDS)
from leduc_decision_repr.data.simulator import simulate_streams, simulate_streams_reference, node_probability_table
from leduc_decision_repr.game.sequence_form import get_sequence_form

T = get_tree()
TAB = get_token_table()
S = get_sequence_form()
CI = {n: i for i, n in enumerate(CAT_FIELDS)}
NI = {n: i for i, n in enumerate(NUM_FIELDS)}


def path_signature(z, include_opp_card):
    """(our rank, public rank, action list, [opp rank]) of terminal z."""
    path = T.path_actions[z]
    node = 0; chance = []; acts = []
    for a in path:
        if T.node_type[node] == CHANCE:
            chance.append(a)
        else:
            acts.append(a)
        node = T.child_table[node, a]
    sig = [card_rank(chance[0]), card_rank(chance[2]) if len(chance) > 2 else -1, tuple(acts)]
    if include_opp_card:
        sig.append(card_rank(chance[1]))
    return tuple(sig)


# 1. deterministic tokenization
def test_deterministic_tokenization():
    tab2 = HandTokenTable(T)
    assert tab2.n_types == TAB.n_types
    assert np.array_equal(tab2.cat, TAB.cat) and np.array_equal(tab2.num, TAB.num)
    assert np.array_equal(tab2.obs_type_of_terminal, TAB.obs_type_of_terminal)
    for z in T.terminals[::97]:
        c1, n1, l1 = tokenize(observe_terminal(T, z)); c2, n2, l2 = tokenize(observe_terminal(T, z))
        assert np.array_equal(c1, c2) and np.array_equal(n1, n2) and l1 == l2


# 2. hidden-card leakage
def test_no_opponent_rank_outside_showdown_event():
    opp = TAB.cat[:, :, CI["opp_rank"]]
    ev = TAB.cat[:, :, CI["event_type"]]
    assert np.all(opp[ev != EV_SHOWDOWN] == 0)
    # SHOWDOWN events only after every action, and only in showdown hands
    for t in range(TAB.n_types):
        evs = ev[t, : TAB.length[t]]
        z = TAB.representative[t]
        if T.terminal_type[z] == TERM_SHOWDOWN:
            assert list(evs[-2:]) == [EV_SHOWDOWN, EV_HAND_END]
        else:
            assert EV_SHOWDOWN not in evs
    # the opponent's card is not recoverable from the tokens of a folded hand: every fold terminal
    # sharing (our rank, public rank, actions) but differing in the opponent card maps to one type
    groups = {}
    for z in T.terminals:
        if T.terminal_type[z] != TERM_SHOWDOWN:
            groups.setdefault(path_signature(z, False), set()).add(int(TAB.obs_type_of_terminal[z]))
    assert all(len(v) == 1 for v in groups.values())
    # and those groups really do contain different opponent cards
    ncards = {}
    for z in T.terminals:
        if T.terminal_type[z] != TERM_SHOWDOWN:
            ncards.setdefault(path_signature(z, False), set()).add(int(T.card[z, 1]))
    assert max(len(v) for v in ncards.values()) >= 3


# 3. folded hands
def test_folded_hands_hide_opponent_card_even_though_tree_knows_it():
    for z in T.terminals:
        if T.terminal_type[z] in (TERM_FOLD_OPP, TERM_FOLD_SELF):
            assert T.card[z, 1] >= 0                       # OpenSpiel/tree knows the card
            obs = observe_terminal(T, z)
            assert obs.revealed_opp_rank is None            # observation does not
            t = TAB.obs_type_of_terminal[z]
            assert np.all(TAB.cat[t, :, CI["opp_rank"]] == 0)
            assert TAB.cat[t, TAB.length[t] - 1, CI["terminal_type"]] == T.terminal_type[z]


# 4. showdown
def test_showdown_reveals_rank_exactly_when_legitimate():
    for z in T.terminals:
        t = TAB.obs_type_of_terminal[z]
        evs = TAB.cat[t, : TAB.length[t], CI["event_type"]]
        if T.terminal_type[z] == TERM_SHOWDOWN:
            j = int(np.flatnonzero(evs == EV_SHOWDOWN)[0])
            assert TAB.cat[t, j, CI["opp_rank"]] == card_rank(int(T.card[z, 1])) + 1
        else:
            assert not np.any(evs == EV_SHOWDOWN)


# 5. rank canonicalization
def test_rank_canonicalization():
    groups = {}
    for z in T.terminals:
        groups.setdefault(path_signature(z, True), set()).add(int(TAB.obs_type_of_terminal[z]))
    assert all(len(v) == 1 for v in groups.values())
    # physical copies exist: e.g. our card 0 and 1 are both J
    sigs_with_multiple_physical = 0
    phys = {}
    for z in T.terminals:
        phys.setdefault(path_signature(z, True), set()).add(tuple(int(c) for c in T.card[z]))
    assert max(len(v) for v in phys.values()) >= 2
    # for showdown hands, all distinct (our, opp, pub) *rank* combos are distinct types when actions equal
    assert TAB.n_types == len(set(path_signature(z, T.terminal_type[z] == TERM_SHOWDOWN) for z in T.terminals))


# 6. action alignment: replay the public state with Leduc rules from the token action sequence
def test_action_tokens_describe_state_before_action():
    for t in range(TAB.n_types):
        cat, num, n = TAB.cat[t], np.round(TAB.num[t].astype(np.float64) * NUM_SCALE, 3), TAB.length[t]
        contrib = [ANTE, ANTE]; rnd = 1; raises = 0
        for j in range(n):
            ev = cat[j, CI["event_type"]]
            if ev == EV_PUBLIC_CARD:
                rnd = 2; raises = 0
                assert cat[j, CI["round"]] == 2 and cat[j, CI["public_rank"]] > 0
            if ev != EV_ACTION:
                continue
            actor = 0 if cat[j, CI["actor"]] == 1 else 1
            assert num[j, NI["pot_before"]] == sum(contrib)
            assert num[j, NI["our_contrib"]] == contrib[0] and num[j, NI["opp_contrib"]] == contrib[1]
            assert num[j, NI["amount_to_call"]] == contrib[1 - actor] - contrib[actor]
            assert num[j, NI["raises_this_round"]] == raises
            assert cat[j, CI["round"]] == rnd
            a = cat[j, CI["action_type"]] - 1
            to_call = contrib[1 - actor] - contrib[actor]
            if a == 1:
                contrib[actor] += to_call
            elif a == 2:
                contrib[actor] += to_call + RAISE_SIZE[rnd]; raises += 1
            # public rank known only after the PUBLIC_CARD event
            assert (cat[j, CI["public_rank"]] > 0) == (rnd == 2)
        # HAND_END carries final pot and payoff consistent with contributions
        last = n - 1
        assert cat[last, CI["event_type"]] == EV_HAND_END
        assert num[last, NI["pot_before"]] == sum(contrib)
        pay = num[last, NI["payoff"]]
        assert pay in (contrib[1], -contrib[0], 0.0)


# 7. action-size transition
def test_contribution_delta_matches_actual_transition():
    money_re = re.compile(r"Money \(player_0 player_1\): (\d+) (\d+)")
    checked = 0
    for z in T.terminals[::13]:
        obs = observe_terminal(T, z)
        node = 0; k = 0
        for a in T.path_actions[z]:
            if T.node_type[node] == PLAYER:
                ao = obs.actions[k]; k += 1
                p = T.player[node]
                child = T.child_table[node, a]
                assert ao.delta == T.contrib[child, p] - T.contrib[node, p]
                if T.node_type[child] != TERMINAL:
                    st = T.state_of_node(child)
                    m = money_re.search(str(st))
                    assert ao.delta == (100 - int(m.group(1 + p))) - T.contrib[node, p]
                    checked += 1
            node = T.child_table[node, a]
    assert checked > 100
    # token-level: delta equals to_call (+ raise size) by action type
    for t in range(TAB.n_types):
        cat, num, n = TAB.cat[t], np.round(TAB.num[t].astype(np.float64) * NUM_SCALE, 3), TAB.length[t]
        for j in range(n):
            if cat[j, CI["event_type"]] != EV_ACTION:
                continue
            a = cat[j, CI["action_type"]] - 1
            rnd = cat[j, CI["round"]]
            expect = {0: 0, 1: num[j, NI["amount_to_call"]], 2: num[j, NI["amount_to_call"]] + RAISE_SIZE[rnd]}[a]
            assert num[j, NI["contribution_delta"]] == expect


# 8. prefix integrity (stream generation)
def test_prefix_integrity_and_reference_simulator():
    rng = np.random.default_rng(0)
    q = S.normalize_policy(1, rng.dirichlet(np.ones(3), size=S.n_infosets[1]))
    p0 = S.normalize_policy(0, rng.dirichlet(np.ones(3), size=S.n_infosets[0]))
    full = simulate_streams(T, p0, q, 3, 500, 7, 0, 5)
    short = simulate_streams(T, p0, q, 3, 20, 7, 0, 5)
    assert np.array_equal(full[:, :20], short)
    ref = simulate_streams_reference(T, p0, q, 3, 60, 7, 0, 5)
    assert np.array_equal(full[:, :60], ref)
    # different streams / opponents differ
    other = simulate_streams(T, p0, q, 3, 500, 7, 0, 6)
    assert not np.array_equal(full, other)
    assert not np.array_equal(full[0], full[1])


def test_simulator_matches_reach_probabilities():
    rng = np.random.default_rng(1)
    q = S.normalize_policy(1, rng.dirichlet(np.ones(3), size=S.n_infosets[1]))
    p0 = S.normalize_policy(0, rng.dirichlet(np.ones(3), size=S.n_infosets[0]))
    x, y = S.behavioral_to_realization(0, p0), S.behavioral_to_realization(1, q)
    reach = np.zeros(T.n_nodes)
    for z in T.terminals:
        reach[z] = T.reach_chance[z] * x[T.seq_before[z, 0]] * y[T.seq_before[z, 1]]
    sims = simulate_streams(T, p0, q, 40, 500, 11, 0, 0).ravel()      # 20000 hands
    counts = np.bincount(sims, minlength=T.n_nodes)
    # aggregate to observation types to keep cells populated; compare frequencies
    emp = np.bincount(TAB.obs_type_of_terminal[sims], minlength=TAB.n_types) / len(sims)
    exp = np.zeros(TAB.n_types)
    np.add.at(exp, TAB.obs_type_of_terminal[T.terminals], reach[T.terminals])
    assert abs(exp.sum() - 1) < 1e-9
    # chi-square-like check on the bins with enough expected mass
    m = exp * len(sims) >= 20
    chi2 = np.sum((emp[m] - exp[m]) ** 2 * len(sims) / exp[m])
    dof = m.sum()
    assert chi2 < dof + 5 * np.sqrt(2 * dof)
    assert abs(np.mean(T.returns0[sims]) - S.value(x, y)) < 0.1
