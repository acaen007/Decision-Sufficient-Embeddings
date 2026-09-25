"""Generalization study (REPORT_LEDUC_GENERALIZATION.md section 0): build the opponent families, their decision
vectors g = A y, oracle values V_0 (eps = 0) and V_eps (eps = 0.10), and observation histories.

Families: ID-REF (original test opponents), NEAR (training generators, parameters outside the training ranges),
FAR-ARCH (rule-based archetypes), FAR-CFR (few-iteration CFR / MCCFR average strategies), FAR-EXPL (logit-softened
best responses to random player-0 strategies), NE (opponent equilibria).  All policies are suit-symmetrized and
stored at the rank level.  Writes outputs/gen/families.npz and families_meta.json."""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, time
import numpy as np
import pyspiel
from .common import OUT, save_json, EPSILONS
from .game.leduc_tree import get_tree, NUM_ACTIONS, FOLD, CALL, RAISE
from .game.sequence_form import get_sequence_form
from .game.symmetry import get_symmetry
from .game.policy_utils import RankPolicyToG, INFOSET_FEATURE_NAMES
from .game.safe_lp import get_solver, OpenSpielAuditor
from .data.opponents import OpponentGenerator, masked_softmax, floor_policy, FAMILIES
from .data.datasets import load_population, find_dataset_dir, load_split
from .data.simulator import simulate_streams
from .data.tokenizer import get_token_table

D = OUT / "gen"
FAMS = ["ID-REF", "NEAR", "FAR-ARCH", "FAR-CFR", "FAR-EXPL", "NE"]
N_HANDS = 500; STREAM_SEED = 7
SPLIT_ID = {"NEAR": 20, "FAR-ARCH": 21, "FAR-CFR": 22, "FAR-EXPL": 23, "NE": 24}
STREAMS = {"NEAR": 2, "FAR-ARCH": 2, "FAR-CFR": 2, "FAR-EXPL": 2, "NE": 20}


class Builder:
    def __init__(self):
        self.T, self.S, self.sym = get_tree(), get_sequence_form(), get_symmetry()
        self.pop = load_population(); self.L = get_solver(); self.aud = OpenSpielAuditor(self.S, self.L.v_star)
        self.gen = OpponentGenerator(self.T, self.sym, self.pop["blueprint1_rank"])
        self.mask = self.sym.rank_legal_mask[1]; self.phi = self.gen.phi; self.fi = self.gen.feat_idx
        self.r2g = RankPolicyToG(self.T, self.S, self.sym)

    # ------------------------------------------------------------------ helpers
    def to_rank(self, phys):
        """Suit-symmetrize a physical player-1 policy (realization-space group average) and return rank level."""
        sym_phys = self.sym.symmetrize_policy(self.S, 1, phys)
        return self.sym.reduce(1, sym_phys, check_tied=True)

    def check(self, q):
        assert np.all(q[~self.mask] == 0) and np.allclose(q.sum(1), 1.0) and np.all(q >= 0)

    # ------------------------------------------------------------------ NEAR
    def near(self, fam, k):
        g = self.gen; rng = np.random.default_rng([2026, 10 + FAMILIES.index(fam), k]); m, F, mask = g.m, g.F, self.mask
        if fam == "NASH_LOGIT_PERTURB":
            tau = float(np.exp(rng.uniform(np.log(2.5), np.log(6.0)))); ns = float(rng.uniform(1.2, 2.0))
            W = rng.normal(0, 1.0, size=(F, NUM_ACTIONS)); eps = rng.normal(0, ns, size=(m, NUM_ACTIONS))
            q = masked_softmax(np.log(floor_policy(g.nash, mask, 0.02) + 1e-300) + tau * (self.phi @ W / g.phi_norm + eps), mask)
            return q, {"tau": tau, "noise_scale": ns}
        if fam == "NASH_RANDOM_MIX":
            eta = float(rng.uniform(0.9, 1.0))
            rand = np.where(mask, rng.dirichlet(0.25 * np.ones(NUM_ACTIONS), size=m), 0.0) + 1e-12 * mask
            rand = rand / rand.sum(1, keepdims=True)
            q = (1 - eta) * g.nash + eta * rand; q = q * mask; q = q / q.sum(1, keepdims=True)
            return q, {"eta": eta, "dirichlet": 0.25}
        if fam == "STRUCTURED_CORRELATED":
            fi = self.fi
            traits = rng.choice([-1.0, 1.0], size=8) * rng.uniform(2.5, 4.0, size=8)
            aggression, passivity, strength_sens, bluff, overfold, overcall, r2_shift, tight = traits
            beta = float(rng.uniform(1.5, 2.5))
            W = rng.normal(0, 0.6, size=(F, NUM_ACTIONS)); b = np.zeros(NUM_ACTIONS)
            b[RAISE] += 1.2 * aggression - 0.8 * passivity; b[CALL] += 0.8 * passivity; b[FOLD] += 1.0 * overfold - 0.8 * overcall + 0.6 * tight
            W[fi["hand_strength"], RAISE] += 1.5 * strength_sens + 0.5; W[fi["hand_strength"], FOLD] -= 1.5 * strength_sens + 0.5
            W[fi["rank_J"], RAISE] += 1.2 * bluff; W[fi["rank_J"], FOLD] -= 0.8 * bluff
            W[fi["facing_raise"], FOLD] += 1.0 * overfold + 0.4 * tight; W[fi["facing_raise"], CALL] += 0.8 * overcall
            W[fi["round2"], RAISE] += 0.8 * r2_shift; W[fi["round2"], FOLD] -= 0.5 * r2_shift
            W[fi["paired"], RAISE] += 1.0 + 0.5 * strength_sens; W[fi["raises_this_round_2"], RAISE] -= 0.5 * tight
            eps = rng.normal(0, 0.3, size=(m, NUM_ACTIONS))
            q = masked_softmax(beta * (b[None, :] + self.phi @ W + eps), mask)
            return q, {"traits": traits.tolist(), "beta": beta}
        if fam == "UNSTRUCTURED_DIRICHLET":
            lo, hi = (0.04, 0.15) if k < 13 else (8.0, 30.0)
            alpha = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            q = np.zeros((m, NUM_ACTIONS))
            for I in range(m):
                legal = np.flatnonzero(mask[I]); d = rng.dirichlet(alpha * np.ones(len(legal))) + 1e-12; q[I, legal] = d / d.sum()
            return q, {"alpha": alpha}

    # ------------------------------------------------------------------ FAR-ARCH
    def arch(self, kind, k):
        rng = np.random.default_rng([2026, 20 + ["ROCK", "CALLING_STATION", "MANIAC", "TAG"].index(kind), k])
        s = self.phi[:, self.fi["hand_strength"]]; facing = self.phi[:, self.fi["facing_raise"]] > 0.5
        paired = self.phi[:, self.fi["paired"]] > 0.5; mask = self.mask
        sig = lambda z: 1.0 / (1.0 + np.exp(-z)); kk = float(rng.uniform(6, 15)); prm = {"slope": kk}
        if kind == "ROCK":
            tf, tr = float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.7, 0.95)); prm.update(tau_fold=tf, tau_raise=tr)
            pf = sig(kk * (tf - s)); pr = sig(kk * (s - tr))
        elif kind == "CALLING_STATION":
            f, r, rp = float(rng.uniform(0, 0.08)), float(rng.uniform(0, 0.08)), float(rng.uniform(0, 0.3)); prm.update(fold=f, raise_=r, raise_paired=rp)
            pf = np.full(len(s), f); pr = r + rp * paired
        elif kind == "MANIAC":
            f, r = float(rng.uniform(0, 0.08)), float(rng.uniform(0.6, 0.95)); prm.update(fold=f, raise_=r)
            pf = np.full(len(s), f); pr = np.minimum(r + 0.05 * (s + 0.6), 0.99)
        else:  # TAG
            tf, tr, b = float(rng.uniform(-0.3, 0.3)), float(rng.uniform(0.2, 0.7)), float(rng.uniform(0.05, 0.25)); prm.update(tau_fold=tf, tau_raise=tr, bluff=b)
            pf = sig(kk * (tf - s)); pr = sig(kk * (s - tr)); pr = np.where(~facing, pr + (1 - pr) * b, pr)
        q = np.zeros((len(s), 3))
        pf = np.where(facing & mask[:, FOLD], pf, 0.0); pr = np.where(mask[:, RAISE], pr, 0.0)
        tot = pf + pr; scale = np.where(tot > 0.99, 0.99 / np.maximum(tot, 1e-12), 1.0); pf, pr = pf * scale, pr * scale
        q[:, FOLD], q[:, RAISE] = pf, pr; q[:, CALL] = 1.0 - pf - pr
        q = np.where(mask, q, 0.0); q = q / q.sum(1, keepdims=True)
        uni = mask / mask.sum(1, keepdims=True)
        return 0.95 * q + 0.05 * uni, {"archetype": kind, **prm}

    # ------------------------------------------------------------------ OpenSpiel policies -> our physical policy
    def p1_states(self):
        if getattr(self, "_p1", None) is None:
            game = self.T.game; idx = self.T.infosets[1].index; found = {}
            def walk(st):
                if st.is_terminal():
                    return
                if st.is_chance_node():
                    for a, _ in st.chance_outcomes():
                        walk(st.child(a))
                    return
                if st.current_player() == 1:
                    key = st.information_state_string(1)
                    if key not in found:
                        found[key] = st.clone()
                for a in st.legal_actions():
                    walk(st.child(a))
            walk(game.new_initial_state())
            self._p1 = [(idx[k], s) for k, s in found.items()]
            assert len(self._p1) == self.T.n_infosets[1]
        return self._p1

    def from_openspiel(self, policy):
        mask_phys = self.T.infosets[1].legal_mask; pol = np.zeros((self.T.n_infosets[1], 3)); n_default = 0
        for I, st in self.p1_states():
            sp = policy.get_state_policy(st)
            for a, p in sp:
                pol[I, a] = p
            if pol[I].sum() <= 0:
                pol[I] = mask_phys[I] / mask_phys[I].sum(); n_default += 1
            pol[I] = np.where(mask_phys[I], pol[I], 0.0); pol[I] /= pol[I].sum()
        return pol, n_default

    def cfr(self, variant, iters, seed=None):
        game = self.T.game
        if variant == "CFR":
            s = pyspiel.CFRSolver(game)
        elif variant == "CFR+":
            s = pyspiel.CFRPlusSolver(game)
        elif variant == "ES-MCCFR":
            s = pyspiel.ExternalSamplingMCCFRSolver(game, seed=int(seed))
        else:
            s = pyspiel.OutcomeSamplingMCCFRSolver(game, seed=int(seed))
        step = s.evaluate_and_update_policy if variant in ("CFR", "CFR+") else s.run_iteration
        for _ in range(iters):
            step()
        pol, nd = self.from_openspiel(s.average_policy())
        return self.to_rank(pol), {"variant": variant, "iterations": iters, "seed": seed, "n_default_uniform_infosets": nd}

    # ------------------------------------------------------------------ FAR-EXPL: quantal response to random x
    def quantal_response(self, x0, temp):
        T, S = self.T, self.S; inf = T.infosets[1]; n1 = T.n_seq[1]; nI = T.n_infosets[1]
        c1 = -(S.A.T @ x0)                                                       # player-1 payoff per own sequence
        if getattr(self, "_seq_of", None) is None:
            self._seq_of = {(T.seq_infoset[1][s], T.seq_action[1][s]): s for s in range(1, n1)}
            kids = [[] for _ in range(n1)]
            for I in range(nI):
                kids[inf.parent_seq[I]].append(I)
            self._kids = kids
        w = np.array([sum(T.reach_chance[n] * x0[T.seq_before[n, 0]] for n in T.infoset_nodes[1][I]) for I in range(nI)])
        Vinf = np.zeros(nI); pol = np.zeros((nI, 3))
        for I in T.infoset_order[1][::-1]:                                      # deepest first
            legal = np.flatnonzero(inf.legal_mask[I])
            vals = np.array([c1[self._seq_of[(I, a)]] + sum(Vinf[J] for J in self._kids[self._seq_of[(I, a)]]) for a in legal])
            cond = vals / w[I] if w[I] > 0 else np.zeros_like(vals)
            z = cond / temp; z = z - z.max(); p = np.exp(z); p /= p.sum()
            pol[I, legal] = p; Vinf[I] = float(p @ vals)
        return pol

    def expl(self, k):
        rng = np.random.default_rng([2026, 40, k]); alpha = float(rng.choice([0.5, 1.0, 2.0]))
        temp = float(np.exp(rng.uniform(np.log(0.1), np.log(2.0))))
        m0 = self.sym.rank_legal_mask[0]; q0 = np.zeros(m0.shape)
        for I in range(m0.shape[0]):
            legal = np.flatnonzero(m0[I]); q0[I, legal] = rng.dirichlet(alpha * np.ones(len(legal)))
        x0 = self.S.behavioral_to_realization(0, self.sym.expand(0, q0))
        pol = self.quantal_response(x0, temp)
        return self.to_rank(pol), {"alpha_p0": alpha, "temperature": temp}

    # ------------------------------------------------------------------ NE
    def ne(self):
        out = []; L, S = self.L, self.S
        s = pyspiel.CFRPlusSolver(self.T.game)
        for _ in range(2000):
            s.evaluate_and_update_policy()
        pol, _ = self.from_openspiel(s.average_policy()); out.append((self.to_rank(pol), {"kind": "CFR+ 2000 iterations"}))
        x_u = S.behavioral_to_realization(0, S.uniform_policy(0))
        dirs = [("value vs uniform learner (blueprint1 rule)", -(S.A.T @ x_u))] + \
               [(f"random Gaussian direction seed {d}", np.random.default_rng([2026, 50, d]).normal(size=S.n_seq[1])) for d in (1, 2, 3)]
        for name, c in dirs:
            ok, y, _ = L.lp1.solve_safe(c, 0.0, L.v_star_p1); assert ok
            pol = S.realization_to_behavioral(1, y); out.append((self.to_rank(pol), {"kind": "LP extreme point: " + name}))
        return out

    # ------------------------------------------------------------------ values
    def values(self, q_rank):
        g = self.r2g.g(q_rank[None])[0]; v = {}
        for eps in (0.0, 0.10):
            ok, pol, x = self.L.solve_safe(g, eps); assert ok
            v[eps] = float(x @ g)
        return g, v[0.0], v[0.10]


def main():
    t0 = time.time(); D.mkdir(parents=True, exist_ok=True); B = Builder(); tab = get_token_table(); pop = B.pop
    Q, fam, params, sub = [], [], [], []
    # ID-REF: 25 per training family from the test set
    test = load_split(find_dataset_dir(), "test"); rng = np.random.default_rng([2026, 1]); idref = []
    for f in range(4):
        ids = test["opp_ids"][pop["family_index"][test["opp_ids"]] == f]; idref += sorted(rng.choice(ids, 25, replace=False).tolist())
    for o in idref:
        Q.append(pop["rank_policies"][o]); fam.append("ID-REF"); params.append({"pop_id": int(o)}); sub.append(FAMILIES[pop["family_index"][o]])
    for f in FAMILIES:
        for k in range(25):
            q, p = B.near(f, k); Q.append(q); fam.append("NEAR"); params.append(p); sub.append(f)
    print(f"NEAR done ({time.time()-t0:.0f}s)", flush=True)
    for kind in ["ROCK", "CALLING_STATION", "MANIAC", "TAG"]:
        for k in range(25):
            q, p = B.arch(kind, k); Q.append(q); fam.append("FAR-ARCH"); params.append(p); sub.append(kind)
    cfr_specs = [("CFR", i) for i in np.linspace(1, 50, 20).astype(int)] + [("CFR+", i) for i in np.linspace(1, 50, 20).astype(int)]
    r = np.random.default_rng([2026, 30])
    cfr_specs += [("ES-MCCFR", int(r.integers(1, 51)), s) for s in range(1, 31)] + [("OS-MCCFR", int(r.integers(1, 51)) * 10, s) for s in range(1, 31)]
    for spec in cfr_specs:
        q, p = B.cfr(*spec) if len(spec) == 3 else B.cfr(spec[0], int(spec[1])); Q.append(q); fam.append("FAR-CFR"); params.append(p); sub.append(spec[0])
    print(f"FAR-CFR done ({time.time()-t0:.0f}s)", flush=True)
    for k in range(100):
        q, p = B.expl(k); Q.append(q); fam.append("FAR-EXPL"); params.append(p); sub.append("QRE-" + str(p["alpha_p0"]))
    print(f"FAR-EXPL done ({time.time()-t0:.0f}s)", flush=True)
    ne = B.ne()
    for q, p in ne:
        Q.append(q); fam.append("NE"); params.append(p); sub.append(p["kind"].split(":")[0])
    Q = np.array(Q); fam = np.array(fam); sub = np.array(sub)
    for q in Q:
        B.check(q)
    # values, exploitability of opponents (player-0 best-response gain over v*), distance to the training bank
    n = len(Q); G = np.zeros((n, B.S.n_seq[0])); V0 = np.zeros(n); Ve = np.zeros(n); opp_expl = np.zeros(n)
    for i, q in enumerate(Q):
        G[i], V0[i], Ve[i] = B.values(q)
        opp_expl[i] = B.aud.learner_br_value(B.sym.expand(1, q)) - B.L.v_star
    tr = np.flatnonzero(pop["split"] == 0); Gtr = pop["G"][tr]
    d2 = (G ** 2).sum(1)[:, None] + (Gtr ** 2).sum(1)[None] - 2 * G @ Gtr.T; nn = np.sqrt(np.maximum(d2, 0).min(1))
    idr = np.flatnonzero(fam == "ID-REF"); pid = [params[i]["pop_id"] for i in idr]
    assert np.allclose(V0[idr], pop["V_oracle"][pid, 0], atol=1e-8) and np.allclose(Ve[idr], pop["V_oracle"][pid, 2], atol=1e-8)
    print(f"values done ({time.time()-t0:.0f}s)", flush=True)
    # histories
    hist_opp, hist_stream, obs = [], [], []
    for i in range(n):
        if fam[i] == "ID-REF":
            pos = int(np.flatnonzero(test["opp_ids"] == params[i]["pop_id"])[0])
            for s_ in range(2):
                hist_opp.append(i); hist_stream.append(s_); obs.append(test["obs_types"][pos, s_, :N_HANDS])
        else:
            S_ = STREAMS[fam[i]]
            terms = simulate_streams(B.T, pop["blueprint0"], B.sym.expand(1, Q[i]), S_, N_HANDS, STREAM_SEED, SPLIT_ID[fam[i]], i)
            o = tab.obs_type_of_terminal[terms].astype(np.int16)
            for s_ in range(S_):
                hist_opp.append(i); hist_stream.append(s_); obs.append(o[s_])
    print(f"histories done ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(D / "families.npz", rank_policies=Q, family=fam, subfamily=sub, G=G, V0=V0, Veps=Ve, opp_expl=opp_expl, nn_dist_train=nn,
                        hist_opp=np.array(hist_opp), hist_stream=np.array(hist_stream), obs=np.array(obs, dtype=np.int16))
    # NE checks: distinctness and exploitability
    ne_idx = np.flatnonzero(fam == "NE"); Y = np.stack([B.S.behavioral_to_realization(1, B.sym.expand(1, Q[i])) for i in ne_idx])
    dist = [[float(np.abs(Y[a] - Y[b]).sum()) for b in range(len(ne_idx))] for a in range(len(ne_idx))]
    meta = {"families": FAMS, "counts": {f: int((fam == f).sum()) for f in FAMS},
            "histories": {f: int(sum(fam[h] == f for h in hist_opp)) for f in FAMS}, "params": params, "subfamily": sub.tolist(),
            "ne_exploitability": opp_expl[ne_idx].tolist(), "ne_pairwise_L1_realization": dist,
            "headroom_Veps_minus_V0": {f: {"median": float(np.median((Ve - V0)[fam == f])), "n_below_0.01": int(((Ve - V0)[fam == f] < 0.01).sum())} for f in FAMS},
            "nn_dist_to_train_bank_median": {f: float(np.median(nn[fam == f])) for f in FAMS},
            "opp_exploitability_median": {f: float(np.median(opp_expl[fam == f])) for f in FAMS},
            "runtime_s": time.time() - t0}
    save_json(meta, D / "families_meta.json")
    print(json.dumps({k: meta[k] for k in ["counts", "histories", "ne_exploitability", "headroom_Veps_minus_V0", "nn_dist_to_train_bank_median", "opp_exploitability_median"]}, indent=1))
    print(f"done ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
