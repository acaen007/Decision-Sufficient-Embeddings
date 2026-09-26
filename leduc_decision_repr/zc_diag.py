"""Post-hoc diagnostics for REPORT_LEDUC_ZCODE.md (declared in its deviations): why does predicting the regret code from
hands under-perform?
  D1 (belief representability): project the train-bank posterior mean g_bar(hands) onto the frozen d = 8 decoder's range
     (the nearest decodable g, found by Adam over the code) and compare the safe value of deploying the LP on g_bar vs on
     its projection.  Test opponents, stream 0, N in {5, 20, 100, 500}, plus the population mean (N = 0).
  D2 (code fragility): R^2 of ZC-DISTILL's predicted code against the true code by N, and the safe value of the true code
     with Gaussian noise added (sigma in standardized code units).
Every deployed strategy is an exact LP solution, audited.  Writes outputs/zcode/diag.json."""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, time
import numpy as np
from .common import OUT, N_BUDGETS, save_json
from .dc_common import LPPool, fraction

D = OUT / "zcode"; E = OUT / "eval" / "test"; AE = OUT / "dcomp" / "ae_runs" / "REGRET_d8_s0.pt"; EPS = 0.10


def main():
    import torch
    from .dc_ae import make_model
    from .data.datasets import load_population, find_dataset_dir, load_split
    from .game.leduc_tree import get_tree
    from .game.symmetry import get_symmetry
    from .game.sequence_form import get_sequence_form
    from .game.policy_utils import RankPolicyToG
    from .models.torch_g import TorchRankPolicyToG
    from .data.tokenizer import get_token_table
    from .baselines.likelihood import HandLikelihood, type_counts
    from .baselines.bank_posterior import BankPosterior
    from .train import load_trained
    torch.set_num_threads(4); t0 = time.time(); pop = load_population(); sym = get_symmetry(); legal = np.asarray(sym.rank_legal_mask[1], bool)
    r2g = RankPolicyToG(get_tree(), get_sequence_form(), sym); tg = TorchRankPolicyToG(r2g); tab = get_token_table(); lik = HandLikelihood(tab, sym)
    ck = torch.load(AE, weights_only=False); ae = make_model(ck["d"], legal); ae.load_state_dict(ck["state"]); ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    tr = np.flatnonzero(pop["split"] == 0); test = load_split(find_dataset_dir(), "test"); ids = test["opp_ids"]; obs = test["obs_types"][:, 0]
    Gt, V0, Ve = pop["G"][ids], pop["V_oracle"][ids, 0], pop["V_oracle"][ids, 2]
    with torch.no_grad():
        Z = ae.enc(torch.as_tensor(pop["rank_policies"], dtype=torch.float32).reshape(len(pop["G"]), -1))
    mu, sd = Z[tr].mean(0), Z[tr].std(0)
    def decode_g(c):                                                   # c: raw (unstandardized) codes
        lg = ae.dec(c).reshape(-1, 144, 3).masked_fill(~ae.legal[None], float("-inf")); return tg(torch.softmax(lg, -1))
    def project(Gtarget, steps=400):
        c = mu.repeat(len(Gtarget), 1).clone().requires_grad_(True); opt = torch.optim.Adam([c], lr=0.05); T = torch.as_tensor(Gtarget, dtype=torch.float32)
        for _ in range(steps):
            loss = ((decode_g(c) - T) ** 2).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            g = decode_g(c).double().numpy()
        return g, float(np.median(np.linalg.norm(g - Gtarget, axis=1) / np.linalg.norm(Gtarget - pop["G"][tr].mean(0), axis=1)))
    pool = LPPool(4, audit=True); res = {"D1": {}, "D2": {}}; audit = []
    def value(Gh):
        X, ex, ok = pool.solve(np.asarray(Gh, dtype=np.float64), EPS); audit.append(ex - EPS); assert ok.all()
        return (X * Gt).sum(1)
    # ---------------- D1: belief representability
    bank = BankPosterior(lik, pop["rank_policies"][tr], pop["G"][tr]); gbar_pop = pop["G"][tr].mean(0)
    targets = {"0": np.repeat(gbar_pop[None], len(ids), 0)}
    for N in (5, 20, 100, 500):
        targets[str(N)] = bank.g_bar(type_counts(obs[:, :N], tab.n_types))[0]
    for key, Gb in targets.items():
        Gp, relerr = project(Gb); u_b, u_p = value(Gb), value(Gp)
        res["D1"][key] = {"frac_belief": fraction(u_b, V0, Ve), "frac_projected": fraction(u_p, V0, Ve), "loss": fraction(u_b, V0, Ve) - fraction(u_p, V0, Ve),
                          "median_rel_err_projection": relerr}
        print(f"D1 N={key}: belief {res['D1'][key]['frac_belief']:.3f} -> projected {res['D1'][key]['frac_projected']:.3f} (rel err {relerr:.2f}) ({time.time()-t0:.0f}s)", flush=True)
    # true individual opponents projected (sanity: the code's own range)
    Gp, relerr = project(Gt); res["D1"]["true_opponent"] = {"frac_true": 1.0, "frac_projected": fraction(value(Gp), V0, Ve), "median_rel_err_projection": relerr}
    # ---------------- D2: code fragility
    zt = ((Z[ids] - mu) / sd).numpy(); rng = np.random.default_rng(0)
    for sig in (0.0, 0.25, 0.5, 1.0):
        zn = zt + sig * rng.standard_normal(zt.shape)
        with torch.no_grad():
            Gn = decode_g(torch.as_tensor(zn, dtype=torch.float32) * sd + mu).double().numpy()
        res["D2"][f"noise_{sig}"] = fraction(value(Gn), V0, Ve)
        print(f"D2 true code + noise {sig}: fraction {res['D2'][f'noise_{sig}']:.3f}", flush=True)
    r2 = {}
    for s in range(3):
        rd = OUT / "runs_zcode" / f"zcdist8_s{s}"
        if not (rd / "best.pt").exists():
            continue
        enc, head, _ = load_trained(rd); P = np.load(E / f"pred_NEURAL_ZC_DIST8_s{s}.npz"); ho = np.load(E / "hist_opp.npy")
        zt_h = ((Z[ho] - mu) / sd).numpy()
        with torch.no_grad():
            for j, N in enumerate(N_BUDGETS):
                zh = head.code(torch.as_tensor(P["z"][:, j])).numpy()
                r2.setdefault(str(N), []).append(float(1 - ((zh - zt_h) ** 2).sum() / ((zt_h - zt_h.mean(0)) ** 2).sum()))
    res["D2"]["distill_code_R2_by_N"] = {k: float(np.mean(v)) for k, v in r2.items()}
    ex = np.concatenate(audit); res["audit"] = {"n": int(len(ex)), "max_expl_minus_eps": float(ex.max()), "violations": int((ex > 1e-7).sum())}
    pool.close(); res["wall_s"] = time.time() - t0; save_json(res, D / "diag.json"); print(json.dumps(res["D2"], indent=1), res["audit"])


if __name__ == "__main__":
    main()
