"""Post-hoc check (declared in REPORT_LEDUC_DECISION_COMPRESSION.md deviations): deploy the *bank argmax* instead of the full
LP on g_hat.  Pure response-bank regret (lambda = 0) collapsed under LP deployment during selection.  The hypothesis is a
train/deploy mismatch: the loss only constrains g_hat along the 1 200 bank directions, while the LP optimizes over the whole
safe polytope.  Deployment here is x = B[argmax_m B_m g_hat]; every bank response is the exact eps = 0.10 safe response of a
training opponent, and the distinct ones used are audited.  Writes outputs/dcomp/posthoc_bank.json."""
import json
import numpy as np
from .common import save_json
from .dc_common import D, OOD_FAMS, load_sets, fraction, boot_fraction, LPPool
from .dc_ae import ARMS, DIMS, SEEDS, RUNS, run_name, make_model


def main():
    import torch
    from .game.leduc_tree import get_tree
    from .game.symmetry import get_symmetry
    from .game.sequence_form import get_sequence_form
    from .game.policy_utils import RankPolicyToG
    torch.set_num_threads(4); S, pop = load_sets(); sym = get_symmetry(); legal = np.asarray(sym.rank_legal_mask[1], bool)
    r2g = RankPolicyToG(get_tree(), get_sequence_form(), sym); Bk = S["train"]["X"][0.10]; used = set(); res = {}
    names = [run_name(a, d, s) for a in ARMS for d in DIMS for s in SEEDS] + sorted(p.stem for p in RUNS.glob("REGRET_d8_s0_b*"))
    for name in names:
        if not (RUNS / f"{name}.pt").exists():
            continue
        ck = torch.load(RUNS / f"{name}.pt", weights_only=False); m = make_model(ck["d"], legal); m.load_state_dict(ck["state"]); m.eval(); o = {}
        for sname in ("test", "ood"):
            s = S[sname]
            with torch.no_grad():
                _, lqh = m(torch.as_tensor(s["Q"], dtype=torch.float32))
            qh = np.where(legal[None], lqh.exp().double().numpy(), 0.0); qh /= qh.sum(2, keepdims=True); gh = r2g.g(qh)
            a = (gh @ Bk.T).argmax(1); used.update(a.tolist()); u = (Bk[a] * s["G"]).sum(1)
            o[f"{sname}_frac_bank"] = fraction(u, s["V0"], s["V"][0.10])
            if sname == "test":
                o["test_frac_bank_ci"] = boot_fraction(u, s["V0"], s["V"][0.10])
            else:
                for f in OOD_FAMS:
                    mm = s["family"] == f; o[f"ood_frac_bank_{f}"] = fraction(u[mm], s["V0"][mm], s["V"][0.10][mm])
        res[name] = o
    agg = {}
    for a in ARMS:
        for d in DIMS:
            vals = [res[run_name(a, d, s)] for s in SEEDS if run_name(a, d, s) in res]
            if vals:
                agg[f"{a}_d{d}"] = {k: float(np.mean([v[k] for v in vals])) for k in ("test_frac_bank", "ood_frac_bank")}
    pool = LPPool(4, audit=True); ids = sorted(used); _, ex, _ = pool.audit(Bk[ids]); pool.close()
    save_json({"per_run": res, "per_arm_d": agg, "audit": {"n": len(ids), "max_expl_minus_eps": float((ex - 0.10).max()), "violations": int(((ex - 0.10) > 1e-7).sum())}},
              D / "posthoc_bank.json")
    for k, v in agg.items():
        print(k, {kk: round(vv, 3) for kk, vv in v.items()})
    for n in sorted(p for p in res if "_b" in p):
        print(n, {kk: round(vv, 3) for kk, vv in res[n].items() if not kk.endswith("ci") and "_bank_" not in kk})


if __name__ == "__main__":
    main()
