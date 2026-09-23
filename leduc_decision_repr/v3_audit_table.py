"""Safety audit table across every deployed strategy of V1/V2/V3: max (Expl - eps) and violation counts (> 1e-7)
from the OpenSpiel best response stored in solve_*.npz, plus the T2/T3 oracle audits."""
import glob, json
from pathlib import Path
import numpy as np
from .common import OUT, EPSILONS, save_json, load_json

TOL = 1e-7
rows = []
for sub in ["test", "test_revealed"]:
    d = OUT / "eval" / sub
    if not d.exists():
        continue
    for f in sorted(glob.glob(str(d / "solve_*.npz"))):
        m = Path(f).stem[len("solve_"):]; s = np.load(f)
        ok = s["ok"]; e = s["e_os"]
        for k, eps in enumerate(EPSILONS):
            sel = ok[:, :, k]
            if not sel.any():
                continue
            viol = e[:, :, k][sel] - eps
            rows.append({"eval_dir": sub, "method": m, "eps": eps, "n_strategies": int(sel.sum()), "max_expl_minus_eps": float(viol.max()),
                         "n_violations": int((viol > TOL).sum()), "lp_failures": int(s["lp_failures"]) if s["lp_failures"].shape == () else int(np.sum(s["lp_failures"]))})
extra = []
t3 = OUT / "t3_eps_rank" / "t3_results.json"
if t3.exists():
    a = load_json(t3)["audit"]; extra.append({"source": "T3 oracle (rank-k, hull, components, kappa)", "n_strategies": a["n_strategies_audited"], "max_expl_minus_eps": a["max_violation"], "lp_failures": a["lp_failures"]})
t2 = OUT / "t2_covariance" / "gaps_summary.json"
if t2.exists():
    a = load_json(t2); extra.append({"source": "T2 covariance gap (subsample)", "n_strategies": a["n_audited"], "max_expl_minus_eps": a["audit_subsample_max_violation"], "lp_failures": 0})
tot = {"n_strategies": int(sum(r["n_strategies"] for r in rows)) + int(sum(x["n_strategies"] for x in extra)),
       "max_expl_minus_eps": float(max([r["max_expl_minus_eps"] for r in rows] + [x["max_expl_minus_eps"] for x in extra])),
       "n_violations": int(sum(r["n_violations"] for r in rows)), "lp_failures": int(sum(r["lp_failures"] for r in rows) + sum(x["lp_failures"] for x in extra))}
save_json({"rows": rows, "extra": extra, "total": tot, "tolerance": TOL}, OUT / "v3_audit_table.json")
# markdown
lines = ["| eval dir | method | ε | strategies | max(Expl − ε) | violations | LP failures |", "|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['eval_dir']} | {r['method']} | {r['eps']} | {r['n_strategies']} | {r['max_expl_minus_eps']:.1e} | {r['n_violations']} | {r['lp_failures']} |")
for x in extra:
    lines.append(f"| — | {x['source']} | mixed | {x['n_strategies']} | {x['max_expl_minus_eps']:.1e} | 0 | {x['lp_failures']} |")
lines.append(f"| **total** | | | **{tot['n_strategies']}** | **{tot['max_expl_minus_eps']:.1e}** | **{tot['n_violations']}** | **{tot['lp_failures']}** |")
(OUT / "v3_audit_table.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines[-6:])); print("total", tot)
