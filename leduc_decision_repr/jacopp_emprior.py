"""JAC-opp EM-prior check: learned-prior EM (kappa_N from validation, V3 T5c recipe) with the q_hat of the best JAC-opp
study arm as prior.  Best arm = lowest seed-averaged selected validation regret among arms with 3 seeds (never test).
Then exact solve + audit at eps = 0.10, compared in the analysis with the existing learned-prior EM (HYB_PRIOR_EM)."""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import fcntl, json, subprocess, sys, time
import numpy as np
from .common import OUT
from .evaluate import solve

if __name__ == "__main__":
    R = OUT / "runs_jacopp"; JO = OUT / "jacopp"; t0 = time.time()
    d = json.loads((JO / "decisions.json").read_text())
    arms = {a: [f"{a}_s{s}" for s in range(3)] for a in ("a0", "a1", "a2", "a3")}
    arms["a4"] = [f"a4l{d['lambda_star']:g}_s{s}" for s in range(3)]
    val = {a: float(np.mean([json.loads((R / r / "result.json").read_text())["best_val_loss"] for r in runs]))
           for a, runs in arms.items() if all((R / r / "result.json").exists() for r in runs)}
    best = min(val, key=lambda a: (val[a], a))
    (JO / "emprior_choice.json").write_text(json.dumps({"seed_mean_val_regret": val, "best_arm": best, "runs": arms[best],
                                                          "rule": "lowest seed-averaged selected validation regret (eps 0.10, 450 fixed LPs)"}, indent=1))
    print(f"EM prior: best arm {best} ({val})", flush=True)
    subprocess.run([sys.executable, "-m", "leduc_decision_repr.t5_hybrids", "--recon_runs", ",".join(str(R / r) for r in arms[best]),
                    "--suffix", "_JOBEST"], check=True)
    out = OUT / "eval" / "test"
    with open(out / "predict.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        subprocess.run([sys.executable, "-m", "leduc_decision_repr.merge_hyb_meta"], check=True)
        fcntl.flock(lk, fcntl.LOCK_UN)
    solve(out, ["HYB_PRIOR_EM_JOBEST"], workers=2, eps_idx=[2])
    (JO / "EMPRIOR_DONE").write_text(f"{time.time()-t0:.0f}s\n"); print(f"EM prior done in {time.time()-t0:.0f}s", flush=True)
