"""EM-prior check: learned-prior EM (kappa_N from validation, V3 T5c recipe) with the fine-tuned RECON-JAC q_hat
(3 seeds) as prior, for the SPO+ arm and for the control arm; then exact solve + audit at eps = 0.10."""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import fcntl, subprocess, sys, time
from .common import OUT
from .evaluate import solve

if __name__ == "__main__":
    ls = (OUT / "ft" / "LAMBDA_STAR").read_text().strip(); R = OUT / "runs_ft"; t0 = time.time()
    arms = {"_FTSPO": [str(R / f"jac_spo{ls}_s{s}") for s in range(3)], "_FTCTRL": [str(R / f"jac_ctrl_s{s}") for s in range(3)]}
    for suffix, runs in arms.items():
        subprocess.run([sys.executable, "-m", "leduc_decision_repr.t5_hybrids", "--recon_runs", ",".join(runs), "--suffix", suffix], check=True)
    out = OUT / "eval" / "test"
    with open(out / "predict.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        subprocess.run([sys.executable, "-m", "leduc_decision_repr.merge_hyb_meta"], check=True)
        fcntl.flock(lk, fcntl.LOCK_UN)
    solve(out, ["HYB_PRIOR_EM_FTSPO", "HYB_PRIOR_EM_FTCTRL"], workers=2, eps_idx=[2])
    (OUT / "ft" / "EMPRIOR_DONE").write_text(f"{time.time()-t0:.0f}s\n"); print(f"EM prior done in {time.time()-t0:.0f}s", flush=True)
