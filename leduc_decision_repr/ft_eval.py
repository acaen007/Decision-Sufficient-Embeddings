"""Evaluate one fine-tuned run on the test split: predict (serialized by a file lock, since predict merges into
predict_meta.json) then exact safe-LP solve + OpenSpiel audit at the given eps indices."""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, fcntl, time
from pathlib import Path
from .common import OUT
from .evaluate import predict, solve

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--run", required=True)
    ap.add_argument("--eps_idx", default="1,2,3"); ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(); out = OUT / "eval" / "test"; t0 = time.time()
    with open(out / "predict.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        predict("test", {a.name: a.run}, out, skip_classical=True, threads=2)
        fcntl.flock(lk, fcntl.LOCK_UN)
    solve(out, [a.name], workers=a.workers, eps_idx=[int(v) for v in a.eps_idx.split(",")])
    Path(a.run, "EVALUATED").write_text(f"{a.name} {time.strftime('%Y-%m-%d %H:%M:%S')} {time.time()-t0:.0f}s\n")
    print(f"evaluated {a.name} in {time.time()-t0:.0f}s", flush=True)
