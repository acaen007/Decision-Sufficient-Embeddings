"""Scheduler for the decision-aware fine-tuning study: <= 4 CPU slots (training job = 1 slot, evaluation = 2),
first-fit in priority order, dependencies on the validation-selected lambda* and on finished runs."""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
O = ROOT / "leduc_decision_repr" / "outputs"; R = O / "runs_ft"; FT = O / "ft"
ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
PY = [sys.executable, "-m"]
BASE = {"jac": [O / "runs_v3" / f"recjac889k_s{s}" for s in range(3)], "dec": [O / "runs" / f"decision_s{s}" for s in range(3)]}
LAMS = [0.1, 0.3, 1.0]
CAP = 4


def lam_star():
    f = FT / "LAMBDA_STAR"
    return float(f.read_text().strip()) if f.exists() else None


def done(run):
    return (R / run / "result.json").exists()


def evaluated(run):
    return (R / run / "EVALUATED").exists()


def spo_name(base, seed):
    ls = lam_star()
    return None if ls is None else f"{base}_spo{ls:g}_s{seed}"


def train_job(base, seed, lam):
    run = f"{base}_ctrl_s{seed}" if lam == 0 else f"{base}_spo{lam:g}_s{seed}"
    return {"name": run, "w": 1, "cmd": PY + ["leduc_decision_repr.finetune", "--base", str(BASE[base][seed]), "--out", str(R / run),
                                              "--lam", str(lam), "--seed", str(seed)], "done": lambda r=run: done(r)}


def eval_job(run, label):
    return {"name": f"eval_{run}", "w": 2, "cmd": PY + ["leduc_decision_repr.ft_eval", "--name", label, "--run", str(R / run), "--eps_idx", "1,2,3"],
            "ready": lambda r=run: done(r), "done": lambda r=run: evaluated(r)}


def emprior_job():
    ls = lam_star()
    runs = [f"jac_ctrl_s{s}" for s in range(3)] + ([f"jac_spo{ls:g}_s{s}" for s in range(3)] if ls is not None else [])
    return {"name": "emprior", "w": 2, "cmd": PY + ["leduc_decision_repr.ft_emprior"], "ready": lambda: ls is not None and all(done(r) for r in runs),
            "done": lambda: (FT / "EMPRIOR_DONE").exists()}


def used_slots():
    """Count live processes (restart-safe): fine-tuning = 1 slot, evaluation / EM-prior = 2 slots."""
    def n(pat):
        out = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True).stdout.split()
        return len(out)
    py = "^[^ ]*python[^ ]* -m leduc_decision_repr[.]"          # anchored: never matches shells whose text mentions a module
    return n(py + "finetune ") + 2 * n(py + "ft_eval ") + 2 * n(py + "ft_emprior")


def jobs():
    J = [train_job("jac", 0, l) for l in LAMS] + [train_job("jac", s, 0) for s in range(3)]
    ls = lam_star()
    if ls is not None:
        J += [train_job("jac", s, ls) for s in (1, 2)]
        J += [eval_job(f"jac_ctrl_s{s}", f"NEURAL_FTJACCTRL_s{s}") for s in range(3)]
        J += [eval_job(f"jac_spo{ls:g}_s{s}", f"NEURAL_FTJACSPO_s{s}") for s in range(3)]
        J += [emprior_job()]
        J += [train_job("dec", s, 0) for s in range(3)] + [train_job("dec", s, ls) for s in range(3)]
        J += [eval_job(f"dec_ctrl_s{s}", f"NEURAL_FTDECCTRL_s{s}") for s in range(3)]
        J += [eval_job(f"dec_spo{ls:g}_s{s}", f"NEURAL_FTDECSPO_s{s}") for s in range(3)]
    return J


def select_lambda(log):
    if lam_star() is not None or not all(done(f"jac_spo{l:g}_s0") for l in LAMS):
        return
    vals = {l: json.loads((R / f"jac_spo{l:g}_s0" / "result.json").read_text())["best_val_regret"] for l in LAMS}
    best = min(LAMS, key=lambda l: (vals[l], l))
    (FT / "LAMBDA_STAR").write_text(f"{best:g}\n")
    (FT / "lambda_selection.json").write_text(json.dumps({"val_regret_by_lambda": {str(k): v for k, v in vals.items()}, "lambda_star": best,
                                                           "rule": "lowest mean exact validation regret (450 points) at the selected checkpoint; ties -> smaller"}, indent=1))
    log.write(f"{time.strftime('%H:%M:%S')} lambda* = {best:g} ({vals})\n"); log.flush()


def main():
    R.mkdir(parents=True, exist_ok=True); FT.mkdir(parents=True, exist_ok=True)
    log = open(FT / "scheduler.log", "a"); running = {}; t0 = time.time()
    while True:
        select_lambda(log)
        for n, (p, w) in list(running.items()):
            if p.poll() is not None:
                log.write(f"{time.strftime('%H:%M:%S')} finished {n} rc={p.returncode}\n"); log.flush(); del running[n]
        used = used_slots()
        J = jobs(); pending = [j for j in J if not j["done"]() and j["name"] not in running and not (R / (j["name"] + ".started")).exists()]
        for j in pending:
            if j.get("ready", lambda: True)() and used + j["w"] <= CAP:
                (R / (j["name"] + ".started")).write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
                p = subprocess.Popen(j["cmd"], env=ENV, cwd=ROOT, stdout=open(R / f"{j['name']}.log", "w"), stderr=subprocess.STDOUT)
                running[j["name"]] = (p, j["w"]); used += j["w"]
                log.write(f"{time.strftime('%H:%M:%S')} launched {j['name']} (w={j['w']})\n"); log.flush()
        if not running and not pending and lam_star() is not None and used_slots() == 0:
            break
        if (FT / "STOP").exists():
            break
        time.sleep(20)
    log.write(f"{time.strftime('%H:%M:%S')} scheduler done ({(time.time()-t0)/3600:.2f} h)\n"); log.close()


if __name__ == "__main__":
    main()
