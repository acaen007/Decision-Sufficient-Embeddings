"""Scheduler for the JAC-opp study (REPORT_LEDUC_JACOPP.md section 0): <= 4 CPU slots (training = 1, evaluation /
EM-prior = 2), first-fit in pre-registered priority order.
  1. seed 0 of arms 0-3 and of arm 4 at lambda in {0.1, 0.3, 1.0}
  2. seeds 1-2 of arm 3, arm 4 (lambda*), and the better of arms 1/2 (seed-0 selected validation regret)
  3. the rest: seeds 1-2 of arm 0 and of the other of arms 1/2
  then test evaluation of every arm (arm 4 at lambda* only) and the EM-prior check.
Decisions (lambda*, better of 1/2) use validation regret only and are written to outputs/jacopp/*.json."""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
O = ROOT / "leduc_decision_repr" / "outputs"; R = O / "runs_jacopp"; JO = O / "jacopp"
ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
PY = [sys.executable, "-m"]
COMMON = ["--method", "recon", "--steps", "6000", "--recon_hidden", "834", "--patience", "0", "--select_by", "regret", "--threads", "1"]
WEIGHTS = {"a0": "jacobian", "a1": "jac_global_proj", "a2": "opp_reach", "a3": "jac_opp"}
LAMS = [0.1, 0.3, 1.0]
LABEL = {"a0": "NEURAL_JO_A0", "a1": "NEURAL_JO_A1", "a2": "NEURAL_JO_A2", "a3": "NEURAL_JO_A3", "a4": "NEURAL_JO_A4"}
CAP = 4


def used_slots():
    """Live processes of any study (restart-safe): training / fine-tuning = 1 slot, evaluation / EM-prior = 2 slots."""
    def n(pat):
        return len(subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True).stdout.split())
    py = "^[^ ]*python[^ ]* -m leduc_decision_repr[.]"          # anchored: never matches shells whose text mentions a module
    return (n(py + "train ") + n(py + "finetune ") + 2 * n(py + "ft_eval ") + 2 * n(py + "ft_emprior") + 2 * n(py + "jacopp_emprior"))


def done(run):
    return (R / run / "result.json").exists()


def evaluated(run):
    return (R / run / "EVALUATED").exists()


def best_val(run):
    return json.loads((R / run / "result.json").read_text())["best_val_loss"]      # = selected validation regret (select_by regret)


def run_name(arm, seed, lam=None):
    return f"a4l{lam:g}_s{seed}" if arm == "a4" else f"{arm}_s{seed}"


def train_job(arm, seed, lam=None):
    name = run_name(arm, seed, lam)
    args = ["--recon_weights", "jac_opp", "--gterm_lambda", str(lam)] if arm == "a4" else ["--recon_weights", WEIGHTS[arm]]
    return {"name": name, "w": 1, "cmd": PY + ["leduc_decision_repr.train"] + COMMON + args + ["--seed", str(seed), "--out", str(R / name)],
            "done": lambda r=name: done(r)}


def eval_job(run, label):
    return {"name": f"eval_{run}", "w": 2, "cmd": PY + ["leduc_decision_repr.ft_eval", "--name", label, "--run", str(R / run), "--eps_idx", "1,2,3"],
            "ready": lambda r=run: done(r), "done": lambda r=run: evaluated(r)}


def decisions(log):
    """lambda* (arm 4, seed 0) and the better of arms 1/2 (seed 0), by selected validation regret; ties -> smaller lambda / arm 1."""
    d = json.loads((JO / "decisions.json").read_text()) if (JO / "decisions.json").exists() else {}
    changed = False
    if "lambda_star" not in d and all(done(run_name("a4", 0, l)) for l in LAMS):
        vals = {l: best_val(run_name("a4", 0, l)) for l in LAMS}
        d["lambda_star"] = min(LAMS, key=lambda l: (vals[l], l)); d["val_regret_by_lambda"] = {str(k): v for k, v in vals.items()}; changed = True
        log.write(f"{time.strftime('%H:%M:%S')} lambda* = {d['lambda_star']:g} ({vals})\n")
    if "better_12" not in d and done("a1_s0") and done("a2_s0"):
        v1, v2 = best_val("a1_s0"), best_val("a2_s0")
        d["better_12"] = "a1" if v1 <= v2 else "a2"; d["val_regret_a1_a2_seed0"] = [v1, v2]; changed = True
        log.write(f"{time.strftime('%H:%M:%S')} better of arms 1/2 = {d['better_12']} ({v1:.5f} vs {v2:.5f})\n")
    if changed:
        d["rule"] = "lowest seed-0 selected validation regret (eps 0.10, 450 fixed LPs); ties -> smaller lambda / arm 1"
        (JO / "decisions.json").write_text(json.dumps(d, indent=1)); log.flush()
    return d


def jobs(d):
    ls, b = d.get("lambda_star"), d.get("better_12")
    J = [train_job(a, 0) for a in ("a0", "a1", "a2", "a3")] + [train_job("a4", 0, l) for l in LAMS]
    J += [train_job("a3", s) for s in (1, 2)]
    if ls is not None:
        J += [train_job("a4", s, ls) for s in (1, 2)]
    if b is not None:
        J += [train_job(b, s) for s in (1, 2)]
    if ls is not None and b is not None:
        other = "a2" if b == "a1" else "a1"
        J += [train_job("a0", s) for s in (1, 2)] + [train_job(other, s) for s in (1, 2)]
    for a in ("a0", "a1", "a2", "a3"):
        J += [eval_job(run_name(a, s), f"{LABEL[a]}_s{s}") for s in range(3)]
    if ls is not None:
        J += [eval_job(run_name("a4", s, ls), f"{LABEL['a4']}_s{s}") for s in range(3)]
        train_names = [j["name"] for j in J if j["w"] == 1]
        J += [{"name": "emprior", "w": 2, "cmd": PY + ["leduc_decision_repr.jacopp_emprior"],
               "ready": lambda t=train_names: all(done(n) for n in t) and len(t) == 17, "done": lambda: (JO / "EMPRIOR_DONE").exists()}]
    return J


def main():
    R.mkdir(parents=True, exist_ok=True); JO.mkdir(parents=True, exist_ok=True)
    log = open(JO / "scheduler.log", "a"); running = {}; t0 = time.time()
    log.write(f"{time.strftime('%H:%M:%S')} scheduler start\n"); log.flush()
    while True:
        d = decisions(log)
        for n, (p, w) in list(running.items()):
            if p.poll() is not None:
                log.write(f"{time.strftime('%H:%M:%S')} finished {n} rc={p.returncode}\n"); log.flush(); del running[n]
        used = max(used_slots(), sum(w for _, w in running.values()))
        J = jobs(d); pending = [j for j in J if not j["done"]() and j["name"] not in running and not (R / (j["name"] + ".started")).exists()]
        for j in pending:
            if j.get("ready", lambda: True)() and used + j["w"] <= CAP:
                (R / (j["name"] + ".started")).write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
                p = subprocess.Popen(j["cmd"], env=ENV, cwd=ROOT, stdout=open(R / f"{j['name']}.log", "w"), stderr=subprocess.STDOUT)
                running[j["name"]] = (p, j["w"]); used += j["w"]
                log.write(f"{time.strftime('%H:%M:%S')} launched {j['name']} (w={j['w']})\n"); log.flush()
        if not running and not pending and "lambda_star" in d and "better_12" in d:
            break
        if (JO / "STOP").exists():
            break
        time.sleep(20)
    log.write(f"{time.strftime('%H:%M:%S')} scheduler done ({(time.time()-t0)/3600:.2f} h)\n"); log.close()


if __name__ == "__main__":
    main()
