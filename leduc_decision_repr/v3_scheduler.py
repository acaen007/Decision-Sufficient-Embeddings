"""Overnight scheduler for V3: keeps <= 4 training processes, launches jobs in priority order, and runs one
evaluation at a time (predict + exact LP solve at the configured eps indices) for every finished run."""
from __future__ import annotations

import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "leduc_decision_repr/outputs/runs_v3"
O = ROOT / "leduc_decision_repr/outputs"
PY = [sys.executable, "-m"]
ENV = dict(os.environ, OMP_NUM_THREADS="1")

COMMON = ["--threads", "1", "--patience", "6"]
def train(method, seed, out, extra):
    return PY + ["leduc_decision_repr.train", "--method", method, "--seed", str(seed), "--out", str(R / out)] + COMMON + extra

JOBS = [
    # ---- T1 round B / C
    ("rec889k_s1",     train("recon", 1, "rec889k_s1", ["--steps", "6000", "--recon_hidden", "834", "--save_at", "3000"])),
    ("rec889k_s2",     train("recon", 2, "rec889k_s2", ["--steps", "6000", "--recon_hidden", "834", "--save_at", "3000"])),
    ("recjac889k_s0",  train("recon", 0, "recjac889k_s0", ["--steps", "6000", "--recon_hidden", "834", "--recon_weights", "jacobian"])),
    ("recjac889k_s1",  train("recon", 1, "recjac889k_s1", ["--steps", "6000", "--recon_hidden", "834", "--recon_weights", "jacobian"])),
    ("recjac889k_s2",  train("recon", 2, "recjac889k_s2", ["--steps", "6000", "--recon_hidden", "834", "--recon_weights", "jacobian"])),
    ("recreach889k_s0", train("recon", 0, "recreach889k_s0", ["--steps", "6000", "--recon_hidden", "834", "--recon_weights", "reach"])),
    # ---- T2 censoring toggle (3000 steps) + uncensored 131k recon at 3000 steps for the budget-matched comparison
    ("censdec133k_s0", train("decision", 0, "censdec133k_s0", ["--steps", "3000", "--decision_hidden", "100", "--dataset_tag", "revealed"])),
    ("censrec131k_s0", train("recon", 0, "censrec131k_s0", ["--steps", "3000", "--recon_hidden", "256", "--dataset_tag", "revealed"])),
    ("censdec133k_s1", train("decision", 1, "censdec133k_s1", ["--steps", "3000", "--decision_hidden", "100", "--dataset_tag", "revealed"])),
    ("censrec131k_s1", train("recon", 1, "censrec131k_s1", ["--steps", "3000", "--recon_hidden", "256", "--dataset_tag", "revealed"])),
    ("censdec133k_s2", train("decision", 2, "censdec133k_s2", ["--steps", "3000", "--decision_hidden", "100", "--dataset_tag", "revealed"])),
    ("censrec131k_s2", train("recon", 2, "censrec131k_s2", ["--steps", "3000", "--recon_hidden", "256", "--dataset_tag", "revealed"])),
    ("rec131k3k_s0",   train("recon", 0, "rec131k3k_s0", ["--steps", "3000", "--recon_hidden", "256"])),
    ("rec131k3k_s1",   train("recon", 1, "rec131k3k_s1", ["--steps", "3000", "--recon_hidden", "256"])),
    ("rec131k3k_s2",   train("recon", 2, "rec131k3k_s2", ["--steps", "3000", "--recon_hidden", "256"])),
    # ---- T4 SPO+ (DEC-133k architecture, eps_train 0.1, exact SPO+ on 8 of 32 samples)
    ("spo0_133k_s0",   train("decision", 0, "spo0_133k_s0", ["--steps", "6000", "--decision_hidden", "100", "--spo_lambda", "0", "--spo_subset", "8"])),
    ("spo0_133k_s1",   train("decision", 1, "spo0_133k_s1", ["--steps", "6000", "--decision_hidden", "100", "--spo_lambda", "0", "--spo_subset", "8"])),
    ("spo0_133k_s2",   train("decision", 2, "spo0_133k_s2", ["--steps", "6000", "--decision_hidden", "100", "--spo_lambda", "0", "--spo_subset", "8"])),
    ("spo1_133k_s0",   train("decision", 0, "spo1_133k_s0", ["--steps", "6000", "--decision_hidden", "100", "--spo_lambda", "1.0", "--spo_subset", "8"])),
    ("spo03_133k_s0",  train("decision", 0, "spo03_133k_s0", ["--steps", "6000", "--decision_hidden", "100", "--spo_lambda", "0.3", "--spo_subset", "8"])),
    ("spo1_133k_s1",   train("decision", 1, "spo1_133k_s1", ["--steps", "6000", "--decision_hidden", "100", "--spo_lambda", "1.0", "--spo_subset", "8"])),
    ("spo1_133k_s2",   train("decision", 2, "spo1_133k_s2", ["--steps", "6000", "--decision_hidden", "100", "--spo_lambda", "1.0", "--spo_subset", "8"])),
    # ---- T5b count features (gated on the T5B_READY marker)
    ("count133k_s0",   train("decision", 0, "count133k_s0", ["--steps", "6000", "--decision_hidden", "100", "--extra_features", "1"])),
    ("count133k_s1",   train("decision", 1, "count133k_s1", ["--steps", "6000", "--decision_hidden", "100", "--extra_features", "1"])),
    ("count133k_s2",   train("decision", 2, "count133k_s2", ["--steps", "6000", "--decision_hidden", "100", "--extra_features", "1"])),
    ("recreach889k_s1", train("recon", 1, "recreach889k_s1", ["--steps", "6000", "--recon_hidden", "834", "--recon_weights", "reach"])),
    ("recreach889k_s2", train("recon", 2, "recreach889k_s2", ["--steps", "6000", "--recon_hidden", "834", "--recon_weights", "reach"])),
]
GATES = {"count133k_s0": O / "T5B_READY", "count133k_s1": O / "T5B_READY", "count133k_s2": O / "T5B_READY"}

# evaluation config per run-name prefix: (eval subdir, dataset_tag, eps_idx, extra ckpts)
def eval_cfg(name):
    base = name.rsplit("_s", 1)[0]
    if base.startswith("cens"):
        return ("test_revealed", "revealed", "2", [])
    if base in ("dec133k", "rec889k"):
        return ("test", "", "1,2,3", ["step3000.pt"])
    return ("test", "", "2", [])


def n_training():
    out = subprocess.run(["pgrep", "-fc", "leduc_decision_repr.train "], capture_output=True, text=True).stdout.strip()
    return int(out or 0)


def evaluate(name):
    sub, tag, eps_idx, extra = eval_cfg(name)
    base = name.rsplit("_s", 1)[0]; seed = name.rsplit("_s", 1)[1]
    out_dir = O / "eval" / sub
    runs = [f"NEURAL_{base.upper()}_s{seed}={R / name}"] + [f"NEURAL_{base.upper()}3K_s{seed}={R / name}:{c}" for c in extra]
    methods = ",".join(r.split("=")[0] for r in runs)
    log = open(O / "v3_eval.log", "a")
    cmd = PY + ["leduc_decision_repr.evaluate", "predict", "--split", "test", "--out", str(out_dir), "--skip_classical", "--threads", "2", "--runs", ",".join(runs)]
    if tag:
        cmd += ["--dataset_tag", tag]
    subprocess.run(["nice", "-n", "5"] + cmd, env=ENV, stdout=log, stderr=log)
    cmd = PY + ["leduc_decision_repr.evaluate", "solve", "--split", "test", "--out", str(out_dir), "--workers", "2", "--methods", methods, "--eps_idx", eps_idx]
    subprocess.run(["nice", "-n", "5"] + cmd, env=ENV, stdout=log, stderr=log)
    (R / name / "EVALUATED").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    log.write(f"evaluated {name} -> {sub} at {time.strftime('%H:%M:%S')}\n"); log.close()


def main():
    R.mkdir(parents=True, exist_ok=True)
    pending = list(JOBS); running = {}
    state_log = open(O / "v3_scheduler.log", "a")
    while pending or running or any((R / n).exists() and (R / n / "result.json").exists() and not (R / n / "EVALUATED").exists() for n in os.listdir(R) if (R / n).is_dir()):
        # launch
        while pending and n_training() < 4:
            name, cmd = pending[0]
            if name in GATES and not GATES[name].exists():
                break
            if (R / name / "result.json").exists():
                pending.pop(0); continue
            pending.pop(0)
            (R / name).mkdir(exist_ok=True)
            p = subprocess.Popen(cmd, env=ENV, stdout=open(R / f"{name}.log", "w"), stderr=subprocess.STDOUT)
            running[name] = p; state_log.write(f"{time.strftime('%H:%M:%S')} launched {name}\n"); state_log.flush()
            time.sleep(5)
        # reap
        for name, p in list(running.items()):
            if p.poll() is not None:
                state_log.write(f"{time.strftime('%H:%M:%S')} finished {name} rc={p.returncode}\n"); state_log.flush(); del running[name]
        # evaluate one finished run
        todo = sorted(n for n in os.listdir(R) if (R / n).is_dir() and (R / n / "result.json").exists() and not (R / n / "EVALUATED").exists())
        if todo:
            evaluate(todo[0])
        else:
            time.sleep(30)
        if (O / "V3_STOP").exists():
            break
    state_log.write(f"{time.strftime('%H:%M:%S')} scheduler done\n"); state_log.close()


if __name__ == "__main__":
    main()
