"""Paths, config hashing, and small I/O helpers."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

EPSILONS = [0.0, 0.05, 0.10, 0.20]
N_BUDGETS = [5, 10, 20, 50, 100, 200, 500]
STREAM_LEN = 500
STREAMS = {"train": 4, "val": 4, "test": 8}
LATENT_DIM = 128
CODE_VERSION = "leduc_v1"


def config_hash(cfg: dict) -> str:
    s = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha1(s.encode()).hexdigest()[:10]


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=_default)


def _default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def environment_info() -> dict:
    import numpy, scipy, torch, pyspiel
    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        git = "unknown"
    return {
        "python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "numpy": numpy.__version__, "scipy": scipy.__version__, "torch": torch.__version__,
        "open_spiel": getattr(pyspiel, "__version__", "unknown"), "git": git,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self):
        return time.time() - self.t0
