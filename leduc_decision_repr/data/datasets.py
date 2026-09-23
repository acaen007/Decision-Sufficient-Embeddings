"""Phase 6: fixed observation-stream datasets (train / val / test)."""
from __future__ import annotations

import glob
import time
from pathlib import Path

import numpy as np

from ..common import OUT, STREAMS, STREAM_LEN, config_hash, save_json, environment_info, CODE_VERSION
from ..game.leduc_tree import get_tree
from .simulator import simulate_streams
from .tokenizer import get_token_table

SPLIT_ID = {"train": 0, "val": 1, "test": 2}


def find_population_dir() -> Path:
    dirs = sorted(glob.glob(str(OUT / "population_*")))
    assert len(dirs) == 1, f"expected exactly one population dir, found {dirs}"
    return Path(dirs[0])


def load_population(pop_dir: Path | None = None) -> dict:
    pop_dir = pop_dir or find_population_dir()
    d = dict(np.load(pop_dir / "population.npz"))
    d["dir"] = pop_dir
    return d


def dataset_dir(pop_dir: Path, data_cfg: dict) -> Path:
    return OUT / f"datasets_{config_hash({'pop': pop_dir.name, **data_cfg})}"


DATA_CFG = {"code": CODE_VERSION, "stream_seed": 7, "streams": STREAMS, "stream_len": STREAM_LEN}


def generate_datasets(pop: dict, data_cfg: dict = DATA_CFG) -> Path:
    t0 = time.time()
    tree = get_tree(); tab = get_token_table()
    out = dataset_dir(pop["dir"], data_cfg)
    out.mkdir(parents=True, exist_ok=True)
    bp0 = pop["blueprint0"]
    for split, sid in SPLIT_ID.items():
        opp_ids = np.flatnonzero(pop["split"] == sid)
        S = data_cfg["streams"][split]
        terms = np.zeros((len(opp_ids), S, data_cfg["stream_len"]), dtype=np.int16)
        for j, k in enumerate(opp_ids):
            terms[j] = simulate_streams(tree, bp0, pop["policies"][k], S, data_cfg["stream_len"],
                                        data_cfg["stream_seed"], sid, int(k))
        obs = tab.obs_type_of_terminal[terms].astype(np.int16)
        np.savez_compressed(out / f"streams_{split}.npz", opp_ids=opp_ids, terminals=terms, obs_types=obs)
        print(f"{split}: {len(opp_ids)} opponents x {S} streams x {data_cfg['stream_len']} hands "
              f"({time.time()-t0:.0f}s)")
    save_json({"config": data_cfg, "population": pop["dir"].name, "env": environment_info(),
               "n_obs_types": tab.n_types, "runtime_s": time.time() - t0}, out / "meta.json")
    return out


def load_split(ds_dir: Path, split: str) -> dict:
    d = dict(np.load(ds_dir / f"streams_{split}.npz"))
    return d


def find_dataset_dir(tag: str = "") -> Path:
    dirs = [d for d in sorted(glob.glob(str(OUT / "datasets_*"))) if (d.endswith("_" + tag) if tag else "_revealed" not in d)]
    assert len(dirs) == 1, f"expected exactly one dataset dir for tag '{tag}', found {dirs}"
    return Path(dirs[0])


def make_revealed_datasets() -> Path:
    """T2 censoring toggle: same streams (terminal indices), tokens with the opponent card revealed every hand."""
    from .tokenizer import get_token_table
    src = find_dataset_dir(); tab = get_token_table(reveal_all=True)
    out = Path(str(src) + "_revealed"); out.mkdir(exist_ok=True)
    for split in SPLIT_ID:
        d = load_split(src, split)
        obs = tab.obs_type_of_terminal[d["terminals"].astype(np.int64)].astype(np.int16)
        np.savez_compressed(out / f"streams_{split}.npz", opp_ids=d["opp_ids"], terminals=d["terminals"], obs_types=obs)
    save_json({"source": src.name, "variant": "revealed", "n_obs_types": tab.n_types}, out / "meta.json")
    return out


if __name__ == "__main__":
    pop = load_population()
    print("written to", generate_datasets(pop))
