"""Phase 13: train one neural model (reconstruction or decision objective) on the train split.

Each step samples N from the observation budgets, a batch of (train opponent, stream) pairs,
uses the first N completed hands of that stream, and regresses the target (true rank-level
policy q for RECON, true g(q) for DECISION).  Validation opponents select the checkpoint.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .common import OUT, N_BUDGETS, LATENT_DIM, save_json, environment_info, config_hash, CODE_VERSION
from .game.leduc_tree import get_tree
from .game.sequence_form import get_sequence_form
from .game.symmetry import get_symmetry
from .game.policy_utils import RankPolicyToG, rank_infoset_features
from .data.tokenizer import get_token_table
from .data.datasets import load_population, find_dataset_dir, load_split
from .models.encoder import OpponentEncoder
from .models.heads import ReconstructionHead, DecisionHead
from .models.torch_g import TorchRankPolicyToG

DEFAULT_CFG = dict(
    d_model=128, z_dim=LATENT_DIM, n_layers=2, n_heads=4, dim_ff=256, dropout=0.1,
    lr=3e-4, weight_decay=0.01, grad_clip=1.0, batch=32, steps=6000, warmup=200, min_lr_frac=0.1,
    val_every=250, g_std_rel_threshold=1e-3, recon_hidden=256, decision_hidden=512, threads=2,
)


def build_model(method, cfg, tab, sym, tree, g_stats):
    enc = OpponentEncoder(tab.cat, tab.num, tab.length, d_model=cfg["d_model"], z_dim=cfg["z_dim"],
                          n_layers=cfg["n_layers"], n_heads=cfg["n_heads"], dim_ff=cfg["dim_ff"],
                          dropout=cfg["dropout"])
    if method == "recon":
        head = ReconstructionHead(rank_infoset_features(tree, sym, 1), sym.rank_legal_mask[1],
                                  z_dim=cfg["z_dim"], d_hidden=cfg["recon_hidden"])
    else:
        head = DecisionHead(g_stats["mean"], g_stats["std"], g_stats["valid"], z_dim=cfg["z_dim"],
                            d_hidden=cfg["decision_hidden"])
    return enc, head


def g_stats_from_train(G_train, rel):
    mean, std = G_train.mean(0), G_train.std(0)
    valid = std > rel * std.max()
    return {"mean": mean, "std": np.where(valid, std, 1.0), "valid": valid, "n_valid": int(valid.sum())}


class Trainer:
    def __init__(self, method, seed, cfg, out_dir):
        self.method, self.seed, self.cfg, self.out = method, seed, cfg, Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(seed); np.random.seed(seed)
        torch.set_num_threads(cfg["threads"])
        self.rng = np.random.default_rng([seed, 11])
        self.tree = get_tree(); self.sf = get_sequence_form(); self.sym = get_symmetry(); self.tab = get_token_table()
        self.pop = load_population(); ds = find_dataset_dir()
        self.train = load_split(ds, "train"); self.val = load_split(ds, "val")
        self.G = self.pop["G"]; self.Q = self.pop["rank_policies"]
        self.g_stats = g_stats_from_train(self.G[self.train["opp_ids"]], cfg["g_std_rel_threshold"])
        self.enc, self.head = build_model(method, cfg, self.tab, self.sym, self.tree, self.g_stats)
        self.r2g = TorchRankPolicyToG(RankPolicyToG(self.tree, self.sf, self.sym))
        params = list(self.enc.parameters()) + list(self.head.parameters())
        self.opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        self.n_params = sum(p.numel() for p in params)
        self.Gt = torch.as_tensor(self.G, dtype=torch.float32)
        self.Qt = torch.as_tensor(self.Q, dtype=torch.float32)
        self.g_mean = torch.as_tensor(self.g_stats["mean"], dtype=torch.float32)
        self.g_std = torch.as_tensor(self.g_stats["std"], dtype=torch.float32)
        self.g_valid = torch.as_tensor(self.g_stats["valid"])
        self.log = []

    def lr_at(self, step):
        cfg = self.cfg
        if step < cfg["warmup"]:
            return cfg["lr"] * (step + 1) / cfg["warmup"]
        prog = (step - cfg["warmup"]) / max(1, cfg["steps"] - cfg["warmup"])
        return cfg["lr"] * (cfg["min_lr_frac"] + (1 - cfg["min_lr_frac"]) * 0.5 * (1 + np.cos(np.pi * prog)))

    def targets(self, opp_ids):
        return self.Qt[opp_ids] if self.method == "recon" else self.Gt[opp_ids]

    def loss_fn(self, z, opp_ids):
        return self.head.loss(z, self.targets(opp_ids))

    def g_nmse(self, z, opp_ids):
        """Normalized g error of the model's implied g_hat (both methods)."""
        with torch.no_grad():
            g_hat = self.r2g(self.head(z)) if self.method == "recon" else self.head(z)
            err = ((g_hat - self.Gt[opp_ids]) / self.g_std) ** 2
            nmse = err[:, self.g_valid].mean().item()
            raw = ((g_hat - self.Gt[opp_ids]) ** 2).sum(1).sqrt().mean().item()
        return nmse, raw

    def sample_batch(self):
        cfg = self.cfg
        N = int(self.rng.choice(N_BUDGETS))
        n_opp, n_streams = self.train["obs_types"].shape[:2]
        idx = self.rng.integers(0, n_opp, cfg["batch"]); st = self.rng.integers(0, n_streams, cfg["batch"])
        x = torch.as_tensor(self.train["obs_types"][idx, st, :N].astype(np.int64))
        return x, torch.as_tensor(self.train["opp_ids"][idx]), N

    def validate(self):
        self.enc.eval(); self.head.eval()
        obs = self.val["obs_types"]; opp = self.val["opp_ids"]
        n_opp, n_streams, _ = obs.shape
        per_N = {}
        with torch.no_grad():
            for N in N_BUDGETS:
                losses, nmses, raws = [], [], []
                for s in range(n_streams):
                    for start in range(0, n_opp, 50):
                        sl = slice(start, start + 50)
                        x = torch.as_tensor(obs[sl, s, :N].astype(np.int64)); ids = torch.as_tensor(opp[sl])
                        z = self.enc(x)
                        losses.append(self.loss_fn(z, ids).item() * len(ids))
                        nm, rw = self.g_nmse(z, ids); nmses.append(nm * len(ids)); raws.append(rw * len(ids))
                tot = n_opp * n_streams
                per_N[N] = {"loss": sum(losses) / tot, "g_nmse": sum(nmses) / tot, "g_raw": sum(raws) / tot}
        self.enc.train(); self.head.train()
        mean_loss = float(np.mean([v["loss"] for v in per_N.values()]))
        return mean_loss, per_N

    def run(self):
        cfg = self.cfg; t0 = time.time(); best = np.inf; best_step = -1
        save_json({"method": self.method, "seed": self.seed, "cfg": cfg, "n_params": self.n_params,
                   "g_stats_n_valid": self.g_stats["n_valid"], "env": environment_info()}, self.out / "config.json")
        for step in range(cfg["steps"]):
            for grp in self.opt.param_groups:
                grp["lr"] = self.lr_at(step)
            x, ids, N = self.sample_batch()
            z = self.enc(x)
            loss = self.loss_fn(z, ids)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(list(self.enc.parameters()) + list(self.head.parameters()), cfg["grad_clip"])
            self.opt.step()
            rec = {"step": step, "N": N, "loss": loss.item(), "grad_norm": float(gn), "lr": self.lr_at(step), "t": time.time() - t0}
            if (step + 1) % cfg["val_every"] == 0 or step == cfg["steps"] - 1:
                vl, per_N = self.validate()
                rec["val_loss"] = vl; rec["val_per_N"] = per_N
                if vl < best:
                    best, best_step = vl, step
                    torch.save({"enc": self.enc.state_dict(), "head": self.head.state_dict(), "step": step,
                                "val_loss": vl, "cfg": cfg, "method": self.method, "seed": self.seed,
                                "g_stats": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in self.g_stats.items()}},
                               self.out / "best.pt")
                print(f"[{self.method} s{self.seed}] step {step+1} loss {loss.item():.4f} val {vl:.4f} "
                      f"(best {best:.4f} @ {best_step+1}) gN={ {N: round(v['g_nmse'],3) for N, v in per_N.items()} } "
                      f"t={time.time()-t0:.0f}s", flush=True)
            self.log.append(rec)
            if (step + 1) % 50 == 0:
                with open(self.out / "log.jsonl", "w") as f:
                    for r in self.log:
                        f.write(json.dumps(r) + "\n")
        torch.save({"enc": self.enc.state_dict(), "head": self.head.state_dict(), "step": cfg["steps"]}, self.out / "last.pt")
        with open(self.out / "log.jsonl", "w") as f:
            for r in self.log:
                f.write(json.dumps(r) + "\n")
        save_json({"best_val_loss": best, "best_step": best_step, "runtime_s": time.time() - t0}, self.out / "result.json")
        return best


def load_trained(run_dir):
    """Load the best checkpoint of a run; returns (enc, head, ckpt)."""
    ck = torch.load(Path(run_dir) / "best.pt", weights_only=False)
    tree, sym, tab = get_tree(), get_symmetry(), get_token_table()
    gs = {k: np.array(v) if isinstance(v, list) else v for k, v in ck["g_stats"].items()}
    enc, head = build_model(ck["method"], ck["cfg"], tab, sym, tree, gs)
    enc.load_state_dict(ck["enc"]); head.load_state_dict(ck["head"])
    enc.eval(); head.eval()
    return enc, head, ck


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["recon", "decision"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    for k, v in DEFAULT_CFG.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    args = ap.parse_args()
    cfg = {k: getattr(args, k) for k in DEFAULT_CFG}
    out = args.out or (OUT / "runs" / f"{args.method}_s{args.seed}")
    Trainer(args.method, args.seed, cfg, out).run()
