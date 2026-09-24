"""Decision-aware fine-tuning from converged checkpoints (REPORT_LEDUC_FT).

loss = anchor + lam * mean_batch(decision loss), anchor = Jacobian-weighted CE (RECON-JAC) or standardized MSE (DEC).
decision loss: "spo" (SPO+, exact LP for all 32 examples of every step, cached x*(g) target, eps = 0.10) or
"pfy" (perturbed Fenchel-Young, K draws); lam = 0 -> control (anchor only, no LPs).
Checkpoint selection: exact deployed regret on a fixed validation set (150 opponents x N {20,100,500} x stream 0),
evaluated at steps 250..1500 (step 0 = reference only); every validation strategy is audited with OpenSpiel."""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .common import OUT, N_BUDGETS, save_json
from .train import Trainer, DEFAULT_CFG

EPS = 0.10; EPS_IDX = 2; VAL_N = [20, 100, 500]


def make_pfy_fn(L, eps, sigma, K, rng, counter):
    """Perturbed Fenchel-Young (maximization): grad = mean_k x*(g_hat + sigma Z_k) - x*(g)."""
    class PFYFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, g_hat, g_true, x_g):
            gh = g_hat.detach().cpu().numpy().astype(np.float64); xg = x_g.detach().cpu().numpy().astype(np.float64)
            losses = np.zeros(gh.shape[0]); grads = np.zeros_like(gh)
            for i in range(gh.shape[0]):
                xs = np.zeros(gh.shape[1]); val = 0.0
                for _ in range(K):
                    w = gh[i] + sigma * rng.standard_normal(gh.shape[1])
                    x = L.solve_safe(w, eps)[2]; counter[0] += 1
                    xs += x / K; val += (w @ x) / K
                losses[i] = val - gh[i] @ xg[i]; grads[i] = xs - xg[i]
            ctx.save_for_backward(torch.as_tensor(grads, dtype=g_hat.dtype))
            return torch.as_tensor(losses, dtype=g_hat.dtype)

        @staticmethod
        def backward(ctx, grad_out):
            (grads,) = ctx.saved_tensors
            return grads * grad_out[:, None], None, None
    return PFYFn


class FineTuner:
    def __init__(self, base_dir, out_dir, lam, seed, loss="spo", steps=1500, lr=1e-4, warmup=100, min_lr=1e-5,
                 val_every=250, sigma_mult=None, K=4):
        self.base_dir, self.out = Path(base_dir), Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
        self.ck = torch.load(self.base_dir / "best.pt", weights_only=False)
        cfg = {**DEFAULT_CFG, **self.ck["cfg"]}; cfg["threads"] = 1
        self.method = self.ck["method"]; self.lam = float(lam); self.loss_kind = loss if self.lam > 0 else "none"
        self.steps, self.lr, self.warmup, self.min_lr, self.val_every = steps, lr, warmup, min_lr, val_every
        self.T = T = Trainer(self.method, seed, cfg, self.out)            # data, model skeleton, r2g, Jacobian weights
        T.enc.load_state_dict(self.ck["enc"]); T.head.load_state_dict(self.ck["head"])
        T.rng = np.random.default_rng([seed, 2026])                        # fine-tuning batches: same for control and SPO+ arms
        torch.manual_seed(10_000 + seed)
        self.params = list(T.enc.parameters()) + list(T.head.parameters())
        self.opt = torch.optim.AdamW(self.params, lr=lr, weight_decay=cfg["weight_decay"])
        from .game.safe_lp import get_solver, OpenSpielAuditor
        self.L = get_solver(); self.aud = OpenSpielAuditor(T.sf, self.L.v_star)
        self.Xstar = torch.as_tensor(np.load(OUT / "weights_v3" / "xstar_train_eps0.1.npy"), dtype=torch.float32)
        self.n_lp = [0]
        if self.loss_kind == "spo":
            from .game.spo_plus import SPOPlus
            self.spo = SPOPlus(self.L, EPS); self.dfn = self.spo.torch_function()
        elif self.loss_kind == "pfy":
            tr = T.train["opp_ids"]; Gtr = T.G[tr]; sigma0 = float(np.sqrt(((Gtr - Gtr.mean(0)) ** 2).mean()))
            self.sigma = float(sigma_mult) * sigma0; self.sigma0 = sigma0
            self.dfn = make_pfy_fn(self.L, EPS, self.sigma, K, np.random.default_rng([seed, 99]), self.n_lp)
        self.meta = {"base_dir": str(self.base_dir), "base_step": int(self.ck["step"]), "method": self.method, "seed": seed, "lam": self.lam,
                     "loss": self.loss_kind, "steps": steps, "lr": lr, "warmup": warmup, "min_lr": min_lr, "val_every": val_every,
                     "sigma_mult": sigma_mult, "sigma": getattr(self, "sigma", None), "sigma0": getattr(self, "sigma0", None), "K": K}

    def lr_at(self, s):
        if s < self.warmup:
            return self.lr * (s + 1) / self.warmup
        prog = (s - self.warmup) / max(1, self.steps - self.warmup)
        return self.min_lr + (self.lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * prog))

    def g_hat_raw(self, out):
        return self.T.r2g(out) if self.method == "recon" else out

    def forward(self, x, ids, extra):
        T = self.T; z = T.enc(x, extra=extra)
        if self.method == "recon":
            logits = T.head.logits(z)
            logp = torch.where(T.head.mask[None], F.log_softmax(logits, dim=-1), torch.zeros_like(logits))
            ce = -(T.Qt[ids] * logp).sum(-1)
            anchor = (ce * T.recon_w[None]).sum(1).mean() / T.recon_w.sum() if T.recon_w is not None else ce.mean()
            g_hat = T.r2g(F.softmax(logits, dim=-1))
        else:
            anchor = T.head.loss(z, T.Gt[ids]); g_hat = T.head(z)
        return anchor, g_hat

    def validate(self):
        T = self.T; T.enc.eval(); T.head.eval()
        obs = T.val["obs_types"][:, 0]; opp = T.val["opp_ids"]
        R, nm, anc, viol, fails = [], [], [], [], 0
        with torch.no_grad():
            for N in VAL_N:
                x = torch.as_tensor(obs[:, :N].astype(np.int64))
                extra = torch.as_tensor(T.cf.features(obs[:, :N])) if T.cf is not None else None
                anchor, g_hat = self.forward(x, torch.as_tensor(opp), extra); anc.append(float(anchor))
                gh = g_hat.numpy().astype(np.float64)
                err = ((g_hat - T.Gt[opp]) / T.g_std) ** 2; nm.append(float(err[:, T.g_valid].mean()))
                for i, k in enumerate(opp):
                    ok, pol, xd = self.L.solve_safe(gh[i], EPS); fails += (not ok)
                    R.append(float(T.pop["V_oracle"][k, EPS_IDX] - T.G[k] @ xd))
                    viol.append(self.aud.exploitability_of_learner(pol) - EPS)
        T.enc.train(); T.head.train()
        R = np.array(R).reshape(len(VAL_N), -1)
        return {"val_regret": float(R.mean()), "val_regret_by_N": {N: float(R[j].mean()) for j, N in enumerate(VAL_N)},
                "val_hit_rate": float((R < 1e-6).mean()), "val_g_nmse": float(np.mean(nm)), "val_anchor": float(np.mean(anc)),
                "val_audit_max": float(np.max(viol)), "val_audit_viol": int((np.array(viol) > 1e-7).sum()), "val_lp_fail": int(fails)}

    def save(self, name, step, v):
        ck = {k: self.ck[k] for k in ("cfg", "method", "g_stats")}
        ck.update(enc=self.T.enc.state_dict(), head=self.T.head.state_dict(), step=int(step), seed=self.meta["seed"],
                  fixed_norm=None, val=v, ft=self.meta)
        torch.save(ck, self.out / name)

    def run(self):
        T = self.T; t0 = time.time(); log = []
        save_json(self.meta, self.out / "ft_config.json")
        v0 = self.validate(); v0["step"] = 0; log.append(v0)
        print(f"[{self.out.name}] step 0 val_regret {v0['val_regret']:.4f} nmse {v0['val_g_nmse']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        best, best_step = np.inf, None; t_lp = 0.0; t_model = 0.0
        for s in range(self.steps):
            for g in self.opt.param_groups:
                g["lr"] = self.lr_at(s)
            x, ids, N, _ = T.sample_batch()
            tm = time.time()
            anchor, g_hat = self.forward(x, ids, T._extra)
            rec = {"step": s + 1, "N": N, "anchor": anchor.item(), "lr": self.lr_at(s)}
            loss = anchor
            if self.loss_kind != "none":
                tl = time.time(); n0 = self.spo.n_lp if self.loss_kind == "spo" else self.n_lp[0]
                d = self.dfn.apply(g_hat, T.Gt[ids], self.Xstar[ids]).mean()
                dt = time.time() - tl; t_lp += dt
                rec.update(dec_loss=d.item(), lp_time=dt, n_lp=(self.spo.n_lp if self.loss_kind == "spo" else self.n_lp[0]) - n0)
                loss = anchor + self.lam * d
            self.opt.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(self.params, 1.0); self.opt.step()
            t_model += time.time() - tm - rec.get("lp_time", 0.0)
            rec.update(loss=loss.item(), grad_norm=float(gn), t=time.time() - t0)
            if not np.isfinite(rec["loss"]):
                print(f"[{self.out.name}] non-finite loss at step {s+1}; stopping", flush=True); rec["diverged"] = True; log.append(rec); break
            if (s + 1) % self.val_every == 0:
                v = self.validate(); rec.update(v)
                if v["val_regret"] < best - 1e-12:
                    best, best_step = v["val_regret"], s + 1; self.save("best.pt", s + 1, v)
                print(f"[{self.out.name}] step {s+1} anchor {rec['anchor']:.4f} dec {rec.get('dec_loss', float('nan')):.4f} val_regret {v['val_regret']:.4f} "
                      f"(best {best:.4f} @ {best_step}) nmse {v['val_g_nmse']:.3f} hit {v['val_hit_rate']:.3f} t={time.time()-t0:.0f}s", flush=True)
            log.append(rec)
            if (s + 1) % 50 == 0:
                with open(self.out / "log.jsonl", "w") as f:
                    f.write("\n".join(json.dumps(r) for r in log) + "\n")
        self.save("last.pt", s + 1, None)
        with open(self.out / "log.jsonl", "w") as f:
            f.write("\n".join(json.dumps(r) for r in log) + "\n")
        vals = [r for r in log if "val_regret" in r]
        res = {"best_step": best_step, "best_val_regret": best, "step0_val_regret": v0["val_regret"], "step0": v0,
               "val_curve": [(r["step"], r["val_regret"], r["val_g_nmse"], r["val_hit_rate"]) for r in vals],
               "runtime_s": time.time() - t0, "lp_time_total_s": t_lp, "model_time_total_s": t_model,
               "lp_time_per_step_s": t_lp / max(1, self.steps), "model_time_per_step_s": t_model / max(1, self.steps),
               "n_lp_train": int(self.spo.n_lp if self.loss_kind == "spo" else self.n_lp[0]),
               "val_audit_max": float(max(r["val_audit_max"] for r in vals)), "val_audit_viol": int(sum(r["val_audit_viol"] for r in vals)),
               "val_lp_fail": int(sum(r["val_lp_fail"] for r in vals)), "diverged": any(r.get("diverged") for r in log), **self.meta}
        save_json(res, self.out / "result.json")
        print(f"[{self.out.name}] done: best val regret {best:.4f} @ {best_step} (step0 {v0['val_regret']:.4f}); {res['runtime_s']/60:.1f} min", flush=True)
        return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--lam", type=float, default=0.0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss", default="spo", choices=["spo", "pfy"]); ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--sigma_mult", type=float, default=None); ap.add_argument("--K", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(1)
    FineTuner(a.base, a.out, a.lam, a.seed, a.loss, steps=a.steps, sigma_mult=a.sigma_mult, K=a.K).run()
