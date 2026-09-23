"""SPO+ surrogate loss (Elmachtoub & Grigas) for the epsilon-safe LP (maximization form).

    l(g_hat, g) = max_{x in S_eps} (2 g_hat - g)^T x  -  2 g_hat^T x*(g)  +  g^T x*(g)
    d l / d g_hat = 2 x*(2 g_hat - g) - 2 x*(g)

Convex in g_hat, non-negative, zero at g_hat = g; upper-bounds the safe-response regret."""
from __future__ import annotations

import numpy as np
import torch


class SPOPlus:
    def __init__(self, solver, eps: float):
        self.L = solver; self.eps = eps
        self.n_lp = 0

    def xstar(self, g):
        ok, pol, x = self.L.solve_safe(np.asarray(g, dtype=np.float64), self.eps)
        self.n_lp += 1
        return x

    def loss_and_grad(self, g_hat, g, x_g):
        """Returns (loss, grad wrt g_hat) for one example; x_g = x*(g) (cached)."""
        w = 2.0 * g_hat - g
        x_w = self.xstar(w)
        loss = w @ x_w - 2.0 * g_hat @ x_g + g @ x_g
        return float(loss), 2.0 * x_w - 2.0 * x_g

    def torch_function(self):
        spo = self

        class SPOPlusFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, g_hat, g_true, x_g):
                gh = g_hat.detach().cpu().numpy().astype(np.float64); gt = g_true.detach().cpu().numpy().astype(np.float64)
                xg = x_g.detach().cpu().numpy().astype(np.float64)
                losses = np.zeros(gh.shape[0]); grads = np.zeros_like(gh)
                for i in range(gh.shape[0]):
                    losses[i], grads[i] = spo.loss_and_grad(gh[i], gt[i], xg[i])
                ctx.save_for_backward(torch.as_tensor(grads, dtype=g_hat.dtype))
                return torch.as_tensor(losses, dtype=g_hat.dtype)

            @staticmethod
            def backward(ctx, grad_out):
                (grads,) = ctx.saved_tensors
                return grads * grad_out[:, None], None, None

        return SPOPlusFn
