"""Prediction heads: behavioral reconstruction (z + infoset -> masked softmax) and decision (z -> g)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionHead(nn.Module):
    """z + representation of an opponent (rank) infoset -> legal-action distribution."""

    def __init__(self, infoset_features, legal_mask, z_dim=128, d_hidden=256, emb_dim=64):
        super().__init__()
        n_inf, n_feat = infoset_features.shape
        self.register_buffer("feat", torch.as_tensor(infoset_features, dtype=torch.float32))
        self.register_buffer("mask", torch.as_tensor(legal_mask, dtype=torch.bool))
        self.inf_emb = nn.Embedding(n_inf, emb_dim)
        self.inf_proj = nn.Linear(n_feat + emb_dim, d_hidden)
        self.z_proj = nn.Linear(z_dim, d_hidden)
        self.mlp = nn.Sequential(nn.GELU(), nn.Linear(d_hidden, d_hidden), nn.GELU(), nn.Linear(d_hidden, 3))
        self.n_inf = n_inf

    def logits(self, z):
        inf = torch.cat([self.feat, self.inf_emb.weight], dim=1)              # (I, F+E)
        h = self.z_proj(z)[:, None, :] + self.inf_proj(inf)[None]              # (B, I, H)
        out = self.mlp(h)                                                      # (B, I, 3)
        return out.masked_fill(~self.mask[None], float("-inf"))

    def forward(self, z):
        return F.softmax(self.logits(z), dim=-1)                                # (B, I, 3), zeros at illegal

    def loss(self, z, q_true, infoset_weights=None):
        """Soft-target cross-entropy averaged over batch and infosets (optionally infoset-weighted)."""
        logp = F.log_softmax(self.logits(z), dim=-1)
        logp = torch.where(self.mask[None], logp, torch.zeros_like(logp))
        ce = -(q_true * logp).sum(-1)                                          # (B, I)
        if infoset_weights is not None and infoset_weights.dim() == 2:           # per-sample (opponent-specific) weights
            return ((ce * infoset_weights).sum(1) / infoset_weights.sum(1)).mean()
        if infoset_weights is not None:
            return (ce * infoset_weights[None]).sum(1).mean() / infoset_weights.sum()
        return ce.mean()


class DecisionHead(nn.Module):
    """z -> standardized decision vector g_hat (dims with ~zero training variance are fixed)."""

    def __init__(self, g_mean, g_std, valid, z_dim=128, d_hidden=512):
        super().__init__()
        self.register_buffer("g_mean", torch.as_tensor(g_mean, dtype=torch.float32))
        self.register_buffer("g_std", torch.as_tensor(g_std, dtype=torch.float32))
        self.register_buffer("valid", torch.as_tensor(valid, dtype=torch.bool))
        self.mlp = nn.Sequential(nn.Linear(z_dim, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_hidden), nn.GELU(),
                                 nn.Linear(d_hidden, len(g_mean)))
        self.fixed_norm = None      # if set, g_hat is rescaled to this L2 norm (closes the tau/scale loophole)

    def forward_standardized(self, z):
        return self.mlp(z)

    def forward(self, z):
        """Unstandardized g_hat; excluded dims are predicted at their training mean."""
        s = self.forward_standardized(z)
        g = self.g_mean + s * self.g_std
        g = torch.where(self.valid[None], g, self.g_mean[None].expand_as(g))
        if self.fixed_norm is not None:
            g = g * (self.fixed_norm / g.norm(dim=1, keepdim=True).clamp(min=1e-8))
        return g

    def loss(self, z, g_true):
        s = self.forward_standardized(z)
        target = (g_true - self.g_mean) / torch.where(self.valid, self.g_std, torch.ones_like(self.g_std))
        err = ((s - target) ** 2)[:, self.valid]
        return err.mean()


class ZCodeHead(nn.Module):
    """z -> compact decision code (d numbers) -> frozen policy decoder of a regret autoencoder -> q_hat
    (REPORT_LEDUC_ZCODE.md).  The code is predicted standardized; code_mu / code_sd undo the standardization."""

    def __init__(self, legal_mask, z_dim=128, d_code=8, d_hidden=834):
        super().__init__()
        self.register_buffer("mask", torch.as_tensor(legal_mask, dtype=torch.bool))
        self.register_buffer("code_mu", torch.zeros(d_code)); self.register_buffer("code_sd", torch.ones(d_code))
        self.map = nn.Sequential(nn.Linear(z_dim, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_code))
        self.dec = nn.Sequential(nn.Linear(d_code, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 432))
        for prm in self.dec.parameters():
            prm.requires_grad_(False)

    def code(self, z):
        return self.map(z)

    def logits(self, z):
        c = self.code(z) * self.code_sd + self.code_mu
        return self.dec(c).reshape(-1, 144, 3).masked_fill(~self.mask[None], float("-inf"))

    def forward(self, z):
        return F.softmax(self.logits(z), dim=-1)
