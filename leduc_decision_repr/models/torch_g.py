"""Differentiable rank-policy -> realization plan -> g map in torch (mirrors RankPolicyToG)."""
import torch
import torch.nn as nn


class TorchRankPolicyToG(nn.Module):
    def __init__(self, r2g):
        super().__init__()
        self.register_buffer("idx", torch.as_tensor(r2g.idx, dtype=torch.long))
        self.register_buffer("valid", torch.as_tensor(r2g.valid, dtype=torch.bool))
        self.register_buffer("A_T", torch.as_tensor(r2g.A_T, dtype=torch.float32))

    def realization(self, rank_policies):
        Q = rank_policies.reshape(rank_policies.shape[0], -1)
        fac = torch.where(self.valid[None], Q[:, self.idx], torch.ones_like(Q[:, self.idx]))
        return fac.prod(dim=2)

    def forward(self, rank_policies):
        return self.realization(rank_policies) @ self.A_T
