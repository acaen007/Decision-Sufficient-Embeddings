"""Hierarchical Transformer opponent encoder: events -> hand embedding -> history -> z.

Within-hand encoder: structured event tokens (9 categorical fields embedded and summed, 7
numeric fields linearly projected, learned event-position embedding) + [HAND_CLS], 2 pre-LN
Transformer layers, 4 heads, d_model 128, dropout 0.1; HAND_CLS output = hand embedding.

Cross-hand encoder: hand embeddings + learned hand-position embedding + [HISTORY_CLS], 2
pre-LN Transformer layers, 4 heads, bidirectional attention over the N completed hands with
a padding mask; HISTORY_CLS output projected to z in R^128.

Computational note: a batch contains many identical hands (there are only 792 distinct
player-0 observation types in Leduc).  The hand encoder is therefore evaluated once per
distinct type per step (`hand_table`) and gathered; this is the same function as encoding each
hand separately (dropout masks are shared between identical hands).  `encode_hands_direct`
provides the per-hand path for tests.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..data.tokenizer import CAT_VOCAB, NUM_FIELDS, MAX_EVENTS


class EventEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.cat_emb = nn.ModuleList([nn.Embedding(v, d_model) for v in CAT_VOCAB])
        self.num_proj = nn.Linear(len(NUM_FIELDS), d_model)
        self.pos = nn.Embedding(MAX_EVENTS + 1, d_model)

    def forward(self, cat: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        """cat (B, E, 9) long, num (B, E, 7) float -> (B, E, d)."""
        h = self.num_proj(num)
        for i, emb in enumerate(self.cat_emb):
            h = h + emb(cat[..., i])
        return h


class PreLNLayer(nn.Module):
    """Pre-LayerNorm Transformer encoder layer using fused scaled-dot-product attention.

    Dropout (p) is applied on the attention-output and feed-forward residual branches and inside
    the feed-forward block; attention-probability dropout is 0 (on CPU it dominated step time).
    """

    def __init__(self, d_model, n_heads, dim_ff, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads; self.dh = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model); self.ln2 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model); self.out = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim_ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_pad=None):
        B, L, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        attn_mask = None
        if key_pad is not None and bool(key_pad.any()):
            attn_mask = (~key_pad)[:, None, None, :]            # True = may attend
        a = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + self.drop(self.out(a.transpose(1, 2).reshape(B, L, D)))
        return x + self.drop(self.ff(self.ln2(x)))


class Encoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, dim_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([PreLNLayer(d_model, n_heads, dim_ff, dropout) for _ in range(n_layers)])

    def forward(self, x, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask)
        return x


def make_encoder(d_model, n_layers, n_heads, dim_ff, dropout):
    return Encoder(d_model, n_layers, n_heads, dim_ff, dropout)


class HandEncoder(nn.Module):
    def __init__(self, d_model=128, n_layers=2, n_heads=4, dim_ff=256, dropout=0.1):
        super().__init__()
        self.embed = EventEmbedding(d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)
        self.enc = make_encoder(d_model, n_layers, n_heads, dim_ff, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, cat, num, lengths):
        """cat (B, E, 9), num (B, E, 7), lengths (B,) -> (B, d) hand embeddings."""
        B, E, _ = cat.shape
        h = self.embed(cat, num)
        pos = torch.arange(E + 1, device=cat.device)
        h = torch.cat([self.cls.expand(B, 1, -1), h], dim=1) + self.embed.pos(pos)[None]
        key_pad = torch.arange(E + 1, device=cat.device)[None] > lengths[:, None]   # CLS at 0 never padded
        out = self.enc(h, src_key_padding_mask=key_pad)
        return self.norm(out[:, 0])


class HistoryEncoder(nn.Module):
    def __init__(self, d_model=128, n_layers=2, n_heads=4, dim_ff=256, dropout=0.1, max_hands=500, z_dim=128):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)
        self.pos = nn.Embedding(max_hands + 1, d_model)
        self.enc = make_encoder(d_model, n_layers, n_heads, dim_ff, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, z_dim)

    def forward(self, hands, mask, cls_extra=None):
        """hands (B, N, d), mask (B, N) bool (True = real hand), cls_extra (B, d) added to the CLS token -> z."""
        B, N, _ = hands.shape
        pos = torch.arange(N + 1, device=hands.device)
        cls = self.cls.expand(B, 1, -1) if cls_extra is None else self.cls.expand(B, 1, -1) + cls_extra[:, None, :]
        h = torch.cat([cls, hands], dim=1) + self.pos(pos)[None]
        key_pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=hands.device), ~mask], dim=1)
        out = self.enc(h, src_key_padding_mask=key_pad)
        return self.proj(self.norm(out[:, 0]))


class OpponentEncoder(nn.Module):
    def __init__(self, type_cat, type_num, type_len, d_model=128, z_dim=128, n_layers=2, n_heads=4,
                 dim_ff=256, dropout=0.1, max_hands=500, extra_dim=0):
        super().__init__()
        self.extra = nn.Linear(extra_dim, d_model) if extra_dim else None
        self.register_buffer("type_cat", torch.tensor(np.asarray(type_cat), dtype=torch.long))
        self.register_buffer("type_num", torch.tensor(np.asarray(type_num), dtype=torch.float32))
        self.register_buffer("type_len", torch.tensor(np.asarray(type_len), dtype=torch.long))
        self.hand = HandEncoder(d_model, n_layers, n_heads, dim_ff, dropout)
        self.history = HistoryEncoder(d_model, n_layers, n_heads, dim_ff, dropout, max_hands, z_dim)

    def hand_table(self):
        return self.hand(self.type_cat, self.type_num, self.type_len)           # (T, d)

    def forward(self, type_ids, mask=None, extra=None):
        """type_ids (B, N) long observation-type ids; mask (B, N) bool; extra (B, extra_dim) -> z (B, z_dim)."""
        if mask is None:
            mask = torch.ones_like(type_ids, dtype=torch.bool)
        table = self.hand_table()
        hands = table[type_ids.clamp(min=0)] * mask[..., None]
        return self.history(hands, mask, self.extra(extra) if (self.extra is not None and extra is not None) else None)

    def encode_hands_direct(self, type_ids, mask=None):
        """Per-hand path (no deduplication) used by tests."""
        if mask is None:
            mask = torch.ones_like(type_ids, dtype=torch.bool)
        B, N = type_ids.shape
        ids = type_ids.clamp(min=0).reshape(-1)
        h = self.hand(self.type_cat[ids], self.type_num[ids], self.type_len[ids]).reshape(B, N, -1)
        return self.history(h * mask[..., None], mask)
