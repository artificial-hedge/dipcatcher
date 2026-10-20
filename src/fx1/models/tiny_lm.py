"""Tiny byte-level transformer definition (torch twin of ``local_engine._TinyLM``)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor as TorchTensor
    from torch.nn import Module as TorchModule

DEFAULT_MLP_RATIO = 4


def build_tiny_lm(vocab: int, dim: int, ctx: int, n_heads: int) -> TorchModule:
    """The torch twin of ``fx1.serve.local_engine._TinyLM`` — identical math."""
    import torch
    from torch import nn

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ln1 = nn.LayerNorm(dim)
            self.attn = nn.ModuleDict({"qkv": nn.Linear(dim, 3 * dim), "proj": nn.Linear(dim, dim)})
            self.ln2 = nn.LayerNorm(dim)
            self.mlp = nn.ModuleDict(
                {
                    "fc1": nn.Linear(dim, DEFAULT_MLP_RATIO * dim),
                    "fc2": nn.Linear(DEFAULT_MLP_RATIO * dim, dim),
                }
            )

        def forward(self, x: TorchTensor) -> TorchTensor:
            n = x.shape[1]
            dh = dim // n_heads
            a = self.ln1(x)
            qkv = self.attn["qkv"](a)
            q, k, v = qkv.split(dim, dim=-1)
            qh = q.reshape(1, n, n_heads, dh).transpose(1, 2)
            kh = k.reshape(1, n, n_heads, dh).transpose(1, 2)
            vh = v.reshape(1, n, n_heads, dh).transpose(1, 2)
            scores = qh @ kh.transpose(-2, -1) / math.sqrt(dh)
            mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
            attended = (torch.softmax(scores, dim=-1) @ vh).transpose(1, 2).reshape(1, n, dim)
            x = x + self.attn["proj"](attended)
            m = self.ln2(x)
            x = x + self.mlp["fc2"](nn.functional.gelu(self.mlp["fc1"](m), approximate="tanh"))
            return x

    class _TinyLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tok_emb = nn.Embedding(vocab, dim)
            self.pos_emb = nn.Embedding(ctx, dim)
            self.blocks = nn.ModuleList([_Block()])
            self.ln_f = nn.LayerNorm(dim)
            self.lm_head = nn.Linear(dim, vocab, bias=False)

        def forward(self, ids: TorchTensor) -> TorchTensor:
            x = self.tok_emb(ids) + self.pos_emb.weight[: ids.shape[1]]
            for blk in self.blocks:
                x = blk(x)
            return self.lm_head(self.ln_f(x))

    return _TinyLM()
