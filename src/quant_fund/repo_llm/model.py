"""Decoder-only transformer over the repository byte stream."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from quant_fund.repo_llm.config import RepoLMConfig
from quant_fund.repo_llm.fold import fold_lanes, lanes_to_unit_interval


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * scale


class CausalSelfAttention(nn.Module):
    def __init__(self, config: RepoLMConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, channels = x.shape
        qkv = self.qkv(x).view(batch, steps, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        mixed = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        mixed = mixed.transpose(1, 2).contiguous().view(batch, steps, channels)
        return self.proj(mixed)


class MLP(nn.Module):
    def __init__(self, config: RepoLMConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, config: RepoLMConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class RepoLM(nn.Module):
    """Causal byte language model. Embeddings are tied to the output head."""

    def __init__(self, config: RepoLMConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.seq_len, config.d_model)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.head.weight = self.tok_emb.weight

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _batch, steps = idx.shape
        if steps > self.config.seq_len:
            raise ValueError(f"sequence length {steps} exceeds seq_len {self.config.seq_len}")
        positions = torch.arange(steps, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new: int, temperature: float = 1.0) -> torch.Tensor:
        if max_new < 1:
            raise ValueError("max_new must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.eval()
        for _ in range(max_new):
            cond = idx[:, -self.config.seq_len :]
            logits, _ = self(cond)
            probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def unique_parameters(model: RepoLM) -> list[tuple[str, nn.Parameter]]:
    """Sorted unique parameters. Tied embedding and head weights are one slot."""
    seen: set[int] = set()
    chosen: list[tuple[str, nn.Parameter]] = []
    for name, param in sorted(model.named_parameters(), key=lambda item: item[0]):
        if id(param) in seen:
            continue
        seen.add(id(param))
        chosen.append((name, param))
    return chosen


def fold_corpus_into_model(model: RepoLM, tokens, seed: int) -> None:
    """Replace every parameter with a rescaled fold of ``tokens``.

    ``tokens`` is a 1-D uint8 array or memmap. The fold reads it in lane-sized
    chunks, so a multi-gigabyte corpus is not copied into a second buffer.
    Norm scales stay near 1. Other matrices are standardized to the GPT-2
    residual scale so training starts from a finite language-model
    initialization that still depends on every corpus byte.
    """
    named = unique_parameters(model)
    total = sum(param.numel() for _name, param in named)
    if isinstance(tokens, np.ndarray) and tokens.dtype == np.uint8 and tokens.ndim == 1:
        byte_view = tokens
    else:
        byte_view = np.asarray(tokens, dtype=np.uint8).reshape(-1)
    lanes = fold_lanes(byte_view, total, seed)
    unit = lanes_to_unit_interval(lanes)
    cursor = 0
    residual = 0.02 / math.sqrt(2.0 * model.config.n_layers)
    with torch.no_grad():
        for name, param in named:
            count = param.numel()
            sl = unit[cursor : cursor + count]
            cursor += count
            values = torch.from_numpy(np.ascontiguousarray(sl, dtype=np.float32)).view(param.shape)
            if "norm.weight" in name:
                param.copy_(1.0 + 0.02 * (values - 0.5))
                continue
            centered = values - values.mean()
            std = centered.std(unbiased=False)
            scale = residual if name.endswith("proj.weight") else 0.02
            if float(std) < 1e-8:
                param.zero_()
            else:
                param.copy_(centered / std * scale)
    if cursor != total:
        raise RuntimeError("fold did not cover every parameter")
