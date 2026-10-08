"""RoPE attention (Su et al. 2021) — rotary position embeddings on (SYNTHETIC)
query/key blocks vs learned absolute embeddings on the induction-head
task, evaluated at TRAIN length and at 2x length (extrapolation).
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._lm_synth import VOCAB, recall_batch


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("rope_attn requires torch (pip install -e .[nn])") from exc
    return torch


def _rotary(q, k):
    torch = _torch()
    B, T, D = q.shape
    pos = torch.arange(T, dtype=torch.float32)
    freqs = 1.0 / (10000 ** (torch.arange(0, D, 2).float() / D))
    ang = torch.outer(pos, freqs)
    cos, sin = ang.cos()[None, :, :, None], ang.sin()[None, :, :, None]

    def rot(x):
        x1, x2 = x[:, :, 0::2], x[:, :, 1::2]
        return torch.stack(
            [x1 * cos[..., 0] - x2 * sin[..., 0], x1 * sin[..., 0] + x2 * cos[..., 0]], -1
        ).reshape(B, T, D)

    return rot(q), rot(k)


def _train_and_eval(seed: int, rope: bool, iters: int = 800) -> tuple[float, float]:
    torch = _torch()
    torch.manual_seed(seed)
    D = 24
    emb = torch.nn.Embedding(VOCAB, D)
    pos = torch.nn.Embedding(32, D)
    qk = torch.nn.Linear(D, 2 * D, bias=False)
    v = torch.nn.Linear(D, D, bias=False)
    head = torch.nn.Linear(D, VOCAB)
    opt = torch.optim.Adam(
        list(emb.parameters())
        + list(pos.parameters())
        + list(qk.parameters())
        + list(v.parameters())
        + list(head.parameters()),
        lr=0.01,
    )

    def forward(x, T):
        h = emb(x)
        if not rope:
            h = h + pos.weight[:T]
        q, k = qk(h).chunk(2, -1)
        if rope:
            q, k = _rotary(q, k)
        att = torch.softmax(q @ k.transpose(1, 2) / D**0.5, -1)
        return head(att @ v(h))[:, -1]

    for _ in range(iters):
        x, y = recall_batch(int(np.random.default_rng().integers(0, 1 << 30)))
        xt, yt = torch.tensor(x), torch.tensor(y)
        loss = torch.nn.functional.cross_entropy(forward(xt, xt.shape[1]), yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    x, y = recall_batch(seed + 1, B=256, T=12)
    xt, yt = torch.tensor(x), torch.tensor(y)
    acc = float((forward(xt, 12).argmax(-1) == yt).float().mean())
    x2, y2 = recall_batch(seed + 2, B=256, T=24)
    xt2, yt2 = torch.tensor(x2), torch.tensor(y2)
    acc2 = float((forward(xt2, 24).argmax(-1) == yt2).float().mean())
    return acc, acc2


def bench_rope_attn(seed: int = 1701, iters: int = 800) -> dict[str, float]:
    acc_r, acc_r2 = _train_and_eval(seed, True, iters)
    acc_a, acc_a2 = _train_and_eval(seed + 1, False, iters)
    return {
        "synthetic_rope_recall": acc_r,
        "synthetic_rope_recall_2x": acc_r2,
        "synthetic_learned_recall": acc_a,
        "synthetic_learned_recall_2x": acc_a2,
        "synthetic_rope_extrap_gain": acc_r2 - acc_a2,
        "synthetic_torch_available": 1.0,
    }
