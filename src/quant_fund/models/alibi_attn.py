"""ALiBi attention (Press et al. 2022) — fixed linear distance bias on (SYNTHETIC)
attention logits (no position embeddings at all) vs learned absolute on
the induction-head task at train length and 2x extrapolation.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._lm_synth import VOCAB, recall_batch


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("alibi_attn requires torch (pip install -e .[nn])") from exc
    return torch


def _train_and_eval(seed: int, alibi: bool, iters: int = 800) -> tuple[float, float]:
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
        if not alibi:
            h = h + pos.weight[:T]
        q, k = qk(h).chunk(2, -1)
        logits = q @ k.transpose(1, 2) / D**0.5
        if alibi:
            dist = torch.arange(T)[:, None] - torch.arange(T)[None, :]
            logits = logits + (dist.float() * -0.25)[None]
        att = torch.softmax(logits, -1)
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


def bench_alibi_attn(seed: int = 1707, iters: int = 800) -> dict[str, float]:
    acc_al, acc_al2 = _train_and_eval(seed, True, iters)
    acc_a, acc_a2 = _train_and_eval(seed + 1, False, iters)
    return {
        "synthetic_alibi_recall": acc_al,
        "synthetic_alibi_recall_2x": acc_al2,
        "synthetic_learned_recall": acc_a,
        "synthetic_learned_recall_2x": acc_a2,
        "synthetic_alibi_extrap_gain": acc_al2 - acc_a2,
        "synthetic_torch_available": 1.0,
    }
