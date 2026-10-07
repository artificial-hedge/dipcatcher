"""Shared SYNTHETIC fixture for wave-175 LM-components canon:
induction-head task — random token sequences where a marked key repeats;
the target is the token following the key's first occurrence. Tests
attention mechanisms' content-based retrieval. Also a multi-regime
density task for MoE routing.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

VOCAB = 16


def recall_batch(
    seed: int, B: int = 32, T: int = 12
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """x: (B,T) tokens; y: (B,) value token after first key occurrence."""
    rng = np.random.default_rng(seed)
    x = rng.integers(1, VOCAB, (B, T))
    key = rng.integers(1, VOCAB, (B,))
    val = rng.integers(1, VOCAB, (B,))
    y = np.zeros(B, dtype=np.int64)
    for b in range(B):
        i1 = int(rng.integers(0, T - 6))
        i2 = int(rng.integers(i1 + 2, T - 1))
        x[b, i1] = key[b]
        x[b, i1 + 1] = val[b]
        x[b, i2] = key[b]
        y[b] = val[b]
    return x, y


def regime_task(seed: int, n: int = 600, k: int = 4, d: int = 8) -> tuple[FloatArray, FloatArray]:
    """n samples, k regimes: y = sigmoid(x @ w_r) with regime-dependent w."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((k, d))
    W = W / np.linalg.norm(W, axis=1, keepdims=True) * 3.0
    r = rng.integers(0, k, n)
    X = rng.standard_normal((n, d))
    s = (X * W[r]).sum(-1)
    # clean decision boundary + label-flip noise
    y = (s + 0.4 * rng.standard_normal(n) > 0).astype(np.float64)
    return X, y


def attn_baseline(seed: int, iters: int = 800) -> float:
    """Single-head full-attention readout on recall_batch — shared comparator."""
    import torch

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        x, y = recall_batch(seed)
        D = 16
        emb = torch.nn.Embedding(VOCAB, D)
        wq = torch.nn.Linear(D, D)
        wk = torch.nn.Linear(D, D)
        wv = torch.nn.Linear(D, D)
        head = torch.nn.Linear(D, VOCAB)
        opt = torch.optim.Adam(
            list(emb.parameters())
            + list(wq.parameters())
            + list(wk.parameters())
            + list(wv.parameters())
            + list(head.parameters()),
            lr=0.005,
        )
        X = torch.tensor(x)
        Y = torch.tensor(y)
        for _ in range(iters):
            h = emb(X)
            q, k, v = wq(h), wk(h), wv(h)
            a = torch.softmax(q @ k.transpose(-1, -2) / D**0.5, -1)
            o = a @ v
            loss = torch.nn.functional.cross_entropy(head(o[:, -1]), Y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            h = emb(X)
            a = torch.softmax(wq(h) @ wk(h).transpose(-1, -2) / D**0.5, -1)
            acc = (head((a @ wv(h))[:, -1]).argmax(-1) == Y).float().mean().item()
        return float(acc)
