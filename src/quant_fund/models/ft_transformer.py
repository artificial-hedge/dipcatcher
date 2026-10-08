"""FT-Transformer: feature-tokenized transformer for tabular alpha (SYNTHETIC).

Gorishniy et al. 2021 (FT-Transformer): tokenize each scalar feature
via a per-feature learned linear+ReLU embedding plus a [CLS] token;
multi-head attention then lets features interact multiplicatively —
something a bagged-MLP cannot do cheaply.

Bench: synthetic tabular rows where the label is an interaction
(x1 * x2 > threshold) plus noise: feature attention should beat a
linear probe and a bagged ensemble of per-feature stumps.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _torch() -> Any:
    try:
        import torch

        return torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("ft_transformer requires the `nn` extra (make sync)") from exc


def synth_interaction(n: int, d: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    """(n, d) features; label = 1[x0*x1 + 0.5x2^2 - x3 > 0.6] + noise."""
    x = rng.standard_normal((n, d))
    score = x[:, 0] * x[:, 1] + 0.5 * x[:, 2] ** 2 - x[:, 3]
    y = (score + 0.5 * rng.standard_normal(n) > 0.6).astype(np.float64)
    return x.astype(np.float64), y


def _acc(pred: FloatArray, y: FloatArray) -> float:
    return float(np.mean((pred > 0.5) == (y > 0.5)))


def bench_ft_transformer(
    seed: int = 20261231,
    n: int = 400,
    d: int = 10,
    iters: int = 2200,
) -> dict[str, float]:
    """Feature-token attention vs logistic and a flat MLP classifier."""
    torch = _torch()
    torch.manual_seed(int(seed))  # audit sweep: seeded determinism
    rng = np.random.default_rng(seed)
    xs, y = synth_interaction(n, d, rng)
    xb = torch.tensor(xs, dtype=torch.float32)
    yb = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    tr = slice(0, 3 * n // 4)
    te = slice(3 * n // 4, n)

    tok = torch.nn.Linear(1, 32)
    cls = torch.nn.Parameter(torch.randn(1, 1, 32) * 0.1)
    attn = torch.nn.MultiheadAttention(32, 4, batch_first=True)
    head = torch.nn.Linear((d + 1) * 32, 1)
    params = torch.nn.ModuleList([tok, attn, head])
    opt = torch.optim.Adam(list(params.parameters()) + [cls], lr=2e-3)
    for _ in range(iters):
        t = tok(xb[tr].unsqueeze(-1))  # (B, d, e)
        t = torch.cat([cls.expand(xb[tr].shape[0], -1, -1), t], 1)
        h, _ = attn(t, t, t)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            head(h.reshape(xb[tr].shape[0], -1)).squeeze(1), yb[tr].squeeze(1)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        t = tok(xb[te].unsqueeze(-1))
        t = torch.cat([cls.expand(xb[te].shape[0], -1, -1), t], 1)
        h, _ = attn(t, t, t)
        ft_acc = _acc(
            torch.sigmoid(head(h.reshape(xb[te].shape[0], -1)).squeeze(1)).numpy(),
            y[te],
        )

    mlp = torch.nn.Sequential(torch.nn.Linear(d, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
    pm = torch.nn.ModuleList([mlp])
    optm = torch.optim.Adam(pm.parameters(), lr=1e-3)
    for _ in range(iters):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            mlp(xb[tr]).squeeze(1), yb[tr].squeeze(1)
        )
        optm.zero_grad()
        loss.backward()
        optm.step()
    with torch.no_grad():
        mlp_acc = _acc(torch.sigmoid(mlp(xb[te]).squeeze(1)).numpy(), y[te])

    beta = np.asarray(
        np.linalg.solve(xs[tr].T @ xs[tr] + 1e-2 * np.eye(d), xs[tr].T @ (y[tr] - 0.5))
    )
    log_acc = _acc(xs[te] @ beta + 0.5, y[te])
    return {
        "synthetic_ft_acc": ft_acc,
        "synthetic_ft_mlp_acc": mlp_acc,
        "synthetic_ft_logistic_acc": log_acc,
        "synthetic_ft_margin_vs_logistic": ft_acc - log_acc,
        "synthetic_ft_margin_vs_mlp": ft_acc - mlp_acc,
        "synthetic_torch_available": 1.0,
    }


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(bench_ft_transformer()))
