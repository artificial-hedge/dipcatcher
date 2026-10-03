"""Set Transformer (ISAB-lite) for order-flow sets (torch).

A set of child orders [size, side, aggressiveness] is embedded, pooled by
an induced set-attention block (learnable seed vectors attending over the
masked set), and mapped to the realized parent-fill shortfall sign. Set
operations must be permutation-invariant — verified explicitly. Requires
the ``nn`` extra; SYNTHETIC order sets only.

Bench: synthetic order sets where the label depends on interactions
across set members; R² vs a mean/std statistic-pooling baseline, sign
accuracy, and a permutation-invariance error.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _torch() -> Any:
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("set_transformer torch path requires the `nn` extra (make sync)") from exc


def torch_available() -> bool:
    try:
        _torch()
        return True
    except ImportError:
        return False


def synth_order_sets(n: int, rng: np.random.Generator) -> tuple[list[FloatArray], FloatArray]:
    """Variable-size order sets; label = nonlinear cross-member interaction."""
    xs: list[FloatArray] = []
    ys = np.zeros(n)
    for i in range(n):
        k = int(rng.integers(3, 9))
        sz = rng.gamma(2.0, 20.0, k)
        side = rng.choice([-1.0, 1.0], k, p=[0.4, 0.6])
        agg = rng.random(k)
        xs.append(np.stack([sz / 100.0, side, agg], 1))
        imbalance = float(np.sum(sz * side)) / (np.sum(sz) + 1e-9)
        front_load = float(np.sum(sz * agg) / (np.sum(sz) + 1e-9))
        dispersion = float(np.std(side * agg))
        iceberg = float(np.max(sz * agg * side)) / 100.0  # single dominant aggressive child
        ys[i] = (
            1.5 * imbalance
            + 0.8 * front_load
            - 1.2 * dispersion
            + 1.4 * iceberg
            + 0.1 * rng.standard_normal()
        )
    return xs, ys


def _pad(xs: list[FloatArray]) -> tuple[Any, Any]:
    k = max(len(x) for x in xs)
    import torch  # noqa: F401 — typing aid only

    pad = np.zeros((len(xs), k, 3))
    mask = np.zeros((len(xs), k), dtype=bool)
    for i, x in enumerate(xs):
        pad[i, : len(x)] = x
        mask[i, : len(x)] = True
    return pad, mask


def _stat_pool_r2(xs: list[FloatArray], ys: FloatArray, tr: int) -> tuple[float, float]:
    def feats(x: FloatArray) -> FloatArray:
        return np.array(
            [
                np.mean(x[:, 0] * x[:, 1]),
                np.sum(x[:, 0] * x[:, 1]) / (np.sum(x[:, 0]) + 1e-9),
                np.mean(x[:, 2]),
                np.std(x[:, 1] * x[:, 2]),
                len(x),
            ]
        )

    f = np.stack([feats(x) for x in xs])
    f = np.hstack([f, np.ones((len(f), 1))])
    w = np.asarray(
        np.linalg.solve(f[:tr].T @ f[:tr] + 1e-3 * np.eye(f.shape[1]), f[:tr].T @ ys[:tr])
    )
    pred = f[tr:] @ w
    ss = 1 - np.sum((ys[tr:] - pred) ** 2) / (np.sum((ys[tr:] - ys[tr:].mean()) ** 2) + 1e-9)
    sign_acc = float(np.mean(np.sign(pred) == np.sign(ys[tr:])))
    return float(ss), sign_acc


def bench_set_transformer(seed: int = 59) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    torch = _torch()
    torch.manual_seed(seed)
    n = 3000
    xs, ys_np = synth_order_sets(n, rng)
    tr = int(n * 0.8)
    pad, mask = _pad(xs)
    xb = torch.tensor(pad, dtype=torch.float32)
    mk = torch.tensor(mask)
    yb = torch.tensor(ys_np, dtype=torch.float32)

    emb = torch.nn.Linear(3, 24)
    induced = torch.nn.MultiheadAttention(24, 2, batch_first=True)
    seed_vec = torch.nn.Parameter(torch.zeros(1, 1, 24))
    head = torch.nn.Linear(24, 1)
    mods = torch.nn.ModuleList([emb, induced, head])

    def forward(x, m):
        h = torch.relu(emb(x))
        ind = seed_vec.expand(len(x), 4, -1)
        h, _ = induced(ind, h, h, key_padding_mask=~m)
        return head(h.mean(1)).squeeze(-1)

    opt = torch.optim.Adam(list(mods.parameters()) + [seed_vec], lr=3e-3)
    for _ in range(200):
        idx = torch.randint(0, tr, (256,))
        loss = torch.mean((forward(xb[idx], mk[idx]) - yb[idx]) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred = forward(xb[tr:], mk[tr:]).numpy()
        yt = yb[tr:].numpy()
        r2 = float(1 - np.sum((yt - pred) ** 2) / (np.sum((yt - yt.mean()) ** 2) + 1e-9))
        sign_acc = float(np.mean(np.sign(pred) == np.sign(yt)))
        # permutation invariance: permute first 40 test sets
        errs = []
        for i in range(tr, tr + 40):
            p = rng.permutation(xs[i].shape[0])
            xi = torch.tensor(xs[i][p][None], dtype=torch.float32)
            mi = torch.ones(1, len(p), dtype=torch.bool)
            pi = forward(xi, mi)
            xo = torch.tensor(xs[i][None], dtype=torch.float32)
            po = forward(xo, mi)
            errs.append(abs(float(pi - po)))
    r2_base, sign_base = _stat_pool_r2(xs, ys_np, tr)
    return {
        "synthetic_setformer_r2": r2,
        "synthetic_setformer_statpool_r2": r2_base,
        "synthetic_setformer_r2_margin": r2 - r2_base,
        "synthetic_setformer_sign_acc": sign_acc,
        "synthetic_setformer_statpool_sign_acc": sign_base,
        "synthetic_setformer_perm_invariance_err": float(np.max(errs)),
        "torch_available": 1.0,
    }
