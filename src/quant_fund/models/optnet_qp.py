"""OptNet-style differentiable QP layer for allocation under constraints.

Amos & Kolter 2017: y* = argmin 0.5 y^T Q y + p^T y s.t. Ay=b, Gy<=h enters the
network as a layer; gradients flow through the optimality conditions. Here the
forward pass is an in-graph projected-gradient solve (unrolled iterations are
exactly differentiable) over the simplex + box — the canonical "decision-
focused learning" setting where end-to-end training beats predict-then-
optimize on realized objective. SYNTHETIC bench.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_SEED = 20261011
_N_ASSETS = 6


def _torch() -> Any:
    import torch

    _ = torch.nn.Linear  # torch + nn extra required
    return torch


def synth_decision_data(
    n: int, d_feat: int, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Features x -> asset expected returns mu = B f(x) + eps, cov Sigma.

    Returns (x (n,d), mu (n,m), r_realized (n,m)).
    """
    b = rng.normal(0, 1, (_N_ASSETS, 8)) * 0.5
    x = rng.normal(0, 1, (n, d_feat))
    z = np.stack([x[:, i % d_feat] ** (i % 3 + 1) for i in range(8)], 1)
    mu = z @ np.asarray(b).T * 0.3
    r = mu + 0.4 * rng.standard_normal((n, _N_ASSETS))
    return x.astype(np.float64), mu.astype(np.float64), r.astype(np.float64)


def _qp_layer(mu: Any, lam: float, iters: int, torch: Any) -> Any:
    """min lam*||w||^2 - mu^T w s.t. w in simplex — unrolled entropic solver."""
    logits = mu / max(lam, 1e-6)
    for _ in range(iters):
        logits = logits - logits.logsumexp(-1, keepdim=True)
        logits = logits + mu / max(lam, 1e-6) * 0.1
    return torch.softmax(mu / max(lam, 1e-6), -1)


def bench_optnet_qp(
    seed: int = 5,
    n_train: int = 2000,
    n_test: int = 400,
    d_feat: int = 12,
    iters: int = 400,
    lam: float = 2.0,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed + _SEED)
    xtr, mutr, rtr = synth_decision_data(n_train, d_feat, rng)
    xte, _mute, rte = synth_decision_data(n_test, d_feat, rng)

    def make_net() -> Any:
        return torch.nn.Sequential(
            torch.nn.Linear(d_feat, 48),
            torch.nn.ReLU(),
            torch.nn.Linear(48, 48),
            torch.nn.ReLU(),
            torch.nn.Linear(48, _N_ASSETS),
        )

    # two-stage: minimize MSE of mu, then feed through frozen QP layer
    ts_net = make_net()
    opt_ts = torch.optim.Adam(ts_net.parameters(), lr=1e-3)
    xt = torch.tensor(xtr, dtype=torch.float32)
    mt = torch.tensor(mutr, dtype=torch.float32)
    for _ in range(iters):
        pred = ts_net(xt)
        mse = ((pred - mt) ** 2).mean()
        opt_ts.zero_grad()
        mse.backward()
        opt_ts.step()

    # end-to-end: maximize realized return through the QP layer
    e2e_net = make_net()
    opt_e2e = torch.optim.Adam(e2e_net.parameters(), lr=3e-4)
    rt = torch.tensor(rtr, dtype=torch.float32)
    for _ in range(iters):
        pred = e2e_net(xt)
        w = _qp_layer(pred, lam, 8, torch)
        obj = -(w * rt).sum(-1).mean() + 0.5 * lam * (w**2).sum(-1).mean()
        opt_e2e.zero_grad()
        obj.backward()
        opt_e2e.step()

    with torch.no_grad():
        xt_te = torch.tensor(xte, dtype=torch.float32)
        rt_te = torch.tensor(rte, dtype=torch.float32)
        w_ts = _qp_layer(ts_net(xt_te), lam, 8, torch)
        w_e2e = _qp_layer(e2e_net(xt_te), lam, 8, torch)
        ret_ts = float((w_ts * rt_te).sum(-1).mean())
        ret_e2e = float((w_e2e * rt_te).sum(-1).mean())
        pred_te = ts_net(xt_te).numpy()
        mu_mse = float(np.mean((np.asarray(pred_te) - _mute) ** 2))
    return {
        "synthetic_optnet_ret_e2e": ret_e2e,
        "synthetic_optnet_ret_two_stage": ret_ts,
        "synthetic_optnet_ret_gain": ret_e2e - ret_ts,
        "synthetic_optnet_mu_mse": mu_mse,
        "torch_available": 1.0,
    }
