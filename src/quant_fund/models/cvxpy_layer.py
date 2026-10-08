"""Differentiable convex layer — entropic-smoothed simplex LP as a net layer.

Agarwal et al. 2019 (cvxpylayers): solution maps of convex programs can be
backpropagated via the optimality conditions. On the probability simplex, the
entropic barrier turns the LP map into softmax(-c/tau), which is exactly
differentiable; this module trains a cost model end-to-end through that map
so the realized LP objective beats predict-then-optimize. SYNTHETIC bench.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.optnet_qp import _N_ASSETS, synth_decision_data

FloatArray = NDArray[np.float64]

_SEED = 20261012


def _torch() -> Any:
    import torch

    _ = torch.nn.Linear  # torch + nn extra required
    return torch


def _lp_layer(c: Any, tau: float, torch: Any) -> Any:
    """argmin c^T w over simplex, entropically smoothed: softmax(-c/tau)."""
    return torch.softmax(-c / max(tau, 1e-6), -1)


def bench_cvxpy_layer(
    seed: int = 9,
    n_train: int = 2000,
    n_test: int = 400,
    d_feat: int = 12,
    iters: int = 400,
    tau: float = 0.3,
) -> dict[str, float]:
    torch = _torch()
    torch.manual_seed(int(seed))  # audit sweep: seeded determinism
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

    xt = torch.tensor(xtr, dtype=torch.float32)
    mt = torch.tensor(mutr, dtype=torch.float32)
    rt = torch.tensor(rtr, dtype=torch.float32)

    ts_net = make_net()
    opt_ts = torch.optim.Adam(ts_net.parameters(), lr=1e-3)
    for _ in range(iters):
        mse = ((ts_net(xt) - mt) ** 2).mean()
        opt_ts.zero_grad()
        mse.backward()
        opt_ts.step()

    e2e_net = make_net()
    opt_e2e = torch.optim.Adam(e2e_net.parameters(), lr=3e-4)
    for _ in range(iters):
        cost = -e2e_net(xt)  # minimize c^T w == maximize mu^T w
        w = _lp_layer(cost, tau, torch)
        obj = -(w * rt).sum(-1).mean()
        opt_e2e.zero_grad()
        obj.backward()
        opt_e2e.step()

    with torch.no_grad():
        xt_te = torch.tensor(xte, dtype=torch.float32)
        rt_te = torch.tensor(rte, dtype=torch.float32)
        w_ts = _lp_layer(-ts_net(xt_te), tau, torch)
        w_e2e = _lp_layer(-e2e_net(xt_te), tau, torch)
        ret_ts = float((w_ts * rt_te).sum(-1).mean())
        ret_e2e = float((w_e2e * rt_te).sum(-1).mean())
        # oracle LP on true expected returns
        w_or = _lp_layer(-torch.tensor(_mute, dtype=torch.float32), tau, torch)
        ret_or = float((w_or * rt_te).sum(-1).mean())
    return {
        "synthetic_cvx_ret_e2e": ret_e2e,
        "synthetic_cvx_ret_two_stage": ret_ts,
        "synthetic_cvx_ret_oracle": ret_or,
        "synthetic_cvx_ret_gain": ret_e2e - ret_ts,
        "synthetic_cvx_regret_frac": (ret_or - ret_e2e) / max(abs(ret_or), 1e-9),
        "synthetic_torch_available": 1.0,
    }
