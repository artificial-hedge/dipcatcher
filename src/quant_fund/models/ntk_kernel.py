"""Empirical NTK — Jacobian-kernel ridge regression with a trained MLP's (SYNTHETIC)
features vs the raw-feature kernel ridge: generalization gain on the
regime task.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._td_synth import make_data, train_mlp

FloatArray = NDArray[np.float64]


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("ntk_kernel requires torch (pip install -e .[nn])") from exc
    return torch


def _emp_ntk(torch, net, Xa: FloatArray, Xb: FloatArray) -> FloatArray:
    params = list(net.parameters())
    J = []
    for x in torch.tensor(Xa).float():
        net.zero_grad()
        out = net(x[None, :])
        g = torch.autograd.grad(out, params, retain_graph=False)
        J.append(torch.cat([q.reshape(-1) for q in g]).detach())
    Ja = torch.stack(J)
    J = []
    for x in torch.tensor(Xb).float():
        net.zero_grad()
        out = net(x[None, :])
        g = torch.autograd.grad(out, params, retain_graph=False)
        J.append(torch.cat([q.reshape(-1) for q in g]).detach())
    Jb = torch.stack(J)
    return np.asarray((Ja @ Jb.T).numpy())


def _krr(K: FloatArray, y: FloatArray, Kt: FloatArray, lam: float = 1e-3) -> FloatArray:
    n = len(K)
    a = np.linalg.solve(K + lam * np.eye(n), y)
    return Kt @ a


def bench_ntk_kernel(seed: int = 2359) -> dict[str, float]:
    torch = _torch()
    torch.manual_seed(int(seed))  # audit sweep: seeded determinism
    X, y, Xt, yt = make_data(seed)
    net, _ = train_mlp(torch, X, y, iters=400, seed=seed)
    Xs, Xts = X[:120], Xt[:120]
    K = _emp_ntk(torch, net, Xs, Xs)
    Kt = _emp_ntk(torch, net, Xs, Xts).T
    pred_ntk = _krr(K, 2 * y[:120] - 1, Kt)
    acc_ntk = float(((pred_ntk > 0).astype(float) == yt[:120]).mean())
    # raw-feature linear-kernel ridge
    Kr = Xs @ Xs.T
    Ktr = Xts @ Xs.T
    pred_lin = _krr(Kr, 2 * y[:120] - 1, Ktr)
    acc_lin = float(((pred_lin > 0).astype(float) == yt[:120]).mean())
    return {
        "synthetic_ntk_acc": acc_ntk,
        "synthetic_lin_kernel_acc": acc_lin,
        "synthetic_ntk_gain": acc_ntk - acc_lin,
        "synthetic_torch_available": 1.0,
    }
