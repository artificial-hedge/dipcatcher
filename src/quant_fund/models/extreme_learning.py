"""Extreme learning machine canon: random-feature single-hidden-layer (SYNTHETIC)
networks — fixed random input weights + analytic ridge output layer
(Huang et al.). ``bench_extreme_learning`` fits a nonlinear regression
target and a classification task, gating ELM over the linear ridge
baseline and hidden-width scaling.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def elm_fit(
    x: FloatArray,
    y: FloatArray,
    n_hidden: int = 200,
    act: str = "sigmoid",
    lam: float = 1e-4,
    seed: int = 0,
) -> dict[str, FloatArray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.shape[0] != x.shape[0] or n_hidden < 1:
        raise ValueError("bad inputs")
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((x.shape[1], n_hidden))
    b = rng.standard_normal(n_hidden)
    z = x @ w + b
    if act == "sigmoid":
        h = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    elif act == "tanh":
        h = np.tanh(z)
    elif act == "relu":
        h = np.maximum(z, 0.0)
    else:
        raise ValueError(f"unknown act {act}")
    beta = np.linalg.solve(h.T @ h + lam * len(y) * np.eye(n_hidden), h.T @ y)
    return {"w": w, "b": b, "beta": beta}


def elm_predict(model: dict[str, FloatArray], x: FloatArray, act: str = "sigmoid") -> FloatArray:
    z = np.asarray(x, dtype=np.float64) @ model["w"] + model["b"]
    if act == "sigmoid":
        h = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    elif act == "tanh":
        h = np.tanh(z)
    else:
        h = np.maximum(z, 0.0)
    return np.asarray(h @ model["beta"])


def _ridge(x: FloatArray, y: FloatArray, lam: float = 1e-3) -> FloatArray:
    xa = np.column_stack([x, np.ones(len(x))])
    return np.asarray(np.linalg.solve(xa.T @ xa + lam * len(y) * np.eye(xa.shape[1]), xa.T @ y))


def bench_extreme_learning(seed: int = 20261231) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    # (a) nonlinear regression: y = sin(2x0) + x1^2 - 0.5
    n = 500
    x = rng.uniform(-2, 2, (n, 3))
    y = np.sin(2.0 * x[:, 0]) + x[:, 1] ** 2 - 0.5 + 0.05 * rng.standard_normal(n)
    tr = np.arange(n) < 400
    m = elm_fit(x[tr], y[tr], n_hidden=250, seed=seed)
    pred = elm_predict(m, x[~tr])
    wr = _ridge(x[tr], y[tr])
    pred_r = np.column_stack([x[~tr], np.ones((~tr).sum())]) @ wr
    rmse = lambda a: float(np.sqrt(np.mean((a - y[~tr]) ** 2)))  # noqa: E731
    # (b) classification: XOR-ish margin
    xc = rng.uniform(-1, 1, (n, 2))
    yc = (xc[:, 0] * xc[:, 1] > 0).astype(np.float64)
    mc = elm_fit(xc[tr], yc[tr], n_hidden=300, act="tanh", seed=seed + 1)
    pc = elm_predict(mc, xc[~tr], act="tanh")
    acc = float(np.mean((pc >= 0.5) == (yc[~tr] >= 0.5)))
    wc = _ridge(xc[tr], yc[tr])
    acc_r = float(
        np.mean((np.column_stack([xc[~tr], np.ones((~tr).sum())]) @ wc >= 0.5) == (yc[~tr] >= 0.5))
    )
    return {
        "synthetic_elm_rmse": rmse(pred),
        "synthetic_ridge_rmse": rmse(pred_r),
        "synthetic_elm_vs_ridge": rmse(pred_r) - rmse(pred),
        "synthetic_elm_cls_acc": acc,
        "synthetic_ridge_cls_acc": acc_r,
    }
