"""Factorization machines (Rendle 2010): degree-2 feature (SYNTHETIC)
interactions through k-dim latent factors, trained by SGD on
squared or logistic loss in O(nk) per epoch via the
(Σv·x)² − Σ(v·x)² identity. Synthetic bench gates that the
FM beats a pure-linear baseline on an interaction-only
target."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def fm_fit(
    x: FloatArray,
    y: FloatArray,
    k: int = 8,
    it: int = 60,
    lr: float = 0.02,
    reg: float = 1e-4,
    binary: bool = False,
    seed: int = 0,
) -> dict[str, object]:
    """SGD over shuffled samples; latent factors v ∈ R^{p×k}."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, p = x.shape
    rng = np.random.default_rng(seed)
    w0 = float(y.mean())
    w = np.zeros(p)
    v = rng.normal(0, 0.1, (p, k))
    for _ in range(it):
        order = rng.permutation(n)
        for i in order:
            xi = x[i]
            vx = v * xi[:, None]
            inter = 0.5 * float((vx.sum(axis=0) ** 2 - (vx**2).sum(axis=0)).sum())
            pred = w0 + float(xi @ w) + inter
            if binary:
                err = float(1 / (1 + np.exp(np.clip(pred, -30, 30)))) - y[i]
            else:
                err = pred - y[i]
            w0 -= lr * err
            w -= lr * (err * xi + reg * w)
            # d(inter)/d(v_if) = x_i*(Σ_j v_jf x_j) - v_if*x_i²
            v -= lr * (err * xi[:, None] * (vx.sum(axis=0)[None, :] - vx) + reg * v)
    return {"w0": w0, "w": w, "v": v}


def fm_predict(model: dict[str, object], x: FloatArray) -> FloatArray:
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(model["v"])
    w = np.asarray(model["w"])
    w0 = float(np.asarray(model["w0"]))
    vx = x @ v
    inter = 0.5 * ((vx**2).sum(axis=1) - ((x**2) @ (v**2)).sum(axis=1))
    return np.asarray(w0 + x @ w + inter)


def _r2(y: FloatArray, pred: FloatArray) -> float:
    ss = float(((y - pred) ** 2).sum())
    base = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss / max(base, 1e-12)


def bench_factorization_machine(seed: int = 551) -> dict[str, float]:
    """SYNTHETIC: y = x0·x1 − x2·x3 + small noise — linear
    models see only noise; FM must explain > 0.7 R² while the
    linear baseline stays near 0."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    n, p = 400, 6
    x = rng.normal(0, 1, (n, p))
    y = x[:, 0] * x[:, 1] - x[:, 2] * x[:, 3] + rng.normal(0, 0.05, n)
    perm = rng.permutation(n)
    tr, te = perm[:300], perm[300:]
    mdl = fm_fit(x[tr], y[tr], k=8, it=80, lr=0.01, seed=seed)
    pred = fm_predict(mdl, x[te])
    r2_fm = _r2(y[te], np.asarray(pred))
    out["synthetic_fm_r2"] = r2_fm
    # pure linear ridge baseline
    xa = np.c_[x[tr], np.ones(tr.shape[0])]
    beta = np.linalg.solve(xa.T @ xa + 1e-3 * np.eye(p + 1), xa.T @ y[tr])
    pred_lin = np.c_[x[te], np.ones(te.shape[0])] @ beta
    r2_lin = _r2(y[te], np.asarray(pred_lin))
    out["synthetic_linear_r2"] = r2_lin
    if r2_fm < 0.7:
        raise ValueError(f"fm r2 off: {r2_fm}")
    if r2_fm < r2_lin + 0.3:
        raise ValueError(f"fm not >> linear: {r2_fm} vs {r2_lin}")
    return out
