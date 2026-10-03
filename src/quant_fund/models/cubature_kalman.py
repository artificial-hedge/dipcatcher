"""Cubature Kalman filter (3rd-degree spherical-radial rule) vs EKF on a
nonlinear range-bearing tracking problem; 2n cubature points.
"""

import numpy as np


def _step(x: np.ndarray) -> np.ndarray:
    a = np.array([[0.9, 0.0], [0.0, 0.95]])
    return np.asarray(a @ x + np.array([0.3 * np.sin(1.5 * x[0]), 0.0]))


def _meas(x: np.ndarray) -> float:
    return float(np.arctan2(x[1], x[0] + 2.0) + 0.3 * x[0])


def _sim(seed: int, n: int = 300) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 2))
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = _step(x[t - 1]) + rng.normal(0, 0.1, 2)
        y[t] = _meas(x[t]) + rng.normal(0, 0.2)
    return x, y


def _ckf(y: np.ndarray) -> np.ndarray:
    q, r = np.eye(2) * 0.01, 0.04
    x, p = np.zeros(2), np.eye(2)
    out = np.zeros((len(y), 2))
    pts = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]) * np.sqrt(2.0)
    for t in range(len(y)):
        s = np.linalg.cholesky(p + 1e-9 * np.eye(2))
        cub = np.array([_step(x + s @ pt) for pt in pts])
        xm = cub.mean(0)
        pm = np.cov(cub.T) + q
        s2 = np.linalg.cholesky(pm + 1e-9 * np.eye(2))
        cub2 = xm[None, :] + pts @ s2.T
        zc = np.array([_meas(c) for c in cub2])
        zm = zc.mean()
        pzz = np.var(zc) + r
        pxz = (cub2 - xm).T @ (zc - zm) / len(zc)
        k = pxz / pzz
        x = xm + k * (y[t] - zm)
        p = pm - np.outer(k, pxz)
        out[t] = x
    return out


def _ekf(y: np.ndarray) -> np.ndarray:
    q, r = np.eye(2) * 0.01, 0.04
    x, p = np.zeros(2), np.eye(2)
    a = np.array([[0.9, 0.0], [0.0, 0.95]])
    out = np.zeros((len(y), 2))
    for t in range(len(y)):
        x = _step(x)
        p = a @ p @ a.T + q
        z = float(np.arctan2(x[1], x[0] + 2.0) + 0.3 * x[0])
        d = (x[0] + 2.0) ** 2 + x[1] ** 2
        h = np.array([-x[1] / d + 0.3, (x[0] + 2.0) / d])
        s_ = float(h @ p @ h + r)
        k = p @ h / s_
        x = x + k * (y[t] - z)
        p = (np.eye(2) - np.outer(k, h)) @ p
        out[t] = x
    return out


def bench_cubature_kalman(seed: int = 5703) -> dict[str, float]:
    x, y = _sim(seed)
    c = _ckf(y)
    e = _ekf(y)
    c_rmse = float(np.sqrt(np.mean((c - x) ** 2)))
    e_rmse = float(np.sqrt(np.mean((e - x) ** 2)))
    return {
        "synthetic_ckf_rmse": c_rmse,
        "synthetic_ckf_ekf_rmse": e_rmse,
        "synthetic_ckf_gain": float(c_rmse < e_rmse),
    }
