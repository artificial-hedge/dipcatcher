"""Rauch-Tung-Striebel Kalman smoother for linear-Gaussian systems."""

from __future__ import annotations

import numpy as np


def kalman_filter(
    ys: np.ndarray,
    F: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Return (filtered means, filtered covs, predicted means, predicted covs)."""
    xs, Ps, xs_p, Ps_p = [], [], [], []
    x, P = np.asarray(x0, float), np.asarray(P0, float)
    for y in ys:
        xp = F @ x
        Pp = F @ P @ F.T + Q
        S = H @ Pp @ H.T + R
        K = Pp @ H.T @ np.linalg.inv(S)
        x = xp + K @ (y - H @ xp)
        P = (np.eye(len(x)) - K @ H) @ Pp
        xs.append(x)
        Ps.append(P)
        xs_p.append(xp)
        Ps_p.append(Pp)
    return np.array(xs), Ps, xs_p, Ps_p


def rts_smoother(
    ys: np.ndarray,
    F: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
) -> np.ndarray:
    """Rauch-Tung-Striebel smoothed state sequence."""
    xs_f, Ps_f, xs_p, Ps_p = kalman_filter(ys, F, H, Q, R, x0, P0)
    T = len(ys)
    xs_s: list[np.ndarray] = [np.zeros(1)] * T
    xs_s[-1] = xs_f[-1]
    Ps_s: list[np.ndarray] = [np.zeros(1)] * T
    Ps_s[-1] = Ps_f[-1]
    for t in reversed(range(T - 1)):
        G = Ps_f[t] @ F.T @ np.linalg.inv(Ps_p[t + 1])
        xs_s[t] = xs_f[t] + G @ (xs_s[t + 1] - xs_p[t + 1])
        Ps_s[t] = Ps_f[t] + G @ (Ps_s[t + 1] - Ps_p[t + 1]) @ G.T
    return np.array(xs_s)


def _batch_map(
    ys: np.ndarray,
    F: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
) -> np.ndarray:
    """Oracle: batch least-squares MAP trajectory (Gauss-Markov process)."""
    T, n = len(ys), len(x0)
    Qi, Ri, Pi = np.linalg.inv(Q), np.linalg.inv(R), np.linalg.inv(P0)
    # solve for stacked states x = [x_0..x_{T-1}]
    # transition: x_{t+1} = F x_t + w  =>  treat ys with process prior on x_t
    # use information form on [x_0..x_{T-1}] with dynamics x_{t}=F x_{t-1}+w
    A = np.zeros((T * n, T * n))
    b = np.zeros(T * n)
    A[:n, :n] += Pi
    b[:n] += Pi @ x0
    for t in range(T):
        A[t * n : (t + 1) * n, t * n : (t + 1) * n] += H.T @ Ri @ H
        b[t * n : (t + 1) * n] += H.T @ Ri @ ys[t]
        if t < T - 1:
            # (x_{t+1} - F x_t)' Qi (x_{t+1} - F x_t)
            A[t * n : (t + 1) * n, t * n : (t + 1) * n] += F.T @ Qi @ F
            A[t * n : (t + 1) * n, (t + 1) * n : (t + 2) * n] -= F.T @ Qi
            A[(t + 1) * n : (t + 2) * n, t * n : (t + 1) * n] -= Qi @ F
            A[(t + 1) * n : (t + 2) * n, (t + 1) * n : (t + 2) * n] += Qi
    sol = np.linalg.solve(A + 1e-9 * np.eye(T * n), b)
    return sol.reshape(T, n)


def bench_rts_smoother(seed: int = 20261231 + 864) -> dict[str, float]:
    """Smoother beats filter MSE and matches batch MAP oracle."""
    rng = np.random.default_rng(seed)
    F = np.array([[1.0, 0.5], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[0.05, 0.0], [0.0, 0.05]])
    R = np.array([[0.5]])
    x0 = np.array([0.0, 0.0])
    P0 = np.eye(2)
    checks = 0.0
    total = 0
    for _ in range(10):
        T = 60
        xt = np.zeros((T, 2))
        xt[0] = x0
        for t in range(1, T):
            xt[t] = F @ xt[t - 1] + rng.multivariate_normal(np.zeros(2), Q)
        ys = np.array([H @ xt[t] + rng.normal(0, np.sqrt(R[0, 0])) for t in range(T)])
        xs_f, _, _, _ = kalman_filter(ys, F, H, Q, R, x0, P0)
        xs_s = rts_smoother(ys, F, H, Q, R, x0, P0)
        mse_f = float(np.mean(np.sum((xs_f - xt) ** 2, axis=1)))
        mse_s = float(np.mean(np.sum((xs_s - xt) ** 2, axis=1)))
        xs_map = _batch_map(ys, F, H, Q, R, x0, P0)
        total += 3
        checks += float(mse_s <= mse_f + 1e-9)
        checks += float(np.max(np.abs(xs_s - xs_map)) < 0.1)
        checks += float(xs_s.shape == xt.shape)
    return {"synthetic_rts": checks / total}
