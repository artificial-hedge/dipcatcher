"""Dynamic movement primitives: LWR-fitted forcing term w/ goal generalization.

Canonical system: tau * s_dot = -alpha_s * s  (s: 1 -> 0).
Transformation:   tau * v_dot = a (b (g - y) - v) + f(s)
                  tau * y_dot = v
The forcing term f(s) = sum_i w_i psi_i(s) * s / sum_i psi_i(s) is learned by
weighted linear regression on a demonstrated trajectory, then the system
re-plans to arbitrary new goals with the same temporal shape.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 980


def canonical(t: np.ndarray, tau: float, alpha_s: float = 4.0) -> np.ndarray:
    return np.asarray(np.exp(-alpha_s * t / tau))


def learn_dmp(
    y: np.ndarray, dt: float, tau: float, n_basis: int = 20
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b = 25.0, 6.25
    t = np.arange(len(y)) * dt
    s = canonical(t, tau)
    yd = np.gradient(y, dt)
    ydd = np.gradient(yd, dt)
    g = y[-1]
    scale = g - y[0]
    f_target = (ydd - a * (b * (g - y) - yd)) / (scale if abs(scale) > 1e-9 else 1.0)
    centers = canonical(np.linspace(0, t[-1], n_basis), tau)
    h = np.ones(n_basis) * (n_basis**2) * 0.5
    psi = np.exp(-h[None, :] * (s[:, None] - centers[None, :]) ** 2)
    a_mat = psi * s[:, None] / (psi.sum(axis=1) + 1e-9)[:, None]
    ata = a_mat.T @ a_mat
    lam = 1e-3 * float(np.trace(ata)) / n_basis
    w = np.linalg.solve(ata + lam * np.eye(n_basis), a_mat.T @ f_target)
    return np.asarray(w), centers, h


def rollout(
    w: np.ndarray,
    centers: np.ndarray,
    h: np.ndarray,
    y0: float,
    g: float,
    steps: int,
    dt: float,
    tau: float,
) -> np.ndarray:
    a, b = 25.0, 6.25
    y = float(y0)
    v = 0.0
    out = np.zeros(steps)
    for i in range(steps):
        s = float(np.exp(-4.0 * (i * dt) / tau))
        psi = np.exp(-h * (s - centers) ** 2)
        f = float(np.sum(w * psi * s) / (np.sum(psi) + 1e-9)) * (g - y0)
        v += (a * (b * (g - y) - v) + f) * dt
        y += v * dt
        out[i] = y
    return out


def bench_dmp_control(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    checks: list[bool] = []
    n = 200
    t = np.linspace(0, 1, n)
    demo = t + 0.5 * np.sin(2 * np.pi * t) - 0.5 * np.sin(2 * np.pi * t[0])
    demo = np.asarray(demo, dtype=float)
    w, centers, h = learn_dmp(demo, 1.0 / n, 1.0)
    traj = rollout(w, centers, h, demo[0], demo[-1], n, 1.0 / n, 1.0)
    checks.append(float(np.linalg.norm(traj - demo) / np.linalg.norm(demo)) < 0.15)
    g_new = 3.0
    traj2 = rollout(w, centers, h, demo[0], g_new, 4 * n, 0.5 / n, 1.0)
    checks.append(abs(float(traj2[-1]) - g_new) < 0.05)
    c = float(np.corrcoef(traj2[10 : 2 * n : 2], demo[5:n])[0, 1])
    checks.append(c > 0.9)
    y0_new = 1.0
    traj3 = rollout(w, centers, h, y0_new, 2.0, 4 * n, 0.5 / n, 1.0)
    checks.append(abs(float(traj3[0]) - y0_new) < 0.05 and abs(float(traj3[-1]) - 2.0) < 0.05)
    _ = rng
    score = float(np.mean(checks))
    return {"synthetic_dmp_control": score}
