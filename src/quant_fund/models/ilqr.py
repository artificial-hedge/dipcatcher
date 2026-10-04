"""iLQR: iterative linear-quadratic regulator for nonlinear dynamics."""

from __future__ import annotations

import numpy as np


def ilqr(
    dyn,
    cost,
    x0: np.ndarray,
    u_init: np.ndarray,
    iters: int = 60,
    reg: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Minimize sum_t l(x_t,u_t) + l_f(x_T) s.t. x_{t+1} = dyn(x_t, u_t).

    `dyn` returns x_{t+1}; `cost` returns (l, lx, lu, lxx, lxu, luu) per step
    via finite differences on scalar functions `lc(x,u)` and `lf(x)`.
    """
    x0 = np.asarray(x0, float)
    n, m = x0.size, u_init.shape[1]
    T = u_init.shape[0]
    xs = np.zeros((T + 1, n))
    us = u_init.copy()
    trace: list[float] = []

    def _rollout(us_: np.ndarray) -> tuple[np.ndarray, float]:
        x = x0.copy()
        xs_ = np.zeros((T + 1, n))
        xs_[0] = x
        total = 0.0
        for t in range(T):
            total += cost(x, us_[t])
            x = dyn(x, us_[t])
            xs_[t + 1] = x
        total += cost(x, np.zeros(m), terminal=True)
        return xs_, total

    for _ in range(iters):
        xs, total = _rollout(us)
        trace.append(total)
        # backward pass
        vx = np.zeros(n)
        vxx = np.zeros((n, n))
        _, vx, _, vxx, _, _ = cost(xs[-1], np.zeros(m), terminal=True, grads=True)
        k_list, K_list = [], []
        for t in reversed(range(T)):
            loss, lx, lu, lxx, lxu, luu = cost(xs[t], us[t], grads=True)
            fx = _fd_jac(lambda x, u=us[t]: dyn(x, u), xs[t])
            fu = _fd_jac(lambda u, x=xs[t]: dyn(x, u), us[t])
            _ = loss
            Qx = lx + fx.T @ vx
            Qu = lu + fu.T @ vx
            Qxx = lxx + fx.T @ vxx @ fx
            Quu = luu + fu.T @ vxx @ fu + reg * np.eye(m)
            Qux = lxu + fu.T @ vxx @ fx
            k = -np.linalg.solve(Quu, Qu)
            K = -np.linalg.solve(Quu, Qux)
            k_list.append(k)
            K_list.append(K)
            vx = Qx + K.T @ Quu @ k + K.T @ Qu + Qux.T @ k
            vxx = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K
        # forward pass with line search
        for alpha in (1.0, 0.5, 0.25, 0.1):
            xs_new = np.zeros_like(xs)
            us_new = np.zeros_like(us)
            xs_new[0] = x0
            for t in range(T):
                dx = xs_new[t] - xs[t]
                us_new[t] = us[t] + alpha * k_list[T - 1 - t] + K_list[T - 1 - t] @ dx
                xs_new[t + 1] = dyn(xs_new[t], us_new[t])
            _, total_new = _rollout(us_new)
            if total_new < total:
                xs, us = xs_new, us_new
                break
        else:
            break
    return xs, us, trace


def _fd_jac(f, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    f0 = f(x)
    J = np.zeros((f0.size, x.size))
    for j in range(x.size):
        xp = x.copy()
        xp[j] += h
        J[:, j] = (f(xp) - f0) / h
    return J


def bench_ilqr(seed: int = 20261231 + 863) -> dict[str, float]:
    """Pendulum swing-up: iLQR beats the zero-torque baseline and holds goal."""
    rng = np.random.default_rng(seed)
    g_, m_, l_, b_ = 9.81, 1.0, 1.0, 0.1
    dt = 0.05

    def dyn(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        th, w = x
        acc = -g_ / l_ * np.sin(th) - b_ * w + u[0] / (m_ * l_ * l_)
        return np.array([th + dt * w, w + dt * acc])

    def _l(x: np.ndarray, u: np.ndarray, terminal: bool = False) -> float:  # noqa: E741
        th_err = np.arctan2(np.sin(x[0] - np.pi), np.cos(x[0] - np.pi))
        q = th_err**2 + 0.1 * x[1] ** 2
        return float(
            (10.0 * q if terminal else 0.01 * q) + (0.0 if terminal else 0.001 * float(u[0] ** 2))
        )

    def cost(x: np.ndarray, u: np.ndarray, terminal: bool = False, grads: bool = False):
        loss = _l(x, u, terminal)
        if not grads:
            return loss
        h = 1e-5
        n = 2
        lx = np.zeros(n)
        lxx = np.zeros((n, n))
        lxu = np.zeros((1, n))
        lu = (
            np.array([(_l(x, u + np.array([h]), terminal) - loss) / h])
            if not terminal
            else np.zeros(1)
        )
        for j in range(n):
            xp, xm = x.copy(), x.copy()
            xp[j] += h
            xm[j] -= h
            lx[j] = (_l(xp, u, terminal) - _l(xm, u, terminal)) / (2 * h)
            lxx[j, j] = (_l(xp, u, terminal) - 2 * loss + _l(xm, u, terminal)) / h**2
            if not terminal:
                xpu = xp.copy()
                lxu[0, j] = (
                    _l(xpu, u + np.array([h])) - _l(xpu, u) - _l(x, u + np.array([h])) + _l(x, u)
                ) / h**2
        luu = np.array([[0.002]]) if not terminal else np.zeros((1, 1))
        return loss, lx, lu, lxx, lxu, luu

    checks = 0.0
    total = 0
    T = 120
    x0 = np.array([0.0, 0.0])
    us0 = np.zeros((T, 1)) + rng.normal(0, 0.01, (T, 1))
    xs, us, trace = ilqr(dyn, cost, x0, us0, iters=40)
    total += 3
    checks += float(trace[-1] < trace[0])
    # terminal angle near pi (mod 2pi)
    th_err = np.arctan2(np.sin(xs[-1][0] - np.pi), np.cos(xs[-1][0] - np.pi))
    checks += float(abs(th_err) < 0.3)
    # torque stays bounded
    checks += float(np.max(np.abs(us)) < 50.0)
    return {"synthetic_ilqr": checks / total}
