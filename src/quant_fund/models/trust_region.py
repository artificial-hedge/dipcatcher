"""Trust-region Newton with Cauchy/dogleg steps (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def tr_minimize(f, g, h, x0: np.ndarray, iters: int = 100) -> np.ndarray:
    x = x0.copy()
    delta = 1.0
    for _ in range(iters):
        gx = g(x)
        hx = h(x)
        # dogleg: Cauchy point + Newton step clipped to radius
        gn = gx @ gx / (gx @ hx @ gx + 1e-30)
        p_u = -gn * gx
        try:
            p_b = -np.linalg.solve(hx + 1e-10 * np.eye(len(x)), gx)
        except np.linalg.LinAlgError:
            p_b = p_u
        if np.linalg.norm(p_b) <= delta:
            p = p_b
        elif np.linalg.norm(p_u) >= delta:
            p = (delta / np.linalg.norm(p_u)) * p_u
        else:
            # interpolate along dogleg segment
            d = p_b - p_u
            a_ = d @ d
            b_ = 2 * p_u @ d
            c_ = p_u @ p_u - delta**2
            tau = (-b_ + np.sqrt(max(b_**2 - 4 * a_ * c_, 0))) / (2 * a_)
            p = p_u + tau * d
        # ratio of actual to predicted reduction
        pred = -(gx @ p + 0.5 * p @ hx @ p)
        act = f(x) - f(x + p)
        rho = act / pred if pred > 0 else -1.0
        if rho < 0.25:
            delta *= 0.25
        elif rho > 0.75 and abs(np.linalg.norm(p) - delta) < 1e-8:
            delta *= 2.0
        if rho > 0:
            x = x + p
        if np.linalg.norm(gx) < 1e-8:
            break
    return x


def _hess_rosen(x: np.ndarray) -> np.ndarray:
    return np.array([[1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]], [-400 * x[0], 200.0]])


def _bench_trust_region(seed: int = 0) -> float:
    checks = []
    # quadratic bowl min at (1,2)
    f = lambda v: float((v[0] - 1) ** 2 + 3 * (v[1] - 2) ** 2)  # noqa: E731
    g = lambda v: np.array([2 * (v[0] - 1), 6 * (v[1] - 2)])  # noqa: E731
    h = lambda v: np.array([[2.0, 0.0], [0.0, 6.0]])  # noqa: E731
    x = tr_minimize(f, g, h, np.array([5.0, -3.0]))
    checks.append(np.allclose(x, [1.0, 2.0], atol=1e-4))
    # Rosenbrock: converge to (1,1)
    fr = lambda v: float((1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2)  # noqa: E731
    gr = lambda v: np.array(  # noqa: E731
        [
            -2 * (1 - v[0]) - 400 * v[0] * (v[1] - v[0] ** 2),
            200 * (v[1] - v[0] ** 2),
        ]
    )
    xr = tr_minimize(fr, gr, _hess_rosen, np.array([-1.0, 1.0]), iters=200)
    checks.append(np.allclose(xr, [1.0, 1.0], atol=1e-2))
    # indefinite Hessian: saddle function stays bounded by radius
    fs = lambda v: float(v[0] ** 2 - v[1] ** 2)  # noqa: E731
    gs = lambda v: np.array([2 * v[0], -2 * v[1]])  # noqa: E731
    hs = lambda v: np.array([[2.0, 0.0], [0.0, -2.0]])  # noqa: E731
    xs = tr_minimize(fs, gs, hs, np.array([0.5, 0.0]), iters=20)
    checks.append(np.linalg.norm(xs) < 100.0)
    # f at found point <= start
    checks.append(f(x) <= f(np.array([5.0, -3.0])))
    return float(sum(checks) / len(checks))


def bench_trust_region(seed: int = 0) -> dict[str, float]:
    return {"synthetic_trust_region": _bench_trust_region(seed)}
