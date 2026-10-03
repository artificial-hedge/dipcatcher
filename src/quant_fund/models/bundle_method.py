"""Proximal bundle method for nonsmooth convex minimization (SYNTHETIC)."""

from __future__ import annotations

import numpy as np

from quant_fund.models.ip_qp import qp_solve


def bundle_minimize(
    f,
    subgrad,
    x0: np.ndarray,
    iters: int = 200,
    mu: float = 0.5,
) -> np.ndarray:
    """Prox bundle: x_{k+1} = argmin_x { max_i [f_i + g_i.(x-x_i)] + mu/2 ||x-x_k||^2 }.

    Recast as a QP in (x, t): minimize t + (mu/2)||x - x_k||^2 subject to
    t >= f_i + g_i.(x - x_i), i.e. g_i.x - t <= g_i.x_i - f_i.
    """
    x = x0.copy()
    cuts_a: list[np.ndarray] = []
    cuts_b: list[float] = []
    for _ in range(iters):
        fx = f(x)
        g = subgrad(x)
        cuts_a.append(g)
        cuts_b.append(float(g @ x - fx))
        n = len(x)
        m = len(cuts_a)
        a = np.zeros((m, n + 1))
        a[:, :n] = np.array(cuts_a)
        a[:, n] = -1.0
        b = np.array(cuts_b)
        q = mu * np.diag(np.concatenate([np.ones(n), [0.0]]))
        c = np.concatenate([np.zeros(n), [1.0]])
        # shift prox center: mu/2 ||x - x_k||^2 -> linear term -mu x_k
        c[:n] = -mu * x
        z = qp_solve(q, c, a, b)
        x_new = z[:n]
        if np.linalg.norm(x_new - x) < 1e-9:
            x = x_new
            break
        x = x_new
        if len(cuts_a) > 60:
            cuts_a.pop(0)
            cuts_b.pop(0)
    return x


def _bench_bundle_method(seed: int = 0) -> float:
    checks = []
    # f(x) = max_i (a_i . x + b_i) piecewise linear -> LP-like min
    a = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [1.0, 1.0]])
    b = np.array([0.0, 0.0, 0.0, 0.0, -1.0])

    # min of max(a.x + b): symmetric -> optimum at origin with value 0
    def f(x: np.ndarray) -> float:
        return float(np.max(a @ x + b))

    def subgrad(x: np.ndarray) -> np.ndarray:
        return np.asarray(a[int(np.argmax(a @ x + b))])

    x = bundle_minimize(f, subgrad, np.array([3.0, -2.0]), iters=40)
    checks.append(f(x) < 0.1)
    # |x| + |y| minimization: optimum origin
    f2 = lambda v: float(np.abs(v).sum())  # noqa: E731

    def sg2(v: np.ndarray) -> np.ndarray:
        return np.asarray(np.sign(v) + (v == 0) * 0.3)

    x2 = bundle_minimize(f2, sg2, np.array([2.0, 2.0]), iters=80)
    checks.append(f2(x2) < 0.3)
    # max(-x, x - 2): min at x in [0,2] boundary value 0 region... f=|x-1|-1
    f3 = lambda v: float(np.abs(v[0] - 1) - 1.0)  # noqa: E731

    def sg3(v: np.ndarray) -> np.ndarray:
        return np.asarray([np.sign(v[0] - 1.0)])

    x3 = bundle_minimize(f3, sg3, np.array([5.0]), iters=60)
    checks.append(abs(f3(x3) - (-1.0)) < 0.2)
    # monotonic descent of best value for case 1
    checks.append(f(x) <= f(np.array([3.0, -2.0])))
    return float(sum(checks) / len(checks))


def bench_bundle_method(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bundle_method": _bench_bundle_method(seed)}
