"""Two-stage stochastic program via the L-shaped method.

First stage: build scalar capacity x in [0,14] at cost 3x. Second
stage under demand scenario omega: recourse LP max 5y1 + 4y2 s.t.
y_i <= omega_i, y1 + y2 <= x. Expected recourse Q(x) is concave, so
Kelley/L-shaped optimality cuts z <= Q(x_hat) + g*(x - x_hat) bound it
from above; the 1-D master picks the argmax of -3x + min-cut. Bench
compares the converged objective vs a dense grid optimum.
"""

import numpy as np
from scipy.optimize import linprog

_P = np.array([5.0, 4.0])
_SCEN = np.array([[8.0, 6.0], [4.0, 10.0], [10.0, 3.0], [6.0, 6.0]])
_PROB = np.array([0.25, 0.25, 0.25, 0.25])
_X_CAP = 14.0
_BUILD = 3.0


def _recourse(x: float) -> tuple[float, float]:
    """E[Q(x)] and a subgradient of E[Q] at x (capacity-row dual)."""
    tot = 0.0
    subg = 0.0
    for w, pr in zip(_SCEN, _PROB, strict=True):
        res = linprog(
            -_P,
            A_ub=np.vstack([np.ones((1, 2)), np.eye(2)]),
            b_ub=np.array([x, w[0], w[1]]),
            bounds=(0, None),
            method="highs",
        )
        q = -float(res.fun)
        tot += pr * q
        m = np.asarray(res.ineqlin.marginals)
        subg += pr * (-m[0])
    return tot, subg


def _lshaped(iters: int = 60) -> tuple[float, int]:
    cuts: list[tuple[float, float]] = []  # z <= intercept + slope*x
    x = 5.0
    for _ in range(iters):
        q, g = _recourse(x)
        cuts.append((q - g * x, g))
        xs = np.linspace(0.0, _X_CAP, 1401)
        z = np.minimum.reduce([a + b * xs for a, b in cuts])
        vals = -_BUILD * xs + z
        x_new = float(xs[int(np.argmax(vals))])
        if abs(x_new - x) < 1e-6:
            x = x_new
            break
        x = x_new
    obj = -_BUILD * x + _recourse(x)[0]
    return float(obj), len(cuts)


def _grid() -> float:
    best = -np.inf
    for x in np.linspace(0.0, _X_CAP, 1401):
        best = max(best, -_BUILD * x + _recourse(x)[0])
    return float(best)


def bench_two_stage_lshaped(seed: int = 5501) -> dict[str, float]:
    obj, n_cuts = _lshaped()
    truth = _grid()
    return {
        "synthetic_ls_obj": obj,
        "synthetic_ls_truth": truth,
        "synthetic_ls_gap": abs(obj - truth),
        "synthetic_ls_cuts": float(n_cuts),
    }
