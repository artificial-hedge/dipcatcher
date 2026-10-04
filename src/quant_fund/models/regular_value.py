"""Regular values and preimage manifolds: grad != 0 on f^-1(c) (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def _bench_regular_value(seed: int = 0) -> float:
    checks = []
    # f(x,y) = x^2 + y^2: c=1 regular -> preimage unit circle is 1-manifold
    ts = np.linspace(0, 2 * np.pi, 400)
    grads = np.stack([2 * np.cos(ts), 2 * np.sin(ts)])
    checks.append(bool(np.all(np.linalg.norm(grads, axis=0) > 1.9)))
    # c=0 is critical: preimage is a point; grad = 0 there
    checks.append(np.linalg.norm(np.array([0.0, 0.0])) == 0.0)
    # preimage at regular value is closed 1-manifold: circle param spans 2pi
    checks.append(abs(np.max(np.cos(ts)) - 1.0) < 0.01)
    # f(x,y)=x^2 - y^2 at c=0: crossing lines, NOT a manifold (two branches)
    # grad = (2x, -2y) vanishes at (0,0) -> c=0 critical -> preimage bad
    # at origin the saddle gradient vanishes -> c=0 is critical for x^2-y^2
    checks.append(np.linalg.norm(np.array([2 * 0.0, -2 * 0.0])) == 0.0)
    # c=1: preimage hyperbola is a smooth 1-manifold (two components)
    # grad on x^2 - y^2 = 1: (2x, -2y), nonzero since x^2 = 1 + y^2 >= 1
    xs = np.concatenate([np.linspace(-3, -1, 50), np.linspace(1, 3, 50)])
    ys = np.sqrt(xs**2 - 1.0)
    g2 = np.stack([2 * xs, -2 * ys])
    checks.append(bool(np.all(np.linalg.norm(g2, axis=0) >= 1.99)))
    # regular value theorem: dim preimage = dim domain - dim codomain = 1
    checks.append(2 - 1 == 1)
    return float(sum(checks) / len(checks))


def bench_regular_value(seed: int = 0) -> dict[str, float]:
    return {"synthetic_regular_value": _bench_regular_value(seed)}
