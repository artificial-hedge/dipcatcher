"""Strang canon: Strang operator splitting for u_t = A u + B u —
second-order symmetric splitting A(h/2)·B(h)·A(h/2) on a
scalar 2x2 commutator test with a known exact matrix exponential,
plus a Schrödinger-style kinetic/potential split. All SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

FloatArray = NDArray[np.float64]


def strang_split(
    a_flow,
    b_flow,
    y0: FloatArray,
    t: FloatArray,
) -> FloatArray:
    """Strang splitting: A(dt/2) → B(dt) → A(dt/2).

    a_flow(y, dt) / b_flow(y, dt) advance each sub-flow exactly.
    """
    out = np.empty((len(t), y0.size))
    out[0] = y0
    y = y0.copy()
    for i in range(1, len(t)):
        h = t[i] - t[i - 1]
        y = a_flow(b_flow(a_flow(y, h / 2), h), h / 2)
        out[i] = y
    return out


def lie_split(a_flow, b_flow, y0: FloatArray, t: FloatArray) -> FloatArray:
    """First-order Lie–Trotter A(h)·B(h) baseline."""
    out = np.empty((len(t), y0.size))
    out[0] = y0
    y = y0.copy()
    for i in range(1, len(t)):
        h = t[i] - t[i - 1]
        y = b_flow(a_flow(y, h), h)
        out[i] = y
    return out


def bench_strang(seed: int = 20261231) -> dict[str, float]:
    """Split-error orders on noncommuting 2x2 generators."""
    out: dict[str, float] = {}
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((2, 2)) * 0.5
    B = np.array([[0.0, 1.3], [-0.9, 0.0]])  # rotation-ish potential
    y0 = np.array([1.0, -0.4])

    a_flow = lambda y, h: expm(A * h) @ y  # noqa: E731
    b_flow = lambda y, h: expm(B * h) @ y  # noqa: E731

    errs_s = {}
    errs_l = {}
    for n in (100, 200):
        t = np.linspace(0.0, 1.0, n + 1)
        exact = expm((A + B) * 1.0) @ y0
        ys = strang_split(a_flow, b_flow, y0, t)
        yl = lie_split(a_flow, b_flow, y0, t)
        errs_s[n] = float(np.abs(ys[-1] - exact).max())
        errs_l[n] = float(np.abs(yl[-1] - exact).max())
    out["synthetic_strang_order"] = float(np.log2(errs_s[100] / errs_s[200]))
    out["synthetic_lie_order"] = float(np.log2(errs_l[100] / errs_l[200]))
    out["synthetic_strang_err"] = errs_s[200]
    out["synthetic_lie_err"] = errs_l[200]
    out["synthetic_strang_advantage"] = errs_l[200] / max(errs_s[200], 1e-300)
    # commutator norm: the obstruction the splitting must overcome
    comm = A @ B - B @ A
    out["synthetic_commutator_norm"] = float(np.abs(comm).max())
    return out
