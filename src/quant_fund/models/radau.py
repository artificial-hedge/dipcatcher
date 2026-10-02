"""Radau canon: Radau IIA implicit Runge–Kutta — the 2-stage
(order 3) and 3-stage (order 5) Radau collocation schemes with
Newton solves, on the stiff scalar test. L-stable: stays sane
where explicit methods die. All SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# Radau IIA 2-stage (order 3): nodes c = (1/3, 1)
_A2 = np.array([[5.0 / 12, -1.0 / 12], [3.0 / 4, 1.0 / 4]])
_B2 = np.array([3.0 / 4, 1.0 / 4])
_C2 = np.array([1.0 / 3, 1.0])

# Radau IIA 3-stage (order 5): nodes c = ((4-√6)/10, (4+√6)/10, 1)
_A3 = np.array(
    [
        [(88 - 7 * np.sqrt(6)) / 360, (296 - 169 * np.sqrt(6)) / 1800, (-2 + 3 * np.sqrt(6)) / 225],
        [(296 + 169 * np.sqrt(6)) / 1800, (88 + 7 * np.sqrt(6)) / 360, (-2 - 3 * np.sqrt(6)) / 225],
        [(16 - np.sqrt(6)) / 36, (16 + np.sqrt(6)) / 36, 1.0 / 9],
    ]
)
_B3 = np.array([(16 - np.sqrt(6)) / 36, (16 + np.sqrt(6)) / 36, 1.0 / 9])
_C3 = np.array([(4 - np.sqrt(6)) / 10, (4 + np.sqrt(6)) / 10, 1.0])


def radau_step(
    f,
    y: FloatArray,
    t: float,
    h: float,
    A: FloatArray,
    B: FloatArray,
    C: FloatArray,
    newton_iters: int = 12,
) -> FloatArray:
    """One Radau step by Newton on the collocation stage values
    Y_i = y + h·Σ_j A_ij·f(t+C_j·h, Y_j), then the Radau-IA-weight
    combination y + h·Σ_j B_j·f(Y_j)."""
    s = len(B)
    n = y.size
    eye = np.eye(s * n)

    def fvals(Ym: FloatArray) -> FloatArray:
        fv = np.empty_like(Ym)
        for i in range(s):
            fv[i] = f(t + C[i] * h, Ym[i])
        return fv

    def resid(Yv: FloatArray) -> FloatArray:
        Ym = Yv.reshape(s, n)
        fv = fvals(Ym)
        return (Ym - y[None, :] - h * (A @ fv)).ravel()

    Yv = np.tile(y, (s, 1)).astype(np.float64).ravel()
    for _ in range(newton_iters):
        r = resid(Yv)
        if np.abs(r).max() < 1e-12:
            break
        # block finite-difference Jacobian
        J = np.empty((s * n, s * n))
        for j in range(s * n):
            d = np.zeros(s * n)
            d[j] = 1e-7
            J[:, j] = (resid(Yv + d) - r) / 1e-7
        step = np.linalg.solve(J + 1e-14 * eye, r)
        Yv -= step
        if np.abs(step).max() < 1e-12:
            break
    Ym = Yv.reshape(s, n)
    fv = fvals(Ym)
    return y + h * (B[:, None] * fv).sum(axis=0)


def radau(
    f,
    y0: FloatArray,
    t: FloatArray,
    stages: int = 3,
) -> FloatArray:
    """Integrate dy/dt = f(t,y) with Radau IIA (`stages` = 2 or 3)."""
    A, B, C = (_A2, _B2, _C2) if stages == 2 else (_A3, _B3, _C3)
    out = np.empty((len(t), y0.size), dtype=np.float64)
    out[0] = y0
    for i in range(1, len(t)):
        h = t[i] - t[i - 1]
        out[i] = radau_step(f, out[i - 1], t[i - 1], h, A, B, C)
    return np.asarray(out, dtype=np.float64)


def bench_radau(seed: int = 20261231) -> dict[str, float]:
    """Radau IIA-3 order ≈5 on the smooth test; stiff λ=-200 demo."""
    out: dict[str, float] = {}

    def f(t: float, y: FloatArray) -> FloatArray:
        lam = -5.0
        return np.asarray(lam * (y - np.cos(t)) - np.sin(t), dtype=np.float64)

    errs = {}
    for n in (50, 100):
        t = np.linspace(0.0, 2.0, n + 1)
        y = radau(f, np.array([1.0]), t, stages=3)
        errs[n] = abs(y[-1, 0] - np.cos(2.0))
    out["synthetic_radau3_order"] = float(np.log2(errs[50] / errs[100]))
    out["synthetic_radau3_err"] = float(errs[100])

    errs2 = {}
    for n in (50, 100):
        t = np.linspace(0.0, 2.0, n + 1)
        y = radau(f, np.array([1.0]), t, stages=2)
        errs2[n] = abs(y[-1, 0] - np.cos(2.0))
    out["synthetic_radau2_order"] = float(np.log2(errs2[50] / errs2[100]))

    # very stiff: λ=-200, h=0.04 (hλ=-8): still stable, error modest
    def fs(t: float, y: FloatArray) -> FloatArray:
        return np.asarray(-200.0 * (y - np.cos(t)) - np.sin(t), dtype=np.float64)

    t = np.linspace(0.0, 1.0, 26)
    y = radau(fs, np.array([1.0]), t, stages=3)
    out["synthetic_radau3_stiff_err"] = float(abs(y[-1, 0] - np.cos(1.0)))
    out["synthetic_radau3_stiff_finite"] = float(np.isfinite(y).all())
    return out
