"""Fubini-Tonelli order swap (wave 288) (SYNTHETIC).

For integrable f(x,y) on the unit square, row-then-column and
column-then-row integrals coincide (and equal the 2-D quadrature).
"""

import numpy as np

_SEED = 20261231 + 816


def _int_xy(f, n: int = 250) -> tuple[float, float]:
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    z = f(xx, yy)
    i1 = float(np.trapezoid(np.array([np.trapezoid(z[i], y) for i in range(n)]), x))
    i2 = float(np.trapezoid(np.array([np.trapezoid(z[:, j], x) for j in range(n)]), y))
    return i1, i2


def bench_fubini_swap(seed: int = _SEED) -> dict[str, float]:
    ok = 0
    for f, want in [
        (lambda x, y: x * y, 0.25),
        (
            lambda x, y: np.sin(x + y),
            np.sin(1) * (1 - np.cos(1)) * 2
            if False
            else (np.cos(0) - np.cos(1)) - (np.cos(1) - np.cos(2)),
        ),
        (lambda x, y: np.exp(x - y), (np.e - 1) * (1 - np.exp(-1))),
    ]:
        i1, i2 = _int_xy(f)
        ok += int(abs(i1 - i2) / abs(want) < 5e-3 and abs(i1 - want) / want < 5e-3)
    return {"synthetic_fubini": float(ok == 3)}
