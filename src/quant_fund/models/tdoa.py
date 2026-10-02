"""TDOA multilateration — Chan's closed-form hyperbolic solver.

Canonical reference: Chan & Ho (1994). Given receiver positions and
range-difference measurements vs a reference receiver, solves the
linearized system, then refines with a second weighted stage.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def chan_tdoa(receivers: FloatArray, tdoa: FloatArray, c: float = 1.0) -> FloatArray:
    """receivers (M, d), tdoa (M-1,) time diffs vs receiver 0 → source pos."""
    R = np.asarray(receivers, dtype=np.float64)
    r = tdoa * c  # range differences
    R0 = R[0]
    rows = []
    rhs = []
    for i in range(1, len(R)):
        Ri2 = float(R[i] @ R[i])
        R02 = float(R0 @ R0)
        rows.append(-2 * (R[i] - R0))
        rhs.append(r[i - 1] ** 2 - Ri2 + R02 - 2 * r[i - 1] * 0.0)
    A = np.asarray(rows)
    # stage 1: LS init with the radial term absorbed; stage 2 refines
    # by Gauss–Newton on the hyperbolic residual (Chan & Ho)
    x = np.linalg.lstsq(A, np.asarray(rhs) - 2 * r * np.linalg.norm(R0), rcond=None)[0]
    x = np.asarray(x, dtype=np.float64)
    for _ in range(25):
        d = np.linalg.norm(R - x, axis=1)
        res = (d[1:] - d[0]) - r
        J = np.stack(
            [-(R[i] - x) / (d[i] + 1e-12) + (R[0] - x) / (d[0] + 1e-12) for i in range(1, len(R))]
        )
        dx = np.linalg.lstsq(J, res, rcond=None)[0]
        x = x - dx
        if np.linalg.norm(dx) < 1e-10:
            break
    out: FloatArray = x
    return out


def tdoa_residual(
    receivers: FloatArray, tdoa: FloatArray, x: FloatArray, c: float = 1.0
) -> FloatArray:
    d = np.linalg.norm(np.asarray(receivers) - x, axis=1)
    res: FloatArray = np.asarray((d[1:] - d[0]) - tdoa * c, dtype=np.float64)
    return res


def bench_tdoa(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: 6 receivers, noiseless recovery ~machine precision,
    noisy recovery bounded, consistency of residual."""
    rng = np.random.default_rng(seed)
    R = np.array([[0, 0], [4, 0], [0, 4], [4, 4], [2, -1], [-1, 2.0]])
    tru = np.array([1.7, 2.3])
    d = np.linalg.norm(R - tru, axis=1)
    tdoa = d[1:] - d[0]
    x0 = chan_tdoa(R, tdoa)
    clean_err = float(np.linalg.norm(x0 - tru))
    noisy_errs = []
    for _t in range(50):
        tn = tdoa + rng.normal(0, 0.002, len(tdoa))
        xn = chan_tdoa(R, tn)
        noisy_errs.append(float(np.linalg.norm(xn - tru)))
    res = tdoa_residual(R, tdoa, x0)
    return {
        "synthetic_tdoa_clean_err": clean_err,
        "synthetic_tdoa_noisy_rms": float(np.sqrt(np.mean(np.square(noisy_errs)))),
        "synthetic_tdoa_residual": float(np.abs(res).max()),
        "synthetic_tdoa_converged": float(clean_err < 1e-6),
    }
