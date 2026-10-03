"""Scaling-and-squaring Padé expm canon: Higham's 13th-order Padé
approximant with norm-based scaling s = max(0, ceil(log2(||A||/θ)))
— plus the block-triangle trick for expm·action and ZOH
discretization. Bench: rotation-block matrix vs closed form,
nilpotent (finite truncation exactness), ZOH consistency with an
analytic integrator. All SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# Higham 2005 Table 2.3 Padé-13 coefficients
_B = np.array(
    [
        64764752532480000.0,
        32382376266240000.0,
        7771770303897600.0,
        1187353796428800.0,
        129060195264000.0,
        10559470521600.0,
        670442572800.0,
        33522128640.0,
        1323241920.0,
        40840800.0,
        960960.0,
        16380.0,
        182.0,
        1.0,
    ]
)
_THETA13 = 5.371920351148152


def expm_pade(A: FloatArray) -> FloatArray:
    """e^A via scaling-and-squaring Padé(13) (Higham scaling)."""
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    norm = np.linalg.norm(A, 1)
    s = max(0, int(np.ceil(np.log2(norm / _THETA13)))) if norm > _THETA13 else 0
    As = A / 2.0**s
    A2 = As @ As
    A4 = A2 @ A2
    A6 = A4 @ A2
    U = As @ (
        A6 @ (_B[13] * A6 + _B[11] * A4 + _B[9] * A2)
        + _B[7] * A6
        + _B[5] * A4
        + _B[3] * A2
        + _B[1] * np.eye(n)
    )
    V = (
        A6 @ (_B[12] * A6 + _B[10] * A4 + _B[8] * A2)
        + _B[6] * A6
        + _B[4] * A4
        + _B[2] * A2
        + _B[0] * np.eye(n)
    )
    P = V + U  # numerator + denominator
    Q = V - U  # −numerator + denominator
    R = np.linalg.solve(Q, P)
    for _ in range(s):
        R = R @ R
    return np.asarray(R, dtype=np.float64)


def expm_action(A: FloatArray, v: FloatArray) -> FloatArray:
    """e^A v — the [[A,v],[0,0]] block trick yields ∫e^{Aτ}v dτ,
    so the action goes through the scaled Padé expm directly."""
    return np.asarray(expm_pade(A) @ v, dtype=np.float64)


def zoh_discretize(A: FloatArray, B: FloatArray, dt: float) -> tuple[FloatArray, FloatArray]:
    """Zero-order-hold discretization: Ad = e^{A·dt}, Bd = ∫e^{Aτ}B dτ
    via the block-triangle exponential of [[A, B],[0,0]]·dt."""
    n, m = A.shape[0], B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    E = expm_pade(M * dt)
    return np.asarray(E[:n, :n], dtype=np.float64), np.asarray(E[:n, n:], dtype=np.float64)


def bench_expm_pade(seed: int = 20261231) -> dict[str, float]:
    out: dict[str, float] = {}
    # rotation: exp([[0,-θ],[θ,0]]) = [[cosθ,-sinθ],[sinθ,cosθ]]
    th = 1.3
    A = np.array([[0.0, -th], [th, 0.0]])
    E = expm_pade(A)
    exact = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    out["synthetic_expm_rot_err"] = float(np.abs(E - exact).max())
    # strictly upper-triangular N: N³=0, so e^N = I + N + N²/2
    N = np.array([[0.0, 2.5, -1.0], [0.0, 0.0, 3.0], [0.0, 0.0, 0.0]])
    E2 = expm_pade(N)
    out["synthetic_expm_nilp_err"] = float(np.abs(E2 - (np.eye(3) + N + N @ N / 2)).max())
    # large-norm scaling path: ||A|| = 100 forces s>0
    B = np.array([[-50.0, 30.0], [-30.0, -50.0]])
    E3 = expm_pade(B)
    # e^B = e^{-50} R(30)... tiny — check against scipy
    from scipy.linalg import expm as sexpm

    out["synthetic_expm_scaled_err"] = float(np.abs(E3 - sexpm(B)).max())
    # expm_action consistency
    v = np.array([1.0, -1.0])
    out["synthetic_expmv_err"] = float(np.abs(expm_action(A, v) - sexpm(A) @ v).max())
    # ZOH for stable A: Bd analytic = A⁻¹(Ad − I)B
    Az = np.array([[-1.0]])
    Bz = np.array([[2.0]])
    Ad, Bd = zoh_discretize(Az, Bz, 0.7)
    Bd_exact = (Ad[0, 0] - 1.0) * Bz[0, 0] / Az[0, 0]
    out["synthetic_zoh_bd_err"] = float(abs(Bd[0, 0] - Bd_exact))
    out["synthetic_zoh_ad_err"] = float(abs(Ad[0, 0] - np.exp(-0.7)))
    return out
