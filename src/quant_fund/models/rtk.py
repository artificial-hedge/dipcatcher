"""RTK double-difference carrier-phase positioning.

Single differences (rover − base) per satellite, then double
differences against a pivot satellite remove receiver and satellite
clock biases. Float solution by weighted LS over [dx; N] (position
correction + integer ambiguities); ambiguities fixed via LAMBDA and the
position re-solved conditional on the fixed integers.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.lambda_method import lambda_ils

FloatArray = NDArray[np.float64]

_L1 = 0.190293672798  # GPS L1 wavelength (m)


def dd_position(
    base_pos: FloatArray,
    rover_guess: FloatArray,
    sats: FloatArray,
    pr_base: FloatArray,
    pr_rover: FloatArray,
    cp_base: FloatArray,
    cp_rover: FloatArray,
    pivot: int = 0,
    sigma_pr: float = 0.3,
    sigma_cp: float = 0.003,
) -> tuple[FloatArray, FloatArray, float]:
    """Estimate rover ECEF position.

    Returns (rover_pos, N_fixed, ambiguity fix-test ratio).
    """
    sats = np.asarray(sats, dtype=np.float64)
    k = sats.shape[0]
    idx = [i for i in range(k) if i != pivot]
    los_r = (sats - rover_guess) / np.linalg.norm(sats - rover_guess, axis=1, keepdims=True)
    rho_rb = np.linalg.norm(sats - base_pos, axis=1)

    def dd_rho(x: FloatArray) -> FloatArray:
        rho_r = np.linalg.norm(sats - x, axis=1)
        out: FloatArray = (rho_r[idx] - rho_r[pivot]) - (rho_rb[idx] - rho_rb[pivot])
        return out

    m = len(idx)
    w_pr = 1.0 / sigma_pr**2
    w_cp = 1.0 / sigma_cp**2
    # observations
    dd_pr = (pr_rover[idx] - pr_rover[pivot]) - (pr_base[idx] - pr_base[pivot])
    dd_cp = (cp_rover[idx] - cp_rover[pivot]) - (cp_base[idx] - cp_base[pivot])
    y = np.concatenate([dd_pr - dd_rho(rover_guess), dd_cp - dd_rho(rover_guess)])
    # design: [dd_los | lam I] for phase rows; [dd_los | 0] for code
    # d(rho)/dx = -LOS
    dlos = -(los_r[idx] - los_r[pivot])
    G = np.zeros((2 * m, 3 + m))
    G[:m, :3] = dlos
    G[m:, :3] = dlos
    G[m:, 3:] = _L1 * np.eye(m)
    W = np.diag([w_pr] * m + [w_cp] * m)
    Q = np.linalg.inv(G.T @ W @ G)
    sol = Q @ (G.T @ W @ y)
    n_hat = sol[3:]
    Q_nn = Q[3:, 3:]
    n_fix, _, ratio = lambda_ils(n_hat, np.eye(m), np.linalg.inv(Q_nn))
    # re-solve dx with fixed N
    resid = y[m:] - _L1 * n_fix
    Gp = dlos
    Wp = np.eye(m) * w_cp
    dx = np.linalg.solve(Gp.T @ Wp @ Gp, Gp.T @ Wp @ resid)
    pos = rover_guess + dx
    return (
        np.asarray(pos, dtype=np.float64),
        np.asarray(n_fix, dtype=np.float64),
        float(ratio),
    )


def bench_rtk(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: 9 satellites, base/rover 400 m apart; float ambiguities
    fixed by LAMBDA, rover recovered to ~cm."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    # satellite constellation: varied az/el at ~20,200 km
    k = 9
    az = np.linspace(0, 2 * np.pi, k)[:-1] + 0.3
    el = np.linspace(np.deg2rad(15), np.deg2rad(75), k - 1)
    orb = 20_200e3
    re_ = 6_371e3
    sats = np.column_stack(
        [
            orb * np.cos(el) * np.cos(az),
            orb * np.cos(el) * np.sin(az),
            orb * np.sin(el) + re_,
        ]
    )
    pivot_sat = np.array([0.0, 0.0, orb + re_])
    sats = np.vstack([pivot_sat, sats])
    base = np.array([re_, 0.0, 0.0])
    true_rover = base + np.array([150.0, 200.0, 60.0])
    k2 = sats.shape[0]
    n_true = rng.integers(-20, 20, k2 - 1)
    pr_base = np.linalg.norm(sats - base, axis=1) + rng.normal(0, 0.3, k2)
    pr_rover = np.linalg.norm(sats - true_rover, axis=1) + rng.normal(0, 0.3, k2)
    cp_base = np.linalg.norm(sats - base, axis=1) + rng.normal(0, 0.003, k2)
    n_full = np.concatenate([[0], n_true]) + rng.integers(0, 1)
    cp_rover = np.linalg.norm(sats - true_rover, axis=1) + _L1 * n_full + rng.normal(0, 0.003, k2)
    guess = true_rover + rng.normal(0, 0.5, 3)
    pos, n_fix, ratio = dd_position(base, guess, sats, pr_base, pr_rover, cp_base, cp_rover)
    out["synthetic_rtk_pos_err_m"] = float(np.linalg.norm(pos - true_rover))
    out["synthetic_rtk_amb_err"] = float(np.abs(n_fix - n_true).max())
    out["synthetic_rtk_ratio"] = float(ratio)
    out["synthetic_rtk_fixed"] = float(out["synthetic_rtk_amb_err"] == 0)
    return out


if __name__ == "__main__":
    print(bench_rtk())
