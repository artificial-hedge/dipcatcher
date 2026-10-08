"""Gassmann fluid substitution (1951): saturated bulk modulus exchange (SYNTHETIC).

K_sat = K_dry + (1 - K_dry/K_m)^2 / (phi/K_fl + (1-phi)/K_m - K_dry/K_m^2);
shear modulus is fluid-independent.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 933


def gassmann_forward(k_dry: float, k_m: float, k_fl: float, phi: float) -> float:
    s = phi / k_fl + (1.0 - phi) / k_m
    return k_dry + (1.0 - k_dry / k_m) ** 2 / (s - k_dry / k_m**2)


def gassmann_dry(k_sat: float, k_m: float, k_fl: float, phi: float, tol: float = 1e-12) -> float:
    """Newton inversion for K_dry; monotone in the physical range."""

    def f(kd: float) -> float:
        return gassmann_forward(kd, k_m, k_fl, phi) - k_sat

    lo, hi = 0.0, k_m * (1.0 - 1e-9)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol * k_m:
            break
    return 0.5 * (lo + hi)


def gassmann_sub(
    k_sat1: float,
    mu: float,
    k_m: float,
    k_fl1: float,
    k_fl2: float,
    phi: float,
) -> tuple[float, float]:
    """Substitute pore fluid 1 -> 2; returns (K_sat2, mu)."""
    k_dry = gassmann_dry(k_sat1, k_m, k_fl1, phi)
    return gassmann_forward(k_dry, k_m, k_fl2, phi), mu


def bench_gassmann_sub(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    k_m = 37.0  # quartz GPa
    k_dry_true = float(rng.uniform(4.0, 10.0))
    mu = float(rng.uniform(3.0, 8.0))
    phi = float(rng.uniform(0.15, 0.30))
    k_fl1 = 2.2  # brine
    k_fl2 = 0.14  # gas-like
    k_fl3 = 2.9  # stiffer brine
    k_sat1 = gassmann_forward(k_dry_true, k_m, k_fl1, phi)
    k_dry_est = gassmann_dry(k_sat1, k_m, k_fl1, phi)
    rt_err = abs(k_dry_est - k_dry_true) / k_dry_true
    k_rt = gassmann_forward(k_dry_est, k_m, k_fl1, phi)
    rt_sat_err = abs(k_rt - k_sat1) / k_sat1
    k_gas, mu2 = gassmann_sub(k_sat1, mu, k_m, k_fl1, k_fl2, phi)
    k_stiff, _ = gassmann_sub(k_sat1, mu, k_m, k_fl1, k_fl3, phi)
    k_dry_limit = gassmann_forward(k_dry_true, k_m, 1e-6, phi)
    checks = [
        rt_err < 1e-8,
        rt_sat_err < 1e-8,
        mu2 == mu,
        k_gas < k_sat1,
        k_stiff > k_sat1,
        k_dry_limit < gassmann_forward(k_dry_true, k_m, k_fl1, phi),
        abs(k_dry_limit - k_dry_true) / k_dry_true < 0.05,
    ]
    return {"synthetic_gassmann_sub": float(np.mean(checks))}
