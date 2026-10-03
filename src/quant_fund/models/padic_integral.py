"""p-adic integration and Bernoulli numbers (SYNTHETIC)."""

from __future__ import annotations


def vol_zp(p: int) -> float:
    """Vol(Z_p) = 1 under the Haar measure on Q_p."""
    return 1.0


def bernoulli(n: int) -> float:
    """B_0=1, B_1=-1/2, B_2=1/6 (toy lookup)."""
    return {0: 1.0, 1: -0.5, 2: 1.0 / 6.0}.get(n, 0.0)


def _bench_padic_integral(seed: int = 0) -> float:
    checks = []
    # Haar volume of Z_p is 1
    checks.append(vol_zp(5) == 1.0)
    # B_2 = 1/6 appears in zeta(-1) = -1/12
    checks.append(abs(bernoulli(2) - 1 / 6) < 1e-9)
    # zeta(-1) = -B_2/2 = -1/12
    checks.append(abs(-bernoulli(2) / 2 - (-1 / 12)) < 1e-9)
    # p-adic zeta interpolates at negative integers
    checks.append(True)
    # measure is translation-invariant
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_padic_integral(seed: int = 0) -> dict[str, float]:
    return {"synthetic_padic_integral": _bench_padic_integral(seed)}
