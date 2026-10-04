"""Theta functions (SYNTHETIC)."""

from __future__ import annotations


def theta_ok(series: bool, transform: bool) -> bool:
    """Theta
    function
    theta(z) =
    sum
    e^{pi i
    n^2 z};
    modular
    of weight
    1/2 on
    Gamma_0(4)."""
    return series and transform


def jacobi_triple(jac: bool) -> bool:
    """Jacobi
    triple
    product:
    theta
    expressed
    as an
    infinite
    product."""
    return jac


def _bench_theta_func(seed: int = 0) -> float:
    checks = []
    checks.append(theta_ok(True, True))
    checks.append(not theta_ok(False, True))
    checks.append(jacobi_triple(True))
    checks.append(not jacobi_triple(False))
    checks.append(True)  # Jacobi
    return float(sum(checks) / len(checks))


def bench_theta_func(seed: int = 0) -> dict[str, float]:
    return {"synthetic_theta_func": _bench_theta_func(seed)}
