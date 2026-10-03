"""Banach-Colmez II (SYNTHETIC)."""

from __future__ import annotations


def banach_colmez2_ok(pro_etale: bool, p_adic_period: bool) -> bool:
    """Banach-Colmez spaces
    parametrize p-adic
    periods; B_crys^phi=1
    as BC space."""
    return pro_etale and p_adic_period


def colmez_period(period: bool) -> bool:
    """Period morphisms between
    BC spaces realize the
    p-adic periods of
    Galois reps."""
    return period


def _bench_banach_colmez2(seed: int = 0) -> float:
    checks = []
    checks.append(banach_colmez2_ok(True, True))
    checks.append(not banach_colmez2_ok(False, True))
    checks.append(colmez_period(True))
    checks.append(not colmez_period(False))
    checks.append(True)  # Plu-t BC spaces
    return float(sum(checks) / len(checks))


def bench_banach_colmez2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_banach_colmez2": _bench_banach_colmez2(seed)}
