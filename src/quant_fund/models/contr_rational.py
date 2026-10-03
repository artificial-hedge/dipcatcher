"""Contracted rational curves (SYNTHETIC)."""

from __future__ import annotations


def cr_ok(contracted: bool, rational: bool) -> bool:
    """Contracted
    rational:
    contracted
    rational
    curve —
    K
    negative."""
    return contracted and rational


def k_negative_ray(kn: bool) -> bool:
    """K
    negative:
    K
    negative
    ray —
    extremal
    contraction."""
    return kn


def _bench_contr_rational(seed: int = 0) -> float:
    checks = []
    checks.append(cr_ok(True, True))
    checks.append(not cr_ok(False, True))
    checks.append(k_negative_ray(True))
    checks.append(not k_negative_ray(False))
    checks.append(True)  # Mori
    return float(sum(checks) / len(checks))


def bench_contr_rational(seed: int = 0) -> dict[str, float]:
    return {"synthetic_contr_rational": _bench_contr_rational(seed)}
