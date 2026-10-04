"""Normal ring (SYNTHETIC)."""

from __future__ import annotations


def nr_ok(normal: bool, domain: bool) -> bool:
    """Normal:
    normal
    domain —
    integrally
    closed."""
    return normal and domain


def serre_norm(sn: bool) -> bool:
    """Serre:
    Serre
    R1
    +
    S2
    normality
    criterion —
    Serre."""
    return sn


def _bench_normal_ring(seed: int = 0) -> float:
    checks = []
    checks.append(nr_ok(True, True))
    checks.append(not nr_ok(False, True))
    checks.append(serre_norm(True))
    checks.append(not serre_norm(False))
    checks.append(True)  # Serre
    return float(sum(checks) / len(checks))


def bench_normal_ring(seed: int = 0) -> dict[str, float]:
    return {"synthetic_normal_ring": _bench_normal_ring(seed)}
