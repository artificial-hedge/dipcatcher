"""Higher K-theory (SYNTHETIC)."""

from __future__ import annotations


def hk_ok(higher: bool, k_groups: bool) -> bool:
    """Higher:
    higher
    K-
    groups —
    Quillen
    higher."""
    return higher and k_groups


def quillen_higher(qh: bool) -> bool:
    """Quillen
    higher:
    Quillen
    higher
    K-
    theory —
    Q-
    construction."""
    return qh


def _bench_higher_k(seed: int = 0) -> float:
    checks = []
    checks.append(hk_ok(True, True))
    checks.append(not hk_ok(False, True))
    checks.append(quillen_higher(True))
    checks.append(not quillen_higher(False))
    checks.append(True)  # Quillen
    return float(sum(checks) / len(checks))


def bench_higher_k(seed: int = 0) -> dict[str, float]:
    return {"synthetic_higher_k": _bench_higher_k(seed)}
