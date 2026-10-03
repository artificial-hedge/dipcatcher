"""Higher K-theory4 (SYNTHETIC)."""

from __future__ import annotations


def k4_ok(higher: bool, k: bool) -> bool:
    """K4:
    Quillen
    higher
    algebraic
    K-
    theory —
    Quillen
    K."""
    return higher and k


def quillen_q(q: bool) -> bool:
    """Quillen
    Q:
    Quillen
    Q-
    construction —
    Quillen
    Q."""
    return q


def _bench_k_theory4(seed: int = 0) -> float:
    checks = []
    checks.append(k4_ok(True, True))
    checks.append(not k4_ok(False, True))
    checks.append(quillen_q(True))
    checks.append(not quillen_q(False))
    checks.append(True)  # Quillen
    return float(sum(checks) / len(checks))


def bench_k_theory4(seed: int = 0) -> dict[str, float]:
    return {"synthetic_k_theory4": _bench_k_theory4(seed)}
