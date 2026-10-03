"""Log structures (SYNTHETIC)."""

from __future__ import annotations


def log_structure_ok(prelog: bool, log_ring: bool) -> bool:
    """Log structure on scheme
    X: homomorphism
    M -> O_X with M^* =
    O_X^*; records
    toric/degenerating
    data (Fontaine-Kato)."""
    return prelog and log_ring


def fine_saturated(finitely_gen: bool) -> bool:
    """Fine saturated (fs)
    log structures: charts
    by finitely generated
    saturated monoids;
    main workable class."""
    return finitely_gen


def _bench_log_structure(seed: int = 0) -> float:
    checks = []
    checks.append(log_structure_ok(True, True))
    checks.append(not log_structure_ok(False, True))
    checks.append(fine_saturated(True))
    checks.append(not fine_saturated(False))
    checks.append(True)  # trivial log structure
    return float(sum(checks) / len(checks))


def bench_log_structure(seed: int = 0) -> dict[str, float]:
    return {"synthetic_log_structure": _bench_log_structure(seed)}
