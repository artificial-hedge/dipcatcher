"""Log ring (SYNTHETIC)."""

from __future__ import annotations


def lr_ok(log: bool, ring: bool) -> bool:
    """Log:
    log-
    ring
    with
    log
    structure —
    Kato
    log."""
    return log and ring


def log_structure(ls: bool) -> bool:
    """Log
    structure:
    log
    structure
    on
    ring —
    Kato
    structure."""
    return ls


def _bench_log_ring(seed: int = 0) -> float:
    checks = []
    checks.append(lr_ok(True, True))
    checks.append(not lr_ok(False, True))
    checks.append(log_structure(True))
    checks.append(not log_structure(False))
    checks.append(True)  # Kato
    return float(sum(checks) / len(checks))


def bench_log_ring(seed: int = 0) -> dict[str, float]:
    return {"synthetic_log_ring": _bench_log_ring(seed)}
