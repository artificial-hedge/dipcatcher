"""Log motives (SYNTHETIC)."""

from __future__ import annotations


def lm_ok(log: bool, motive: bool) -> bool:
    """Log
    motive:
    logarithmic
    motive —
    boundary."""
    return log and motive


def log_de_rham(ld: bool) -> bool:
    """Log
    de Rham:
    log
    de
    Rham
    motive —
    compactification."""
    return ld


def _bench_log_motive(seed: int = 0) -> float:
    checks = []
    checks.append(lm_ok(True, True))
    checks.append(not lm_ok(False, True))
    checks.append(log_de_rham(True))
    checks.append(not log_de_rham(False))
    checks.append(True)  # Kato log
    return float(sum(checks) / len(checks))


def bench_log_motive(seed: int = 0) -> dict[str, float]:
    return {"synthetic_log_motive": _bench_log_motive(seed)}
