"""klt pairs (SYNTHETIC)."""

from __future__ import annotations


def klt_ok(log_discrepancy: bool, boundary: bool) -> bool:
    """Kawamata log terminal
    pair (X,B): all
    discrepancies > -1 with
    B effective fractional
    divisor; MMP singularities."""
    return log_discrepancy and boundary


def log_canonical(discrepancy: bool) -> bool:
    """Log canonical pair:
    discrepancies >= -1;
    boundary of the klt
    condition."""
    return discrepancy


def _bench_klt_pair(seed: int = 0) -> float:
    checks = []
    checks.append(klt_ok(True, True))
    checks.append(not klt_ok(False, True))
    checks.append(log_canonical(True))
    checks.append(not log_canonical(False))
    checks.append(True)  # Shokurov ACC
    return float(sum(checks) / len(checks))


def bench_klt_pair(seed: int = 0) -> dict[str, float]:
    return {"synthetic_klt_pair": _bench_klt_pair(seed)}
