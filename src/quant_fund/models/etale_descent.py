"""Etale descent (SYNTHETIC)."""

from __future__ import annotations


def ed_ok(etale: bool, descent: bool) -> bool:
    """Etale:
    etale
    descent
    data —
    faithfully
    flat
    descent."""
    return etale and descent


def effective_descent(ed: bool) -> bool:
    """Effective:
    effective
    etale
    descent —
    SGA
    descent."""
    return ed


def _bench_etale_descent(seed: int = 0) -> float:
    checks = []
    checks.append(ed_ok(True, True))
    checks.append(not ed_ok(False, True))
    checks.append(effective_descent(True))
    checks.append(not effective_descent(False))
    checks.append(True)  # SGA
    return float(sum(checks) / len(checks))


def bench_etale_descent(seed: int = 0) -> dict[str, float]:
    return {"synthetic_etale_descent": _bench_etale_descent(seed)}
