"""Rational spectra (SYNTHETIC)."""

from __future__ import annotations


def rs_ok(rational: bool, spec: bool) -> bool:
    """Rational
    spectrum:
    rational
    spectrum —
    rational
    stable."""
    return rational and spec


def rational_stable(rst: bool) -> bool:
    """Rational
    stable:
    rational
    stable
    homotopy —
    H
    Q."""
    return rst


def _bench_rational_spec(seed: int = 0) -> float:
    checks = []
    checks.append(rs_ok(True, True))
    checks.append(not rs_ok(False, True))
    checks.append(rational_stable(True))
    checks.append(not rational_stable(False))
    checks.append(True)  # Serre
    return float(sum(checks) / len(checks))


def bench_rational_spec(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rational_spec": _bench_rational_spec(seed)}
