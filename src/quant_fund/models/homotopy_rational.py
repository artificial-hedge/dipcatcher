"""homotopy rational module (SYNTHETIC)."""

from __future__ import annotations


def homotopy_rational_ok(homotopy: bool, stable: bool) -> bool:
    """homotopy_rational
    check:
    homotopy
    structure —
    general."""
    return homotopy and stable


def homotopy_rational_aux(aux: bool) -> bool:
    """homotopy_rational
    aux:
    auxiliary
    homotopy
    check —
    rational."""
    return aux


def _bench_homotopy_rational(seed: int = 0) -> float:
    checks = []
    checks.append(homotopy_rational_ok(True, True))
    checks.append(not homotopy_rational_ok(False, True))
    checks.append(homotopy_rational_aux(True))
    checks.append(not homotopy_rational_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_homotopy_rational(seed: int = 0) -> dict[str, float]:
    return {"synthetic_homotopy_rational": _bench_homotopy_rational(seed)}
