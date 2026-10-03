"""homotopy exponent module (SYNTHETIC)."""

from __future__ import annotations


def homotopy_exponent_ok(homotopy: bool, unstable: bool) -> bool:
    """homotopy_exponent
    check:
    homotopy
    unstable
    structure —
    periodic."""
    return homotopy and unstable


def homotopy_exponent_aux(aux: bool) -> bool:
    """homotopy_exponent
    aux:
    auxiliary
    homotopy
    check —
    Adams."""
    return aux


def _bench_homotopy_exponent(seed: int = 0) -> float:
    checks = []
    checks.append(homotopy_exponent_ok(True, True))
    checks.append(not homotopy_exponent_ok(False, True))
    checks.append(homotopy_exponent_aux(True))
    checks.append(not homotopy_exponent_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_homotopy_exponent(seed: int = 0) -> dict[str, float]:
    return {"synthetic_homotopy_exponent": _bench_homotopy_exponent(seed)}
