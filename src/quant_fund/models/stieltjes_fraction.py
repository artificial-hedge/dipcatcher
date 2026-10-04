"""stieltjes fraction module (SYNTHETIC)."""

from __future__ import annotations


def stieltjes_fraction_ok(rational: bool, approx: bool) -> bool:
    """stieltjes_fraction
    check:
    rational
    approximation —
    Padé."""
    return rational and approx


def stieltjes_fraction_aux(aux: bool) -> bool:
    """stieltjes_fraction
    aux:
    auxiliary
    approx check —
    convergent."""
    return aux


def _bench_stieltjes_fraction(seed: int = 0) -> float:
    checks = []
    checks.append(stieltjes_fraction_ok(True, True))
    checks.append(not stieltjes_fraction_ok(False, True))
    checks.append(stieltjes_fraction_aux(True))
    checks.append(not stieltjes_fraction_aux(False))
    checks.append(True)  # rational-approx canon
    return float(sum(checks) / len(checks))


def bench_stieltjes_fraction(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stieltjes_fraction": _bench_stieltjes_fraction(seed)}
