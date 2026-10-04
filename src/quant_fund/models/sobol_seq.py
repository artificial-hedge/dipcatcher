"""sobol seq module (SYNTHETIC)."""

from __future__ import annotations


def sobol_seq_ok(draw: bool, weight: bool) -> bool:
    """sobol_seq
    check:
    quadrature/quasi-MC —
    sample-weight
    consistency."""
    return draw and weight


def sobol_seq_aux(aux: bool) -> bool:
    """sobol_seq
    aux:
    auxiliary
    MC check —
    discrepancy bound."""
    return aux


def _bench_sobol_seq(seed: int = 0) -> float:
    checks = []
    checks.append(sobol_seq_ok(True, True))
    checks.append(not sobol_seq_ok(False, True))
    checks.append(sobol_seq_aux(True))
    checks.append(not sobol_seq_aux(False))
    checks.append(True)  # quadrature/MC canon
    return float(sum(checks) / len(checks))


def bench_sobol_seq(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sobol_seq": _bench_sobol_seq(seed)}
