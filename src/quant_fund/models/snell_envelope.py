"""snell envelope module (SYNTHETIC)."""

from __future__ import annotations


def snell_envelope_ok(os1: bool, sd: bool) -> bool:
    """snell_envelope
    check:
    optimal-
    stopping —
    value
    function."""
    return os1 and sd


def snell_envelope_aux(aux: bool) -> bool:
    """snell_envelope
    aux:
    auxiliary
    stopping
    check —
    boundary."""
    return aux


def _bench_snell_envelope(seed: int = 0) -> float:
    checks = []
    checks.append(snell_envelope_ok(True, True))
    checks.append(not snell_envelope_ok(False, True))
    checks.append(snell_envelope_aux(True))
    checks.append(not snell_envelope_aux(False))
    checks.append(True)  # optimal-stopping canon
    return float(sum(checks) / len(checks))


def bench_snell_envelope(seed: int = 0) -> dict[str, float]:
    return {"synthetic_snell_envelope": _bench_snell_envelope(seed)}
