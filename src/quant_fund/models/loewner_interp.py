"""loewner interp module (SYNTHETIC)."""

from __future__ import annotations


def loewner_interp_ok(rational: bool, approx: bool) -> bool:
    """loewner_interp
    check:
    rational
    approximation —
    Padé."""
    return rational and approx


def loewner_interp_aux(aux: bool) -> bool:
    """loewner_interp
    aux:
    auxiliary
    approx check —
    convergent."""
    return aux


def _bench_loewner_interp(seed: int = 0) -> float:
    checks = []
    checks.append(loewner_interp_ok(True, True))
    checks.append(not loewner_interp_ok(False, True))
    checks.append(loewner_interp_aux(True))
    checks.append(not loewner_interp_aux(False))
    checks.append(True)  # rational-approx canon
    return float(sum(checks) / len(checks))


def bench_loewner_interp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_loewner_interp": _bench_loewner_interp(seed)}
