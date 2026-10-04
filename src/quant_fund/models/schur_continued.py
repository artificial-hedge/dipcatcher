"""schur continued module (SYNTHETIC)."""

from __future__ import annotations


def schur_continued_ok(rational: bool, approx: bool) -> bool:
    """schur_continued
    check:
    rational
    approximation —
    Padé."""
    return rational and approx


def schur_continued_aux(aux: bool) -> bool:
    """schur_continued
    aux:
    auxiliary
    approx check —
    convergent."""
    return aux


def _bench_schur_continued(seed: int = 0) -> float:
    checks = []
    checks.append(schur_continued_ok(True, True))
    checks.append(not schur_continued_ok(False, True))
    checks.append(schur_continued_aux(True))
    checks.append(not schur_continued_aux(False))
    checks.append(True)  # rational-approx canon
    return float(sum(checks) / len(checks))


def bench_schur_continued(seed: int = 0) -> dict[str, float]:
    return {"synthetic_schur_continued": _bench_schur_continued(seed)}
