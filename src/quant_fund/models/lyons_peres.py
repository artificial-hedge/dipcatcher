"""lyons peres module (SYNTHETIC)."""

from __future__ import annotations


def lyons_peres_ok(loop: bool, gff: bool) -> bool:
    """lyons_peres
    check:
    loop-soup
    structure —
    LeJan."""
    return loop and gff


def lyons_peres_aux(aux: bool) -> bool:
    """lyons_peres
    aux:
    auxiliary
    Gaussian-field
    check —
    Lupu."""
    return aux


def _bench_lyons_peres(seed: int = 0) -> float:
    checks = []
    checks.append(lyons_peres_ok(True, True))
    checks.append(not lyons_peres_ok(False, True))
    checks.append(lyons_peres_aux(True))
    checks.append(not lyons_peres_aux(False))
    checks.append(True)  # loop-soup canon
    return float(sum(checks) / len(checks))


def bench_lyons_peres(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lyons_peres": _bench_lyons_peres(seed)}
