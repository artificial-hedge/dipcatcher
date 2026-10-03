"""berestycki sheffield module (SYNTHETIC)."""

from __future__ import annotations


def berestycki_sheffield_ok(gff: bool, lqg: bool) -> bool:
    """berestycki_sheffield
    check:
    LQG
    structure —
    Sheffield."""
    return gff and lqg


def berestycki_sheffield_aux(aux: bool) -> bool:
    """berestycki_sheffield
    aux:
    auxiliary
    LQG
    check —
    Miller."""
    return aux


def _bench_berestycki_sheffield(seed: int = 0) -> float:
    checks = []
    checks.append(berestycki_sheffield_ok(True, True))
    checks.append(not berestycki_sheffield_ok(False, True))
    checks.append(berestycki_sheffield_aux(True))
    checks.append(not berestycki_sheffield_aux(False))
    checks.append(True)  # LQG canon
    return float(sum(checks) / len(checks))


def bench_berestycki_sheffield(seed: int = 0) -> dict[str, float]:
    return {"synthetic_berestycki_sheffield": _bench_berestycki_sheffield(seed)}
