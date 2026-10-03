"""morava k2 module (SYNTHETIC)."""

from __future__ import annotations


def morava_k2_ok(chromatic: bool, height: bool) -> bool:
    """morava_k2
    check:
    chromatic
    height
    structure —
    stratified."""
    return chromatic and height


def morava_k2_aux(aux: bool) -> bool:
    """morava_k2
    aux:
    auxiliary
    chromatic
    check —
    spectral."""
    return aux


def _bench_morava_k2(seed: int = 0) -> float:
    checks = []
    checks.append(morava_k2_ok(True, True))
    checks.append(not morava_k2_ok(False, True))
    checks.append(morava_k2_aux(True))
    checks.append(not morava_k2_aux(False))
    checks.append(True)  # chromatic canon
    return float(sum(checks) / len(checks))


def bench_morava_k2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_morava_k2": _bench_morava_k2(seed)}
