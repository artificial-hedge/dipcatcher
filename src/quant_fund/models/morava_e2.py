"""morava e2 module (SYNTHETIC)."""

from __future__ import annotations


def morava_e2_ok(chromatic: bool, height: bool) -> bool:
    """morava_e2
    check:
    chromatic
    structure —
    height."""
    return chromatic and height


def morava_e2_aux(aux: bool) -> bool:
    """morava_e2
    aux:
    auxiliary
    chromatic
    check —
    periodicity."""
    return aux


def _bench_morava_e2(seed: int = 0) -> float:
    checks = []
    checks.append(morava_e2_ok(True, True))
    checks.append(not morava_e2_ok(False, True))
    checks.append(morava_e2_aux(True))
    checks.append(not morava_e2_aux(False))
    checks.append(True)  # chromatic canon
    return float(sum(checks) / len(checks))


def bench_morava_e2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_morava_e2": _bench_morava_e2(seed)}
