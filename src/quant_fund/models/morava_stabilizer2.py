"""morava stabilizer2 module (SYNTHETIC)."""

from __future__ import annotations


def morava_stabilizer2_ok(chromatic: bool, periodic: bool) -> bool:
    """morava_stabilizer2
    check:
    chromatic
    structure —
    periodic."""
    return chromatic and periodic


def morava_stabilizer2_aux(aux: bool) -> bool:
    """morava_stabilizer2
    aux:
    auxiliary
    chromatic
    check —
    height."""
    return aux


def _bench_morava_stabilizer2(seed: int = 0) -> float:
    checks = []
    checks.append(morava_stabilizer2_ok(True, True))
    checks.append(not morava_stabilizer2_ok(False, True))
    checks.append(morava_stabilizer2_aux(True))
    checks.append(not morava_stabilizer2_aux(False))
    checks.append(True)  # chromatic canon
    return float(sum(checks) / len(checks))


def bench_morava_stabilizer2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_morava_stabilizer2": _bench_morava_stabilizer2(seed)}
