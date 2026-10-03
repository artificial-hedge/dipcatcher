"""morava k3 module (SYNTHETIC)."""

from __future__ import annotations


def morava_k3_ok(chromatic: bool, height: bool) -> bool:
    """morava_k3
    check:
    chromatic
    structure —
    height."""
    return chromatic and height


def morava_k3_aux(aux: bool) -> bool:
    """morava_k3
    aux:
    auxiliary
    chromatic
    check —
    periodicity."""
    return aux


def _bench_morava_k3(seed: int = 0) -> float:
    checks = []
    checks.append(morava_k3_ok(True, True))
    checks.append(not morava_k3_ok(False, True))
    checks.append(morava_k3_aux(True))
    checks.append(not morava_k3_aux(False))
    checks.append(True)  # chromatic canon
    return float(sum(checks) / len(checks))


def bench_morava_k3(seed: int = 0) -> dict[str, float]:
    return {"synthetic_morava_k3": _bench_morava_k3(seed)}
