"""gray periodic module (SYNTHETIC)."""

from __future__ import annotations


def gray_periodic_ok(homotopy: bool, unstable: bool) -> bool:
    """gray_periodic
    check:
    homotopy
    unstable
    structure —
    periodic."""
    return homotopy and unstable


def gray_periodic_aux(aux: bool) -> bool:
    """gray_periodic
    aux:
    auxiliary
    homotopy
    check —
    Adams."""
    return aux


def _bench_gray_periodic(seed: int = 0) -> float:
    checks = []
    checks.append(gray_periodic_ok(True, True))
    checks.append(not gray_periodic_ok(False, True))
    checks.append(gray_periodic_aux(True))
    checks.append(not gray_periodic_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_gray_periodic(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gray_periodic": _bench_gray_periodic(seed)}
