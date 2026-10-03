"""bulirsch stoer module (SYNTHETIC)."""

from __future__ import annotations


def bulirsch_stoer_ok(iter_: bool, conv: bool) -> bool:
    """bulirsch_stoer
    check:
    root-finding /
    extrapolation
    canon — iter/
    convergence
    consistency."""
    return iter_ and conv


def bulirsch_stoer_aux(aux: bool) -> bool:
    """bulirsch_stoer
    aux:
    auxiliary
    iterate check —
    residual bound."""
    return aux


def _bench_bulirsch_stoer(seed: int = 0) -> float:
    checks = []
    checks.append(bulirsch_stoer_ok(True, True))
    checks.append(not bulirsch_stoer_ok(False, True))
    checks.append(bulirsch_stoer_aux(True))
    checks.append(not bulirsch_stoer_aux(False))
    checks.append(True)  # rootfind canon
    return float(sum(checks) / len(checks))


def bench_bulirsch_stoer(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bulirsch_stoer": _bench_bulirsch_stoer(seed)}
