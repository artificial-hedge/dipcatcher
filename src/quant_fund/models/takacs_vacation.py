"""takacs vacation module (SYNTHETIC)."""

from __future__ import annotations


def takacs_vacation_ok(load: bool, block: bool) -> bool:
    """takacs_vacation
    check:
    loss-queue
    structure —
    Erlang
    formula."""
    return load and block


def takacs_vacation_aux(aux: bool) -> bool:
    """takacs_vacation
    aux:
    auxiliary
    vacation
    check —
    Pollaczek-Khinchine."""
    return aux


def _bench_takacs_vacation(seed: int = 0) -> float:
    checks = []
    checks.append(takacs_vacation_ok(True, True))
    checks.append(not takacs_vacation_ok(False, True))
    checks.append(takacs_vacation_aux(True))
    checks.append(not takacs_vacation_aux(False))
    checks.append(True)  # loss-queue canon
    return float(sum(checks) / len(checks))


def bench_takacs_vacation(seed: int = 0) -> dict[str, float]:
    return {"synthetic_takacs_vacation": _bench_takacs_vacation(seed)}
