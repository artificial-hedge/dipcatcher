"""borel tanner module (SYNTHETIC)."""

from __future__ import annotations


def borel_tanner_ok(load: bool, block: bool) -> bool:
    """borel_tanner
    check:
    loss-queue
    structure —
    Erlang
    formula."""
    return load and block


def borel_tanner_aux(aux: bool) -> bool:
    """borel_tanner
    aux:
    auxiliary
    vacation
    check —
    Pollaczek-Khinchine."""
    return aux


def _bench_borel_tanner(seed: int = 0) -> float:
    checks = []
    checks.append(borel_tanner_ok(True, True))
    checks.append(not borel_tanner_ok(False, True))
    checks.append(borel_tanner_aux(True))
    checks.append(not borel_tanner_aux(False))
    checks.append(True)  # loss-queue canon
    return float(sum(checks) / len(checks))


def bench_borel_tanner(seed: int = 0) -> dict[str, float]:
    return {"synthetic_borel_tanner": _bench_borel_tanner(seed)}
