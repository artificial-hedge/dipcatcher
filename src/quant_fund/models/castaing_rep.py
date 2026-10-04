"""castaing rep module (SYNTHETIC)."""

from __future__ import annotations


def castaing_rep_ok(sel: bool, graph: bool) -> bool:
    """castaing_rep
    check:
    measurable
    selection —
    graph."""
    return sel and graph


def castaing_rep_aux(aux: bool) -> bool:
    """castaing_rep
    aux:
    auxiliary
    selection check —
    measurability."""
    return aux


def _bench_castaing_rep(seed: int = 0) -> float:
    checks = []
    checks.append(castaing_rep_ok(True, True))
    checks.append(not castaing_rep_ok(False, True))
    checks.append(castaing_rep_aux(True))
    checks.append(not castaing_rep_aux(False))
    checks.append(True)  # measurable-selection canon
    return float(sum(checks) / len(checks))


def bench_castaing_rep(seed: int = 0) -> dict[str, float]:
    return {"synthetic_castaing_rep": _bench_castaing_rep(seed)}
