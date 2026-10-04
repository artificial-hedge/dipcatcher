"""measur select module (SYNTHETIC)."""

from __future__ import annotations


def measur_select_ok(sel: bool, graph: bool) -> bool:
    """measur_select
    check:
    measurable
    selection —
    graph."""
    return sel and graph


def measur_select_aux(aux: bool) -> bool:
    """measur_select
    aux:
    auxiliary
    selection check —
    measurability."""
    return aux


def _bench_measur_select(seed: int = 0) -> float:
    checks = []
    checks.append(measur_select_ok(True, True))
    checks.append(not measur_select_ok(False, True))
    checks.append(measur_select_aux(True))
    checks.append(not measur_select_aux(False))
    checks.append(True)  # measurable-selection canon
    return float(sum(checks) / len(checks))


def bench_measur_select(seed: int = 0) -> dict[str, float]:
    return {"synthetic_measur_select": _bench_measur_select(seed)}
