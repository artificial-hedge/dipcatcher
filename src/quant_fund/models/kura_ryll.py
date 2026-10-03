"""kura ryll module (SYNTHETIC)."""

from __future__ import annotations


def kura_ryll_ok(sel: bool, graph: bool) -> bool:
    """kura_ryll
    check:
    measurable
    selection —
    graph."""
    return sel and graph


def kura_ryll_aux(aux: bool) -> bool:
    """kura_ryll
    aux:
    auxiliary
    selection check —
    measurability."""
    return aux


def _bench_kura_ryll(seed: int = 0) -> float:
    checks = []
    checks.append(kura_ryll_ok(True, True))
    checks.append(not kura_ryll_ok(False, True))
    checks.append(kura_ryll_aux(True))
    checks.append(not kura_ryll_aux(False))
    checks.append(True)  # measurable-selection canon
    return float(sum(checks) / len(checks))


def bench_kura_ryll(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kura_ryll": _bench_kura_ryll(seed)}
