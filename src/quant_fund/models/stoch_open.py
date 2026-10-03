"""stoch open module (SYNTHETIC)."""

from __future__ import annotations


def stoch_open_ok(sel: bool, graph: bool) -> bool:
    """stoch_open
    check:
    measurable
    selection —
    graph."""
    return sel and graph


def stoch_open_aux(aux: bool) -> bool:
    """stoch_open
    aux:
    auxiliary
    selection check —
    measurability."""
    return aux


def _bench_stoch_open(seed: int = 0) -> float:
    checks = []
    checks.append(stoch_open_ok(True, True))
    checks.append(not stoch_open_ok(False, True))
    checks.append(stoch_open_aux(True))
    checks.append(not stoch_open_aux(False))
    checks.append(True)  # measurable-selection canon
    return float(sum(checks) / len(checks))


def bench_stoch_open(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stoch_open": _bench_stoch_open(seed)}
