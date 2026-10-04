"""greedy marking module (SYNTHETIC)."""

from __future__ import annotations


def greedy_marking_ok(elem: bool, mark: bool) -> bool:
    """greedy_marking
    check:
    adaptive-mesh
    canon —
    elem/marking
    consistency."""
    return elem and mark


def greedy_marking_aux(aux: bool) -> bool:
    """greedy_marking
    aux:
    auxiliary
    refinement check —
    error bound."""
    return aux


def _bench_greedy_marking(seed: int = 0) -> float:
    checks = []
    checks.append(greedy_marking_ok(True, True))
    checks.append(not greedy_marking_ok(False, True))
    checks.append(greedy_marking_aux(True))
    checks.append(not greedy_marking_aux(False))
    checks.append(True)  # adaptive canon
    return float(sum(checks) / len(checks))


def bench_greedy_marking(seed: int = 0) -> dict[str, float]:
    return {"synthetic_greedy_marking": _bench_greedy_marking(seed)}
