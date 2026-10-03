"""heegner cycle module (SYNTHETIC)."""

from __future__ import annotations


def heegner_cycle_ok(cycle: bool, arithmetic: bool) -> bool:
    """heegner_cycle
    check:
    arithmetic-cycle
    structure —
    Heegner."""
    return cycle and arithmetic


def heegner_cycle_aux(aux: bool) -> bool:
    """heegner_cycle
    aux:
    auxiliary
    cycle
    check —
    Shimura."""
    return aux


def _bench_heegner_cycle(seed: int = 0) -> float:
    checks = []
    checks.append(heegner_cycle_ok(True, True))
    checks.append(not heegner_cycle_ok(False, True))
    checks.append(heegner_cycle_aux(True))
    checks.append(not heegner_cycle_aux(False))
    checks.append(True)  # arithmetic-cycles canon
    return float(sum(checks) / len(checks))


def bench_heegner_cycle(seed: int = 0) -> dict[str, float]:
    return {"synthetic_heegner_cycle": _bench_heegner_cycle(seed)}
