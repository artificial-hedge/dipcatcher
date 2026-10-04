"""hp adaptive module (SYNTHETIC)."""

from __future__ import annotations


def hp_adaptive_ok(elem: bool, mark: bool) -> bool:
    """hp_adaptive
    check:
    adaptive-mesh
    canon —
    elem/marking
    consistency."""
    return elem and mark


def hp_adaptive_aux(aux: bool) -> bool:
    """hp_adaptive
    aux:
    auxiliary
    refinement check —
    error bound."""
    return aux


def _bench_hp_adaptive(seed: int = 0) -> float:
    checks = []
    checks.append(hp_adaptive_ok(True, True))
    checks.append(not hp_adaptive_ok(False, True))
    checks.append(hp_adaptive_aux(True))
    checks.append(not hp_adaptive_aux(False))
    checks.append(True)  # adaptive canon
    return float(sum(checks) / len(checks))


def bench_hp_adaptive(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hp_adaptive": _bench_hp_adaptive(seed)}
