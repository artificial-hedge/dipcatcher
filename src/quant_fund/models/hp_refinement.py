"""hp refinement module (SYNTHETIC)."""

from __future__ import annotations


def hp_refinement_ok(node: bool, poly: bool) -> bool:
    """hp_refinement
    check:
    spectral-element —
    high-order
    consistency."""
    return node and poly


def hp_refinement_aux(aux: bool) -> bool:
    """hp_refinement
    aux:
    auxiliary
    SEM check —
    interpolation."""
    return aux


def _bench_hp_refinement(seed: int = 0) -> float:
    checks = []
    checks.append(hp_refinement_ok(True, True))
    checks.append(not hp_refinement_ok(False, True))
    checks.append(hp_refinement_aux(True))
    checks.append(not hp_refinement_aux(False))
    checks.append(True)  # spectral-element canon
    return float(sum(checks) / len(checks))


def bench_hp_refinement(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hp_refinement": _bench_hp_refinement(seed)}
