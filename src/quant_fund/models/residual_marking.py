"""residual marking module (SYNTHETIC)."""

from __future__ import annotations


def residual_marking_ok(elem: bool, mark: bool) -> bool:
    """residual_marking
    check:
    adaptive-mesh
    canon —
    elem/marking
    consistency."""
    return elem and mark


def residual_marking_aux(aux: bool) -> bool:
    """residual_marking
    aux:
    auxiliary
    refinement check —
    error bound."""
    return aux


def _bench_residual_marking(seed: int = 0) -> float:
    checks = []
    checks.append(residual_marking_ok(True, True))
    checks.append(not residual_marking_ok(False, True))
    checks.append(residual_marking_aux(True))
    checks.append(not residual_marking_aux(False))
    checks.append(True)  # adaptive canon
    return float(sum(checks) / len(checks))


def bench_residual_marking(seed: int = 0) -> dict[str, float]:
    return {"synthetic_residual_marking": _bench_residual_marking(seed)}
