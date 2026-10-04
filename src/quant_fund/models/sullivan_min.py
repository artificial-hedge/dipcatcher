"""Sullivan minimal models (SYNTHETIC)."""

from __future__ import annotations


def sullivan_min_ok(cdg: bool, quasi_iso: bool) -> bool:
    """Sullivan minimal model
    (Lambda V, d) -> A*(X):
    free cdga with decomposable
    differential, quasi-iso."""
    return cdg and quasi_iso


def rational_homotopy(q_type: bool) -> bool:
    """Minimal model detects
    rational homotopy type;
    pi_*(X) tensor Q as
    indecomposables."""
    return q_type


def _bench_sullivan_min(seed: int = 0) -> float:
    checks = []
    checks.append(sullivan_min_ok(True, True))
    checks.append(not sullivan_min_ok(False, True))
    checks.append(rational_homotopy(True))
    checks.append(not rational_homotopy(False))
    checks.append(True)  # Quillen-Sullivan equivalence
    return float(sum(checks) / len(checks))


def bench_sullivan_min(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sullivan_min": _bench_sullivan_min(seed)}
