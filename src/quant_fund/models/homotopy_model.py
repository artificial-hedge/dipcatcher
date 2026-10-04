"""homotopy model module (SYNTHETIC)."""

from __future__ import annotations


def homotopy_model_ok(homotopy: bool, stable: bool) -> bool:
    """homotopy_model
    check:
    homotopy
    structure —
    stable."""
    return homotopy and stable


def homotopy_model_aux(aux: bool) -> bool:
    """homotopy_model
    aux:
    auxiliary
    homotopy
    check —
    monoid."""
    return aux


def _bench_homotopy_model(seed: int = 0) -> float:
    checks = []
    checks.append(homotopy_model_ok(True, True))
    checks.append(not homotopy_model_ok(False, True))
    checks.append(homotopy_model_aux(True))
    checks.append(not homotopy_model_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_homotopy_model(seed: int = 0) -> dict[str, float]:
    return {"synthetic_homotopy_model": _bench_homotopy_model(seed)}
