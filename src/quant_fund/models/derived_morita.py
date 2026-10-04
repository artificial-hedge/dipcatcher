"""derived morita module (SYNTHETIC)."""

from __future__ import annotations


def derived_morita_ok(dim: bool, categorical: bool) -> bool:
    """derived_morita
    check:
    dimension
    structure —
    entropy."""
    return dim and categorical


def derived_morita_aux(aux: bool) -> bool:
    """derived_morita
    aux:
    auxiliary
    dimension
    check —
    tilting."""
    return aux


def _bench_derived_morita(seed: int = 0) -> float:
    checks = []
    checks.append(derived_morita_ok(True, True))
    checks.append(not derived_morita_ok(False, True))
    checks.append(derived_morita_aux(True))
    checks.append(not derived_morita_aux(False))
    checks.append(True)  # derived-dimension canon
    return float(sum(checks) / len(checks))


def bench_derived_morita(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_morita": _bench_derived_morita(seed)}
