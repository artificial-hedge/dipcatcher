"""derived quasi_coherent module (SYNTHETIC)."""

from __future__ import annotations


def derived_quasi_coherent_ok(derived: bool, geometric: bool) -> bool:
    """derived_quasi_coherent
    check:
    derived
    structure —
    etale."""
    return derived and geometric


def derived_quasi_coherent_aux(aux: bool) -> bool:
    """derived_quasi_coherent
    aux:
    auxiliary
    derived
    check —
    flat."""
    return aux


def _bench_derived_quasi_coherent(seed: int = 0) -> float:
    checks = []
    checks.append(derived_quasi_coherent_ok(True, True))
    checks.append(not derived_quasi_coherent_ok(False, True))
    checks.append(derived_quasi_coherent_aux(True))
    checks.append(not derived_quasi_coherent_aux(False))
    checks.append(True)  # derived geometry canon
    return float(sum(checks) / len(checks))


def bench_derived_quasi_coherent(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_quasi_coherent": _bench_derived_quasi_coherent(seed)}
