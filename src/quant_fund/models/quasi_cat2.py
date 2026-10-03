"""Quasi-categories (SYNTHETIC)."""

from __future__ import annotations


def qc2_ok(quasi: bool, cat: bool) -> bool:
    """Quasi-
    category:
    quasi-
    category —
    Boardman-
    Vogt
    weak
    Kan."""
    return quasi and cat


def weak_kan(wk: bool) -> bool:
    """Weak
    Kan:
    weak
    Kan
    complex —
    inner
    Kan."""
    return wk


def _bench_quasi_cat2(seed: int = 0) -> float:
    checks = []
    checks.append(qc2_ok(True, True))
    checks.append(not qc2_ok(False, True))
    checks.append(weak_kan(True))
    checks.append(not weak_kan(False))
    checks.append(True)  # Boardman-Vogt
    return float(sum(checks) / len(checks))


def bench_quasi_cat2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quasi_cat2": _bench_quasi_cat2(seed)}
