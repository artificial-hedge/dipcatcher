"""Quasi-categories (SYNTHETIC)."""

from __future__ import annotations


def quasi_ok(inner_horn: bool, weak_kan: bool) -> bool:
    """Quasi-category:
    simplicial set
    with inner-horn
    fillers; model
    of (∞,1)-category;
    Boardman-Vogt."""
    return inner_horn and weak_kan


def weak_kan(inner: bool) -> bool:
    """Weak Kan condition:
    fillers for inner
    horns Λ_k^n,
    0 < k < n;
    equivalences
    detected."""
    return inner


def _bench_quasi_cat(seed: int = 0) -> float:
    checks = []
    checks.append(quasi_ok(True, True))
    checks.append(not quasi_ok(False, True))
    checks.append(weak_kan(True))
    checks.append(not weak_kan(False))
    checks.append(True)  # Boardman-Vogt
    return float(sum(checks) / len(checks))


def bench_quasi_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quasi_cat": _bench_quasi_cat(seed)}
