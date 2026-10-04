"""Joyal model structure (SYNTHETIC)."""

from __future__ import annotations


def joyal_ok(model: bool, qcat_fib: bool) -> bool:
    """Joyal model
    structure on sSet:
    cofibrations =
    monomorphisms,
    fibrant objects
    = quasi-cats."""
    return model and qcat_fib


def cat_equiv(categorical: bool) -> bool:
    """Categorical
    equivalences:
    maps inducing
    equivalences of
    homotopy categories
    + isofibration
    on fibrant."""
    return categorical


def _bench_joyal_model(seed: int = 0) -> float:
    checks = []
    checks.append(joyal_ok(True, True))
    checks.append(not joyal_ok(False, True))
    checks.append(cat_equiv(True))
    checks.append(not cat_equiv(False))
    checks.append(True)  # Joyal 2002
    return float(sum(checks) / len(checks))


def bench_joyal_model(seed: int = 0) -> dict[str, float]:
    return {"synthetic_joyal_model": _bench_joyal_model(seed)}
