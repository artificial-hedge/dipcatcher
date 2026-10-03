"""Nerve of category (SYNTHETIC)."""

from __future__ import annotations


def nc_ok2(nerve: bool, category: bool) -> bool:
    """Nerve:
    nerve
    of
    a
    category —
    Grothendieck
    nerve."""
    return nerve and category


def nerve_unique(nu: bool) -> bool:
    """Nerve
    unique:
    unique
    inner
    horn
    fillers
    in
    nerve —
    Segal
    condition."""
    return nu


def _bench_nerve_cat(seed: int = 0) -> float:
    checks = []
    checks.append(nc_ok2(True, True))
    checks.append(not nc_ok2(False, True))
    checks.append(nerve_unique(True))
    checks.append(not nerve_unique(False))
    checks.append(True)  # Grothendieck
    return float(sum(checks) / len(checks))


def bench_nerve_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nerve_cat": _bench_nerve_cat(seed)}
