"""Distributive law (SYNTHETIC)."""

from __future__ import annotations


def dl_ok(distributive: bool, law: bool) -> bool:
    """Distributive:
    distributive
    law
    between
    monads —
    Beck
    distributive."""
    return distributive and law


def beck_distributive(bd: bool) -> bool:
    """Beck
    distributive:
    distributive
    law
    coherence
    —
    Beck
    law."""
    return bd


def _bench_distributive_law(seed: int = 0) -> float:
    checks = []
    checks.append(dl_ok(True, True))
    checks.append(not dl_ok(False, True))
    checks.append(beck_distributive(True))
    checks.append(not beck_distributive(False))
    checks.append(True)  # Beck
    return float(sum(checks) / len(checks))


def bench_distributive_law(seed: int = 0) -> dict[str, float]:
    return {"synthetic_distributive_law": _bench_distributive_law(seed)}
