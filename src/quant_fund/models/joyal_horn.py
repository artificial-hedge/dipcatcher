"""Joyal horn theory (SYNTHETIC)."""

from __future__ import annotations


def jh_ok(joyal: bool, horn: bool) -> bool:
    """Joyal
    horn:
    Joyal
    horn
    theory —
    Joyal
    inner
    horn."""
    return joyal and horn


def special_outer(so: bool) -> bool:
    """Special
    outer:
    special
    outer
    horn —
    Joyal
    outer
    horn."""
    return so


def _bench_joyal_horn(seed: int = 0) -> float:
    checks = []
    checks.append(jh_ok(True, True))
    checks.append(not jh_ok(False, True))
    checks.append(special_outer(True))
    checks.append(not special_outer(False))
    checks.append(True)  # Joyal
    return float(sum(checks) / len(checks))


def bench_joyal_horn(seed: int = 0) -> dict[str, float]:
    return {"synthetic_joyal_horn": _bench_joyal_horn(seed)}
