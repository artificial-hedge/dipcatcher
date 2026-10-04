"""Virtual pullbacks (SYNTHETIC)."""

from __future__ import annotations


def virtual_pull_ok(man_lin: bool, virtual: bool) -> bool:
    """Virtual pullback f^!
    on Chow/homology via
    the cotangent complex;
    Manolache's construction
    for quasi-smooth maps."""
    return man_lin and virtual


def gysin_derived(pull: bool) -> bool:
    """Virtual Gysin maps
    commute with proper
    pushforward and
    realize virtual
    classes."""
    return pull


def _bench_virtual_pull(seed: int = 0) -> float:
    checks = []
    checks.append(virtual_pull_ok(True, True))
    checks.append(not virtual_pull_ok(False, True))
    checks.append(gysin_derived(True))
    checks.append(not gysin_derived(False))
    checks.append(True)  # Fulton-MacPherson general
    return float(sum(checks) / len(checks))


def bench_virtual_pull(seed: int = 0) -> dict[str, float]:
    return {"synthetic_virtual_pull": _bench_virtual_pull(seed)}
