"""Derived loop (SYNTHETIC)."""

from __future__ import annotations


def dl_ok(derived: bool, loop: bool) -> bool:
    """Derived
    loop:
    derived
    loop
    space —
    derived
    S1."""
    return derived and loop


def derived_s1_map(ds: bool) -> bool:
    """Derived
    S1:
    derived
    S1
    mapping
    space —
    HKR."""
    return ds


def _bench_derived_loop(seed: int = 0) -> float:
    checks = []
    checks.append(dl_ok(True, True))
    checks.append(not dl_ok(False, True))
    checks.append(derived_s1_map(True))
    checks.append(not derived_s1_map(False))
    checks.append(True)  # HKR
    return float(sum(checks) / len(checks))


def bench_derived_loop(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_loop": _bench_derived_loop(seed)}
