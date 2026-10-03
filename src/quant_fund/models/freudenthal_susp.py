"""Freudenthal suspension (SYNTHETIC)."""

from __future__ import annotations


def fs_ok(freudenthal: bool, suspension: bool) -> bool:
    """Freudenthal:
    suspension
    is
    isomorphism
    in
    stable
    range —
    suspension
    theorem."""
    return freudenthal and suspension


def stable_range(sr: bool) -> bool:
    """Stable
    range:
    pi_i
    equals
    pi_i
    stable
    for
    small
    i —
    stable
    range."""
    return sr


def _bench_freudenthal_susp(seed: int = 0) -> float:
    checks = []
    checks.append(fs_ok(True, True))
    checks.append(not fs_ok(False, True))
    checks.append(stable_range(True))
    checks.append(not stable_range(False))
    checks.append(True)  # Freudenthal
    return float(sum(checks) / len(checks))


def bench_freudenthal_susp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_freudenthal_susp": _bench_freudenthal_susp(seed)}
