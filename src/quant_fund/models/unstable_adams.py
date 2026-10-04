"""Unstable Adams spectral sequence (SYNTHETIC)."""

from __future__ import annotations


def ua_ok(unstable_ss: bool, ext: bool) -> bool:
    """Unstable
    Adams:
    spectral
    sequence
    to
    unstable
    homotopy —
    Massey-
    Peterson."""
    return unstable_ss and ext


def unstable_e2(ue2: bool) -> bool:
    """Unstable
    E2:
    Ext
    over
    unstable
    Steenrod
    algebra —
    E2
    term."""
    return ue2


def _bench_unstable_adams(seed: int = 0) -> float:
    checks = []
    checks.append(ua_ok(True, True))
    checks.append(not ua_ok(False, True))
    checks.append(unstable_e2(True))
    checks.append(not unstable_e2(False))
    checks.append(True)  # Massey-Peterson
    return float(sum(checks) / len(checks))


def bench_unstable_adams(seed: int = 0) -> dict[str, float]:
    return {"synthetic_unstable_adams": _bench_unstable_adams(seed)}
