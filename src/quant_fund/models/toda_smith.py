"""Toda-Smith spectra (SYNTHETIC)."""

from __future__ import annotations


def ts_ok(toda: bool, smith: bool) -> bool:
    """Toda
    Smith:
    Toda
    Smith
    spectrum —
    periodicity."""
    return toda and smith


def v_self_map(v: bool) -> bool:
    """V
    self:
    v
    self
    map —
    telescope."""
    return v


def _bench_toda_smith(seed: int = 0) -> float:
    checks = []
    checks.append(ts_ok(True, True))
    checks.append(not ts_ok(False, True))
    checks.append(v_self_map(True))
    checks.append(not v_self_map(False))
    checks.append(True)  # Toda-Smith
    return float(sum(checks) / len(checks))


def bench_toda_smith(seed: int = 0) -> dict[str, float]:
    return {"synthetic_toda_smith": _bench_toda_smith(seed)}
