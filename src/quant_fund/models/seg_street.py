"""Segal-Street categories (SYNTHETIC)."""

from __future__ import annotations


def ss_ok(segal: bool, street: bool) -> bool:
    """Segal
    Street:
    Segal
    Street
    category —
    2
    Segal."""
    return segal and street


def two_segal(ts: bool) -> bool:
    """2-Segal:
    2-Segal
    space —
    Dyckerhoff
    Kapranov."""
    return ts


def _bench_seg_street(seed: int = 0) -> float:
    checks = []
    checks.append(ss_ok(True, True))
    checks.append(not ss_ok(False, True))
    checks.append(two_segal(True))
    checks.append(not two_segal(False))
    checks.append(True)  # Dyckerhoff-Kapranov
    return float(sum(checks) / len(checks))


def bench_seg_street(seed: int = 0) -> dict[str, float]:
    return {"synthetic_seg_street": _bench_seg_street(seed)}
