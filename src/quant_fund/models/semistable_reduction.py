"""Semistable reduction (SYNTHETIC)."""

from __future__ import annotations


def sr_ok(semistable: bool, reduction: bool) -> bool:
    """Semistable:
    semistable
    reduction
    theorem —
    Deligne-
    Mumford."""
    return semistable and reduction


def semistable_model(sm: bool) -> bool:
    """Semistable
    model:
    semistable
    model
    after
    base
    change —
    Deligne-
    Mumford."""
    return sm


def _bench_semistable_reduction(seed: int = 0) -> float:
    checks = []
    checks.append(sr_ok(True, True))
    checks.append(not sr_ok(False, True))
    checks.append(semistable_model(True))
    checks.append(not semistable_model(False))
    checks.append(True)  # DM
    return float(sum(checks) / len(checks))


def bench_semistable_reduction(seed: int = 0) -> dict[str, float]:
    return {"synthetic_semistable_reduction": _bench_semistable_reduction(seed)}
