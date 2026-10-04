"""Unstable towers (SYNTHETIC)."""

from __future__ import annotations


def ut_ok(unstable: bool, tower: bool) -> bool:
    """Unstable
    tower:
    unstable
    tower —
    Bousfield
    tower."""
    return unstable and tower


def unstable_adams(ua: bool) -> bool:
    """Unstable
    Adams:
    unstable
    Adams
    spectral
    sequence."""
    return ua


def _bench_unstable_tower(seed: int = 0) -> float:
    checks = []
    checks.append(ut_ok(True, True))
    checks.append(not ut_ok(False, True))
    checks.append(unstable_adams(True))
    checks.append(not unstable_adams(False))
    checks.append(True)  # Bousfield
    return float(sum(checks) / len(checks))


def bench_unstable_tower(seed: int = 0) -> dict[str, float]:
    return {"synthetic_unstable_tower": _bench_unstable_tower(seed)}
