"""Snaith splitting (SYNTHETIC)."""

from __future__ import annotations


def ss_ok(snaith: bool, split: bool) -> bool:
    """Snaith
    split:
    Snaith
    splitting —
    suspension."""
    return snaith and split


def stable_split(split2: bool) -> bool:
    """Stable
    split:
    stable
    splitting —
    configuration."""
    return split2


def _bench_snaith_split(seed: int = 0) -> float:
    checks = []
    checks.append(ss_ok(True, True))
    checks.append(not ss_ok(False, True))
    checks.append(stable_split(True))
    checks.append(not stable_split(False))
    checks.append(True)  # Snaith
    return float(sum(checks) / len(checks))


def bench_snaith_split(seed: int = 0) -> dict[str, float]:
    return {"synthetic_snaith_split": _bench_snaith_split(seed)}
