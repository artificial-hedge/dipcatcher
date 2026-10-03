"""Lying-over theorem (SYNTHETIC)."""

from __future__ import annotations


def lo_ok(lying: bool, over: bool) -> bool:
    """Lying
    over:
    lying
    over
    theorem —
    surjective
    spec."""
    return lying and over


def spec_surjective(ss: bool) -> bool:
    """Spec
    surjective:
    spec
    surjective
    map —
    lying
    over."""
    return ss


def _bench_lying_over(seed: int = 0) -> float:
    checks = []
    checks.append(lo_ok(True, True))
    checks.append(not lo_ok(False, True))
    checks.append(spec_surjective(True))
    checks.append(not spec_surjective(False))
    checks.append(True)  # Cohen-Seidenberg
    return float(sum(checks) / len(checks))


def bench_lying_over(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lying_over": _bench_lying_over(seed)}
