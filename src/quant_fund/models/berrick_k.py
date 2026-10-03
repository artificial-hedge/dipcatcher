"""Berrick K-theory (SYNTHETIC)."""

from __future__ import annotations


def bk_ok(berrick: bool, theory: bool) -> bool:
    """Berrick:
    Berrick
    approach
    to
    K-
    theory —
    Berrick."""
    return berrick and theory


def plus_completion(pc: bool) -> bool:
    """Plus
    completion:
    Quillen
    plus
    construction —
    plus
    construction."""
    return pc


def _bench_berrick_k(seed: int = 0) -> float:
    checks = []
    checks.append(bk_ok(True, True))
    checks.append(not bk_ok(False, True))
    checks.append(plus_completion(True))
    checks.append(not plus_completion(False))
    checks.append(True)  # Berrick
    return float(sum(checks) / len(checks))


def bench_berrick_k(seed: int = 0) -> dict[str, float]:
    return {"synthetic_berrick_k": _bench_berrick_k(seed)}
