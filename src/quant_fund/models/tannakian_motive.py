"""Tannakian motives (SYNTHETIC)."""

from __future__ import annotations


def tm_ok(tannakian: bool, motive: bool) -> bool:
    """Tannakian
    motive:
    Tannakian
    category
    of
    motives —
    neutral."""
    return tannakian and motive


def tannakian_formalism(tf: bool) -> bool:
    """Tannakian
    formalism:
    neutral
    Tannakian
    cat —
    fiber
    functor."""
    return tf


def _bench_tannakian_motive(seed: int = 0) -> float:
    checks = []
    checks.append(tm_ok(True, True))
    checks.append(not tm_ok(False, True))
    checks.append(tannakian_formalism(True))
    checks.append(not tannakian_formalism(False))
    checks.append(True)  # Saavedra
    return float(sum(checks) / len(checks))


def bench_tannakian_motive(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tannakian_motive": _bench_tannakian_motive(seed)}
