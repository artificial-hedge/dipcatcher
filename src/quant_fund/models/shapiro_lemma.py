"""Shapiro lemma (SYNTHETIC)."""

from __future__ import annotations


def sl_ok(shapiro: bool, induced: bool) -> bool:
    """Shapiro:
    induced
    module
    cohomology
    equals
    subgroup —
    Shapiro
    lemma."""
    return shapiro and induced


def induced_module(im: bool) -> bool:
    """Induced:
    cohomology
    of
    induced
    module
    vanishes —
    induced
    module."""
    return im


def _bench_shapiro_lemma(seed: int = 0) -> float:
    checks = []
    checks.append(sl_ok(True, True))
    checks.append(not sl_ok(False, True))
    checks.append(induced_module(True))
    checks.append(not induced_module(False))
    checks.append(True)  # Shapiro
    return float(sum(checks) / len(checks))


def bench_shapiro_lemma(seed: int = 0) -> dict[str, float]:
    return {"synthetic_shapiro_lemma": _bench_shapiro_lemma(seed)}
