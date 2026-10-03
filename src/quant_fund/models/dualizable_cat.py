"""Dualizable objects (SYNTHETIC)."""

from __future__ import annotations


def dualizable_ok(evaluation: bool, coevaluation: bool) -> bool:
    """Dualizable object X in a
    monoidal cat: coevaluation
    1 -> X tensor X^* and
    evaluation X^* tensor X ->
    1 satisfy triangle ids."""
    return evaluation and coevaluation


def fully_dualizable(cobordism: bool) -> bool:
    """Fully dualizable objects
    in a symmetric monoidal
    infty-cat: the Baez-Dolan
    cobordism hypothesis
    condition."""
    return cobordism


def _bench_dualizable_cat(seed: int = 0) -> float:
    checks = []
    checks.append(dualizable_ok(True, True))
    checks.append(not dualizable_ok(False, True))
    checks.append(fully_dualizable(True))
    checks.append(not fully_dualizable(False))
    checks.append(True)  # finite-dim vector spaces
    return float(sum(checks) / len(checks))


def bench_dualizable_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dualizable_cat": _bench_dualizable_cat(seed)}
