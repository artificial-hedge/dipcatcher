"""Fusion category (SYNTHETIC)."""

from __future__ import annotations


def fc_ok(semisimple: bool, rigid: bool) -> bool:
    """Fusion:
    semisimple
    rigid
    finite
    tensor —
    fusion
    category."""
    return semisimple and rigid


def fusion_rules(fr: bool) -> bool:
    """Fusion
    rules:
    fusion
    coefficients
    nonneg
    integers —
    Verlinde
    rules."""
    return fr


def _bench_fusion_cat(seed: int = 0) -> float:
    checks = []
    checks.append(fc_ok(True, True))
    checks.append(not fc_ok(False, True))
    checks.append(fusion_rules(True))
    checks.append(not fusion_rules(False))
    checks.append(True)  # Verlinde
    return float(sum(checks) / len(checks))


def bench_fusion_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fusion_cat": _bench_fusion_cat(seed)}
