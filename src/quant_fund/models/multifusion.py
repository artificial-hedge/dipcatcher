"""Multifusion categories (SYNTHETIC)."""

from __future__ import annotations


def mf_ok(multi: bool, fusion: bool) -> bool:
    """Multifusion:
    multifusion
    category —
    multi-
    fusion."""
    return multi and fusion


def fusion_rules(fr: bool) -> bool:
    """Fusion:
    fusion
    rules
    of
    multifusion
    categories —
    fusion
    rules."""
    return fr


def _bench_multifusion(seed: int = 0) -> float:
    checks = []
    checks.append(mf_ok(True, True))
    checks.append(not mf_ok(False, True))
    checks.append(fusion_rules(True))
    checks.append(not fusion_rules(False))
    checks.append(True)  # multifusion
    return float(sum(checks) / len(checks))


def bench_multifusion(seed: int = 0) -> dict[str, float]:
    return {"synthetic_multifusion": _bench_multifusion(seed)}
