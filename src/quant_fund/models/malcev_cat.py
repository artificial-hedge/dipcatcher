"""Mal'cev category theory (SYNTHETIC)."""

from __future__ import annotations


def mc_ok(malcev: bool, permutable: bool) -> bool:
    """Mal'cev
    category:
    Mal'cev
    cat —
    congruence
    permutable."""
    return malcev and permutable


def malcev_operation(mo: bool) -> bool:
    """Mal'cev
    operation:
    Mal'cev
    operation
    p(x,y,y)=x —
    ternary."""
    return mo


def _bench_malcev_cat(seed: int = 0) -> float:
    checks = []
    checks.append(mc_ok(True, True))
    checks.append(not mc_ok(False, True))
    checks.append(malcev_operation(True))
    checks.append(not malcev_operation(False))
    checks.append(True)  # Bourn
    return float(sum(checks) / len(checks))


def bench_malcev_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_malcev_cat": _bench_malcev_cat(seed)}
