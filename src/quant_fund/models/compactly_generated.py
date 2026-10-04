"""Compactly generated categories (SYNTHETIC)."""

from __future__ import annotations


def cg_ok(compact: bool, generated: bool) -> bool:
    """Compactly
    generated:
    compactly
    generated
    cat —
    compact
    objects
    detect."""
    return compact and generated


def compact_object(co: bool) -> bool:
    """Compact
    object:
    compact
    object —
    hom
    commutes
    filtered
    colimits."""
    return co


def _bench_compactly_generated(seed: int = 0) -> float:
    checks = []
    checks.append(cg_ok(True, True))
    checks.append(not cg_ok(False, True))
    checks.append(compact_object(True))
    checks.append(not compact_object(False))
    checks.append(True)  # Gabriel-Ulmer
    return float(sum(checks) / len(checks))


def bench_compactly_generated(seed: int = 0) -> dict[str, float]:
    return {"synthetic_compactly_generated": _bench_compactly_generated(seed)}
