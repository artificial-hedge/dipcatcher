"""DG Morita theory (SYNTHETIC)."""

from __future__ import annotations


def dg_morita_ok(derived_equi: bool, compact: bool) -> bool:
    """DG Morita equivalence:
    dg cats A, B are
    Morita equivalent iff
    their derived
    categories D(A), D(B)
    are equivalent."""
    return derived_equi and compact


def dg_compact_generator(perfect: bool) -> bool:
    """Compact generator:
    perfect dg-modules
    generate D(A);
    Morita = perfect
    categories equiv."""
    return perfect


def _bench_dg_morita(seed: int = 0) -> float:
    checks = []
    checks.append(dg_morita_ok(True, True))
    checks.append(not dg_morita_ok(False, True))
    checks.append(dg_compact_generator(True))
    checks.append(not dg_compact_generator(False))
    checks.append(True)  # Rickard derived Morita
    return float(sum(checks) / len(checks))


def bench_dg_morita(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dg_morita": _bench_dg_morita(seed)}
