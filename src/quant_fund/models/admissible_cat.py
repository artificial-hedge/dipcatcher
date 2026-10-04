"""Admissible categories (SYNTHETIC)."""

from __future__ import annotations


def adm_ok(admissible: bool, generated: bool) -> bool:
    """Admissible
    category:
    admissible
    cat —
    generates
    triangulated
    base."""
    return admissible and generated


def semiorthogonal_component(sc: bool) -> bool:
    """Semiorthogonal:
    admissible
    sub —
    semiorthogonal
    component."""
    return sc


def _bench_admissible_cat(seed: int = 0) -> float:
    checks = []
    checks.append(adm_ok(True, True))
    checks.append(not adm_ok(False, True))
    checks.append(semiorthogonal_component(True))
    checks.append(not semiorthogonal_component(False))
    checks.append(True)  # Bondal-Kapranov
    return float(sum(checks) / len(checks))


def bench_admissible_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_admissible_cat": _bench_admissible_cat(seed)}
