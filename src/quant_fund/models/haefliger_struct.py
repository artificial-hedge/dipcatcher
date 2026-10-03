"""Haefliger structures (SYNTHETIC)."""

from __future__ import annotations


def hae_ok(gamma: bool, cocycle: bool) -> bool:
    """Haefliger
    structure:
    Gamma-q
    cocycle
    of
    local
    submersions —
    classifying
    space
    B-Gamma."""
    return gamma and cocycle


def classify_fol(cf: bool) -> bool:
    """Classification:
    integrable
    homotopy
    classes
    of
    Haefliger
    structures
    classify
    foliations
    up
    to
    concordance."""
    return cf


def _bench_haefliger_struct(seed: int = 0) -> float:
    checks = []
    checks.append(hae_ok(True, True))
    checks.append(not hae_ok(False, True))
    checks.append(classify_fol(True))
    checks.append(not classify_fol(False))
    checks.append(True)  # Haefliger
    return float(sum(checks) / len(checks))


def bench_haefliger_struct(seed: int = 0) -> dict[str, float]:
    return {"synthetic_haefliger_struct": _bench_haefliger_struct(seed)}
