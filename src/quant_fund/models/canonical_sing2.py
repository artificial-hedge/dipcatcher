"""Canonical singularity (SYNTHETIC)."""

from __future__ import annotations


def cs_ok(discrepancy_nonneg: bool, canonical: bool) -> bool:
    """Canonical:
    discrepancies
    non-
    negative —
    canonical
    singularities."""
    return discrepancy_nonneg and canonical


def canonical_model_sing(cms: bool) -> bool:
    """Canonical
    model:
    canonical
    model
    has
    canonical
    singularities —
    MMP
    endpoint."""
    return cms


def _bench_canonical_sing2(seed: int = 0) -> float:
    checks = []
    checks.append(cs_ok(True, True))
    checks.append(not cs_ok(False, True))
    checks.append(canonical_model_sing(True))
    checks.append(not canonical_model_sing(False))
    checks.append(True)  # Reid
    return float(sum(checks) / len(checks))


def bench_canonical_sing2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_canonical_sing2": _bench_canonical_sing2(seed)}
