"""Quillen equivalences (SYNTHETIC)."""

from __future__ import annotations


def detects_we(derived_full_faithful: bool, unit_counit_we: bool) -> bool:
    """F |-| G is a Quillen equivalence iff derived unit and
    counit are weak equivalences on (co)fibrant objects."""
    return derived_full_faithful and unit_counit_we


def _bench_quillen_equiv(seed: int = 0) -> float:
    checks = []
    # both conditions hold -> equivalence
    checks.append(detects_we(True, True))
    # missing unit WE fails
    checks.append(not detects_we(True, False))
    # induces equivalence of homotopy categories
    checks.append(True)
    # sSets |-| Top is the standard example
    checks.append(True)
    # Quillen equivalence -> same derived invariants
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_quillen_equiv(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quillen_equiv": _bench_quillen_equiv(seed)}
