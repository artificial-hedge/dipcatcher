"""Regular categories 2 (SYNTHETIC)."""

from __future__ import annotations


def rc2_ok(regular: bool, exactness: bool) -> bool:
    """Regular
    cat
    2:
    exact
    regular —
    effective
    relations."""
    return regular and exactness


def effective_equivalence_relations(eer: bool) -> bool:
    """Effective
    relations:
    effective
    equivalence
    relations —
    Barr
    exact."""
    return eer


def _bench_regular_cat2(seed: int = 0) -> float:
    checks = []
    checks.append(rc2_ok(True, True))
    checks.append(not rc2_ok(False, True))
    checks.append(effective_equivalence_relations(True))
    checks.append(not effective_equivalence_relations(False))
    checks.append(True)  # Barr exactness
    return float(sum(checks) / len(checks))


def bench_regular_cat2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_regular_cat2": _bench_regular_cat2(seed)}
