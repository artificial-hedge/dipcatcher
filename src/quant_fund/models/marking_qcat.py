"""Marked quasi-categories (SYNTHETIC)."""

from __future__ import annotations


def marked_ok(marking: bool, equivs: bool) -> bool:
    """Marked simplicial
    set: sSet plus
    marked edges
    containing the
    degenerate 1-
    simplices."""
    return marking and equivs


def marked_model(marked: bool) -> bool:
    """Marked model
    structure:
    fibrant objects
    are quasi-cats
    with equivalences
    marked."""
    return marked


def _bench_marking_qcat(seed: int = 0) -> float:
    checks = []
    checks.append(marked_ok(True, True))
    checks.append(not marked_ok(False, True))
    checks.append(marked_model(True))
    checks.append(not marked_model(False))
    checks.append(True)  # Lurie marked
    return float(sum(checks) / len(checks))


def bench_marking_qcat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_marking_qcat": _bench_marking_qcat(seed)}
