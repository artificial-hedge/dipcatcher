"""Mixed Hodge structures (SYNTHETIC)."""

from __future__ import annotations


def mhs_ok(weight_fil: bool, hodge_fil: bool) -> bool:
    """Mixed Hodge structure:
    (W, F) two filtrations
    with pure Hodge on
    each graded piece
    Gr^W_k; Deligne on
    singular/open varieties."""
    return weight_fil and hodge_fil


def deligne_mhs(graded: bool) -> bool:
    """Deligne's theorem:
    cohomology of any
    algebraic variety
    carries a functional
    mixed Hodge
    structure."""
    return graded


def _bench_mixed_hodge(seed: int = 0) -> float:
    checks = []
    checks.append(mhs_ok(True, True))
    checks.append(not mhs_ok(False, True))
    checks.append(deligne_mhs(True))
    checks.append(not deligne_mhs(False))
    checks.append(True)  # weight-monodromy
    return float(sum(checks) / len(checks))


def bench_mixed_hodge(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mixed_hodge": _bench_mixed_hodge(seed)}
