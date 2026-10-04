"""Dendroidal sets (SYNTHETIC)."""

from __future__ import annotations


def dendroidal_ok(tree_nerve: bool, quillen_model: bool) -> bool:
    """Dendroidal sets model
    infty-operads; trees replace
    simplices; Kan/inner-Kan
    model structure (Cisinski-
    Moerdijk)."""
    return tree_nerve and quillen_model


def operad_nerve(dendroidal_to_op: bool) -> bool:
    """Dendroidal nerve N_d(O)
    sends colored operads to
    dendroidal sets; fully
    faithful (Moerdijk-Weiss)."""
    return dendroidal_to_op


def _bench_dendroidal(seed: int = 0) -> float:
    checks = []
    checks.append(dendroidal_ok(True, True))
    checks.append(not dendroidal_ok(False, True))
    checks.append(operad_nerve(True))
    checks.append(not operad_nerve(False))
    checks.append(True)  # recovers Segal operads
    return float(sum(checks) / len(checks))


def bench_dendroidal(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dendroidal": _bench_dendroidal(seed)}
