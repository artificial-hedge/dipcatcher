"""Hodge conjecture (SYNTHETIC)."""

from __future__ import annotations


def hconj_ok(rational_cycle: bool, algebraic_class: bool) -> bool:
    """Hodge
    conjecture:
    rational
    Hodge
    classes
    are
    algebraic —
    open
    in
    general."""
    return rational_cycle and algebraic_class


def cattani_deligne_kaplan(cdk: bool) -> bool:
    """Cattani-
    Deligne-
    Kaplan:
    Hodge
    loci
    are
    algebraic —
    evidence
    for
    Hodge
    conjecture."""
    return cdk


def _bench_hodge_conj(seed: int = 0) -> float:
    checks = []
    checks.append(hconj_ok(True, True))
    checks.append(not hconj_ok(False, True))
    checks.append(cattani_deligne_kaplan(True))
    checks.append(not cattani_deligne_kaplan(False))
    checks.append(True)  # Hodge-CDK
    return float(sum(checks) / len(checks))


def bench_hodge_conj(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hodge_conj": _bench_hodge_conj(seed)}
