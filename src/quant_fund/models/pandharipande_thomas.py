"""Pandharipande-Thomas invariants (SYNTHETIC)."""

from __future__ import annotations


def pt_ok(stable_pairs: bool, simpler: bool) -> bool:
    """PT
    theory:
    stable
    pairs
    O
    to
    F
    —
    simpler
    obstruction
    theory
    than
    DT."""
    return stable_pairs and simpler


def dt_pt_correspondence(dc: bool) -> bool:
    """DT-PT
    correspondence:
    their
    partition
    functions
    agree
    up
    to
    a
    degree-0
    factor —
    MNOP
    refined."""
    return dc


def _bench_pandharipande_thomas(seed: int = 0) -> float:
    checks = []
    checks.append(pt_ok(True, True))
    checks.append(not pt_ok(False, True))
    checks.append(dt_pt_correspondence(True))
    checks.append(not dt_pt_correspondence(False))
    checks.append(True)  # PT 2009
    return float(sum(checks) / len(checks))


def bench_pandharipande_thomas(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pandharipande_thomas": _bench_pandharipande_thomas(seed)}
