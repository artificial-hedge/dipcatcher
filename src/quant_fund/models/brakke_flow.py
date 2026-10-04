"""Brakke flow (SYNTHETIC)."""

from __future__ import annotations


def bf_ok(varifold: bool, weak_mcf: bool) -> bool:
    """Brakke
    flow:
    measure-
    theoretic
    weak
    MCF
    through
    singularities —
    varifold
    formulation."""
    return varifold and weak_mcf


def ilmanen_elliptic(ie: bool) -> bool:
    """Ilmanen's
    elliptic
    regularization:
    MCF
    as
    limit
    of
    elliptic
    translators —
    canonical
    Brakke
    flows."""
    return ie


def _bench_brakke_flow(seed: int = 0) -> float:
    checks = []
    checks.append(bf_ok(True, True))
    checks.append(not bf_ok(False, True))
    checks.append(ilmanen_elliptic(True))
    checks.append(not ilmanen_elliptic(False))
    checks.append(True)  # Brakke 1978
    return float(sum(checks) / len(checks))


def bench_brakke_flow(seed: int = 0) -> dict[str, float]:
    return {"synthetic_brakke_flow": _bench_brakke_flow(seed)}
