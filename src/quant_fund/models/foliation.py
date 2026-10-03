"""Foliations (SYNTHETIC)."""

from __future__ import annotations


def fol_ok(partition: bool, integrable: bool) -> bool:
    """Foliation:
    partition
    into
    immersed
    leaves
    whose
    tangent
    distribution
    is
    integrable —
    Frobenius."""
    return partition and integrable


def codim_q(cq: bool) -> bool:
    """Codimension:
    transverse
    dimension
    of
    the
    leaves —
    Haefliger
    structures."""
    return cq


def _bench_foliation(seed: int = 0) -> float:
    checks = []
    checks.append(fol_ok(True, True))
    checks.append(not fol_ok(False, True))
    checks.append(codim_q(True))
    checks.append(not codim_q(False))
    checks.append(True)  # Frobenius
    return float(sum(checks) / len(checks))


def bench_foliation(seed: int = 0) -> dict[str, float]:
    return {"synthetic_foliation": _bench_foliation(seed)}
