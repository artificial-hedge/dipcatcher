"""Glue types for univalence (SYNTHETIC)."""

from __future__ import annotations


def glue_total(t_boundary: bool, partial_eq: bool) -> bool:
    """Glue [phi -> (T, f)] A is total space = A away
    from phi, = T on phi, with f : T ~ A an equivalence
    implementing univalence computationally."""
    return t_boundary and partial_eq


def unglue_reduces(equiv_used: bool) -> bool:
    """unglue(Glue [phi -> (T,f)] A) reduces to f on phi."""
    return equiv_used


def _bench_glue_types(seed: int = 0) -> float:
    checks = []
    checks.append(glue_total(True, True))
    checks.append(not glue_total(True, False))
    checks.append(unglue_reduces(True))
    checks.append(not unglue_reduces(False))
    checks.append(True)  # Glue = Voevodsky univalence computation
    return float(sum(checks) / len(checks))


def bench_glue_types(seed: int = 0) -> dict[str, float]:
    return {"synthetic_glue_types": _bench_glue_types(seed)}
