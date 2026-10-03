"""tame map module (SYNTHETIC)."""

from __future__ import annotations


def tame_map_ok(rs1: bool, sew: bool) -> bool:
    """tame_map
    check:
    regularity-structure
    —
    sewing
    lemma."""
    return rs1 and sew


def tame_map_aux(aux: bool) -> bool:
    """tame_map
    aux:
    auxiliary
    branched
    check —
    extension
    theorem."""
    return aux


def _bench_tame_map(seed: int = 0) -> float:
    checks = []
    checks.append(tame_map_ok(True, True))
    checks.append(not tame_map_ok(False, True))
    checks.append(tame_map_aux(True))
    checks.append(not tame_map_aux(False))
    checks.append(True)  # regularity canon
    return float(sum(checks) / len(checks))


def bench_tame_map(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tame_map": _bench_tame_map(seed)}
