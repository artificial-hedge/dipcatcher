"""Structured space (SYNTHETIC)."""

from __future__ import annotations


def ss_ok(structured: bool, topos: bool) -> bool:
    """Structured:
    structured
    infinity
    topos —
    Lurie
    structured."""
    return structured and topos


def lurie_structure(ls: bool) -> bool:
    """Lurie
    structure:
    Lurie
    structure
    sheaf
    —
    DAG
    V."""
    return ls


def _bench_structured_space(seed: int = 0) -> float:
    checks = []
    checks.append(ss_ok(True, True))
    checks.append(not ss_ok(False, True))
    checks.append(lurie_structure(True))
    checks.append(not lurie_structure(False))
    checks.append(True)  # Lurie
    return float(sum(checks) / len(checks))


def bench_structured_space(seed: int = 0) -> dict[str, float]:
    return {"synthetic_structured_space": _bench_structured_space(seed)}
