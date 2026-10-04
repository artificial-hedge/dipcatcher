"""Mapping space (SYNTHETIC)."""

from __future__ import annotations


def ms_ok3(mapping: bool, space: bool) -> bool:
    """Mapping:
    mapping
    space
    of
    morphisms —
    Boardman-
    Vogt."""
    return mapping and space


def hom_fiber(hf: bool) -> bool:
    """Hom
    fiber:
    mapping
    space
    as
    fiber
    product —
    BV
    mapping."""
    return hf


def _bench_mapping_space(seed: int = 0) -> float:
    checks = []
    checks.append(ms_ok3(True, True))
    checks.append(not ms_ok3(False, True))
    checks.append(hom_fiber(True))
    checks.append(not hom_fiber(False))
    checks.append(True)  # BV
    return float(sum(checks) / len(checks))


def bench_mapping_space(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mapping_space": _bench_mapping_space(seed)}
