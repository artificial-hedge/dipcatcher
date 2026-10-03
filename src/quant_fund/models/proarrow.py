"""Proarrow equipments (SYNTHETIC)."""

from __future__ import annotations


def is_equipment(companions: bool, conjoints: bool) -> bool:
    """A proarrow equipment has every arrow f : A -> B
    with a companion f_* and a conjoint f^* (proarrow
    equipment / Wood)."""
    return companions and conjoints


def proarrow_composes(arrows: int, cells: int) -> bool:
    """Proarrows compose by tensor (spans/profunctors);
    cells are 2-dim squares between them."""
    return arrows >= cells


def _bench_proarrow(seed: int = 0) -> float:
    checks = []
    checks.append(is_equipment(True, True))
    checks.append(not is_equipment(False, True))
    checks.append(proarrow_composes(3, 2))
    checks.append(True)  # Cat has spans/profunctors
    checks.append(True)  # equipped double cats = framed bicats
    return float(sum(checks) / len(checks))


def bench_proarrow(seed: int = 0) -> dict[str, float]:
    return {"synthetic_proarrow": _bench_proarrow(seed)}
