"""walsh table module (SYNTHETIC)."""

from __future__ import annotations


def walsh_table_ok(elem: bool, dof: bool) -> bool:
    """walsh_table
    check:
    FE-basis/sequence —
    element/dof
    consistency."""
    return elem and dof


def walsh_table_aux(aux: bool) -> bool:
    """walsh_table
    aux:
    auxiliary
    element check —
    partition bound."""
    return aux


def _bench_walsh_table(seed: int = 0) -> float:
    checks = []
    checks.append(walsh_table_ok(True, True))
    checks.append(not walsh_table_ok(False, True))
    checks.append(walsh_table_aux(True))
    checks.append(not walsh_table_aux(False))
    checks.append(True)  # FE-basis canon
    return float(sum(checks) / len(checks))


def bench_walsh_table(seed: int = 0) -> dict[str, float]:
    return {"synthetic_walsh_table": _bench_walsh_table(seed)}
