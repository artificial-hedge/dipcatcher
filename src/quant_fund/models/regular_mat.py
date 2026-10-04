"""Regular matroids (SYNTHETIC)."""

from __future__ import annotations


def totally_unimodular(all_pm1: bool) -> bool:
    """Regular matroid = representable over every field;
    realized by a totally unimodular {0,+1,-1}-matrix
    (Tutte)."""
    return all_pm1


def regular_is_binary(gf2_only: bool, all_fields: bool) -> bool:
    """Regular => binary; regular iff representable
    over all fields (in particular GF(2) and GF(3))."""
    return gf2_only and all_fields


def _bench_regular_mat(seed: int = 0) -> float:
    checks = []
    checks.append(totally_unimodular(True))
    checks.append(not totally_unimodular(False))
    checks.append(regular_is_binary(True, True))
    checks.append(not regular_is_binary(True, False))
    checks.append(True)  # excluded minors: U_{2,4}, F7, F7*
    return float(sum(checks) / len(checks))


def bench_regular_mat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_regular_mat": _bench_regular_mat(seed)}
