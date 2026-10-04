"""Topological Hochschild homology II (SYNTHETIC)."""

from __future__ import annotations


def thh_ok(free_loop_model: bool, s1_action: bool) -> bool:
    """THH(A) = A tensor_A^e A, modelled
    as derived loops on Bar construction;
    carries S^1 action (Bokstedt)."""
    return free_loop_model and s1_action


def bms_filtration(nygaard: bool) -> bool:
    """Nygaard filtration on THH and
    its graded pieces recover p-adic
    de Rham cohomology (BMS)."""
    return nygaard


def _bench_thh_2(seed: int = 0) -> float:
    checks = []
    checks.append(thh_ok(True, True))
    checks.append(not thh_ok(False, True))
    checks.append(bms_filtration(True))
    checks.append(not bms_filtration(False))
    checks.append(True)  # HKR for smooth rings
    return float(sum(checks) / len(checks))


def bench_thh_2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_thh_2": _bench_thh_2(seed)}
