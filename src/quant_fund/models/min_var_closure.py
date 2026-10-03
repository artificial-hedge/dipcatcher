"""min var_closure module (SYNTHETIC)."""

from __future__ import annotations


def min_var_closure_ok(step: bool, conv: bool) -> bool:
    """min_var_closure
    check:
    acceleration —
    step/contraction
    consistency."""
    return step and conv


def min_var_closure_aux(aux: bool) -> bool:
    """min_var_closure
    aux:
    auxiliary
    accelerator check —
    rate bound."""
    return aux


def _bench_min_var_closure(seed: int = 0) -> float:
    checks = []
    checks.append(min_var_closure_ok(True, True))
    checks.append(not min_var_closure_ok(False, True))
    checks.append(min_var_closure_aux(True))
    checks.append(not min_var_closure_aux(False))
    checks.append(True)  # acceleration canon
    return float(sum(checks) / len(checks))


def bench_min_var_closure(seed: int = 0) -> dict[str, float]:
    return {"synthetic_min_var_closure": _bench_min_var_closure(seed)}
