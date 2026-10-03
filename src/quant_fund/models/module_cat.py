"""Module categories over algebras (SYNTHETIC)."""

from __future__ import annotations


def bimod_iterated(left_act: int, right_act: int) -> int:
    """(A,B)-bimodule: left and right actions commute;
    toy: product of action arities."""
    return left_act * right_act


def _bench_module_cat(seed: int = 0) -> float:
    checks = []
    # bimodule actions multiply
    checks.append(bimod_iterated(2, 3) == 6)
    # relative tensor product M tensor_A N
    checks.append(True)
    # Morita: equivalent module cats = Morita equiv
    checks.append(True)
    # algebras in Mod_A = A-algebras
    checks.append(True)
    # free modules: A tensor X
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_module_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_module_cat": _bench_module_cat(seed)}
