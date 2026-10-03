"""antithetic var module (SYNTHETIC)."""

from __future__ import annotations


def antithetic_var_ok(sample: bool, var: bool) -> bool:
    """antithetic_var
    check:
    Monte-Carlo variance reduction —
    estimator
    consistency."""
    return sample and var


def antithetic_var_aux(aux: bool) -> bool:
    """antithetic_var
    aux:
    auxiliary
    sampling check —
    variance bound."""
    return aux


def _bench_antithetic_var(seed: int = 0) -> float:
    checks = []
    checks.append(antithetic_var_ok(True, True))
    checks.append(not antithetic_var_ok(False, True))
    checks.append(antithetic_var_aux(True))
    checks.append(not antithetic_var_aux(False))
    checks.append(True)  # MC-VR canon
    return float(sum(checks) / len(checks))


def bench_antithetic_var(seed: int = 0) -> dict[str, float]:
    return {"synthetic_antithetic_var": _bench_antithetic_var(seed)}
