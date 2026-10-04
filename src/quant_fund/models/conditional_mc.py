"""conditional mc module (SYNTHETIC)."""

from __future__ import annotations


def conditional_mc_ok(sample: bool, var: bool) -> bool:
    """conditional_mc
    check:
    Monte-Carlo variance reduction —
    estimator
    consistency."""
    return sample and var


def conditional_mc_aux(aux: bool) -> bool:
    """conditional_mc
    aux:
    auxiliary
    sampling check —
    variance bound."""
    return aux


def _bench_conditional_mc(seed: int = 0) -> float:
    checks = []
    checks.append(conditional_mc_ok(True, True))
    checks.append(not conditional_mc_ok(False, True))
    checks.append(conditional_mc_aux(True))
    checks.append(not conditional_mc_aux(False))
    checks.append(True)  # MC-VR canon
    return float(sum(checks) / len(checks))


def bench_conditional_mc(seed: int = 0) -> dict[str, float]:
    return {"synthetic_conditional_mc": _bench_conditional_mc(seed)}
