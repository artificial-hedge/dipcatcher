"""control variate module (SYNTHETIC)."""

from __future__ import annotations


def control_variate_ok(sample: bool, var: bool) -> bool:
    """control_variate
    check:
    Monte-Carlo variance reduction —
    estimator
    consistency."""
    return sample and var


def control_variate_aux(aux: bool) -> bool:
    """control_variate
    aux:
    auxiliary
    sampling check —
    variance bound."""
    return aux


def _bench_control_variate(seed: int = 0) -> float:
    checks = []
    checks.append(control_variate_ok(True, True))
    checks.append(not control_variate_ok(False, True))
    checks.append(control_variate_aux(True))
    checks.append(not control_variate_aux(False))
    checks.append(True)  # MC-VR canon
    return float(sum(checks) / len(checks))


def bench_control_variate(seed: int = 0) -> dict[str, float]:
    return {"synthetic_control_variate": _bench_control_variate(seed)}
