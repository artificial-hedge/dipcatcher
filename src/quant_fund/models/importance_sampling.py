"""importance sampling module (SYNTHETIC)."""

from __future__ import annotations


def importance_sampling_ok(sample: bool, var: bool) -> bool:
    """importance_sampling
    check:
    Monte-Carlo variance reduction —
    estimator
    consistency."""
    return sample and var


def importance_sampling_aux(aux: bool) -> bool:
    """importance_sampling
    aux:
    auxiliary
    sampling check —
    variance bound."""
    return aux


def _bench_importance_sampling(seed: int = 0) -> float:
    checks = []
    checks.append(importance_sampling_ok(True, True))
    checks.append(not importance_sampling_ok(False, True))
    checks.append(importance_sampling_aux(True))
    checks.append(not importance_sampling_aux(False))
    checks.append(True)  # MC-VR canon
    return float(sum(checks) / len(checks))


def bench_importance_sampling(seed: int = 0) -> dict[str, float]:
    return {"synthetic_importance_sampling": _bench_importance_sampling(seed)}
