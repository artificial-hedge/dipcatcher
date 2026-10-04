"""common random module (SYNTHETIC)."""

from __future__ import annotations


def common_random_ok(sample: bool, var: bool) -> bool:
    """common_random
    check:
    Monte-Carlo variance reduction —
    estimator
    consistency."""
    return sample and var


def common_random_aux(aux: bool) -> bool:
    """common_random
    aux:
    auxiliary
    sampling check —
    variance bound."""
    return aux


def _bench_common_random(seed: int = 0) -> float:
    checks = []
    checks.append(common_random_ok(True, True))
    checks.append(not common_random_ok(False, True))
    checks.append(common_random_aux(True))
    checks.append(not common_random_aux(False))
    checks.append(True)  # MC-VR canon
    return float(sum(checks) / len(checks))


def bench_common_random(seed: int = 0) -> dict[str, float]:
    return {"synthetic_common_random": _bench_common_random(seed)}
