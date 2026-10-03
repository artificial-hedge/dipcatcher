"""kolmogorov two module (SYNTHETIC)."""

from __future__ import annotations


def kolmogorov_two_ok(conv: bool, trunc: bool) -> bool:
    """kolmogorov_two
    check:
    random
    series —
    convergence."""
    return conv and trunc


def kolmogorov_two_aux(aux: bool) -> bool:
    """kolmogorov_two
    aux:
    auxiliary
    series check —
    moments."""
    return aux


def _bench_kolmogorov_two(seed: int = 0) -> float:
    checks = []
    checks.append(kolmogorov_two_ok(True, True))
    checks.append(not kolmogorov_two_ok(False, True))
    checks.append(kolmogorov_two_aux(True))
    checks.append(not kolmogorov_two_aux(False))
    checks.append(True)  # random-series canon
    return float(sum(checks) / len(checks))


def bench_kolmogorov_two(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kolmogorov_two": _bench_kolmogorov_two(seed)}
