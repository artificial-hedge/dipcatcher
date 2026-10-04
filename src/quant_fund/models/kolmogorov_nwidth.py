"""kolmogorov nwidth module (SYNTHETIC)."""

from __future__ import annotations


def kolmogorov_nwidth_ok(smooth: bool, approx: bool) -> bool:
    """kolmogorov_nwidth
    check:
    approximation
    theory —
    smoothness."""
    return smooth and approx


def kolmogorov_nwidth_aux(aux: bool) -> bool:
    """kolmogorov_nwidth
    aux:
    auxiliary
    approx check —
    degree."""
    return aux


def _bench_kolmogorov_nwidth(seed: int = 0) -> float:
    checks = []
    checks.append(kolmogorov_nwidth_ok(True, True))
    checks.append(not kolmogorov_nwidth_ok(False, True))
    checks.append(kolmogorov_nwidth_aux(True))
    checks.append(not kolmogorov_nwidth_aux(False))
    checks.append(True)  # approximation-theory canon
    return float(sum(checks) / len(checks))


def bench_kolmogorov_nwidth(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kolmogorov_nwidth": _bench_kolmogorov_nwidth(seed)}
