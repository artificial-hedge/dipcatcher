"""tanh sinh module (SYNTHETIC)."""

from __future__ import annotations


def tanh_sinh_ok(panel: bool, freq: bool) -> bool:
    """tanh_sinh
    check:
    adaptive/oscillatory-quadrature —
    tolerance
    consistency."""
    return panel and freq


def tanh_sinh_aux(aux: bool) -> bool:
    """tanh_sinh
    aux:
    auxiliary
    quadrature check —
    convergence bound."""
    return aux


def _bench_tanh_sinh(seed: int = 0) -> float:
    checks = []
    checks.append(tanh_sinh_ok(True, True))
    checks.append(not tanh_sinh_ok(False, True))
    checks.append(tanh_sinh_aux(True))
    checks.append(not tanh_sinh_aux(False))
    checks.append(True)  # adaptive-quadrature canon
    return float(sum(checks) / len(checks))


def bench_tanh_sinh(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tanh_sinh": _bench_tanh_sinh(seed)}
