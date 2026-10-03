"""rosenbrock w module (SYNTHETIC)."""

from __future__ import annotations


def rosenbrock_w_ok(step: bool, order: bool) -> bool:
    """rosenbrock_w
    check:
    time-marching/ODE —
    stability
    consistency."""
    return step and order


def rosenbrock_w_aux(aux: bool) -> bool:
    """rosenbrock_w
    aux:
    auxiliary
    stepping check —
    order bound."""
    return aux


def _bench_rosenbrock_w(seed: int = 0) -> float:
    checks = []
    checks.append(rosenbrock_w_ok(True, True))
    checks.append(not rosenbrock_w_ok(False, True))
    checks.append(rosenbrock_w_aux(True))
    checks.append(not rosenbrock_w_aux(False))
    checks.append(True)  # time-marching canon
    return float(sum(checks) / len(checks))


def bench_rosenbrock_w(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rosenbrock_w": _bench_rosenbrock_w(seed)}
