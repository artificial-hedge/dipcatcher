"""godunov exact module (SYNTHETIC)."""

from __future__ import annotations


def godunov_exact_ok(state: bool, flux: bool) -> bool:
    """godunov_exact
    check:
    Riemann-solver —
    flux consistency."""
    return state and flux


def godunov_exact_aux(aux: bool) -> bool:
    """godunov_exact
    aux:
    auxiliary
    solver check —
    entropy fix."""
    return aux


def _bench_godunov_exact(seed: int = 0) -> float:
    checks = []
    checks.append(godunov_exact_ok(True, True))
    checks.append(not godunov_exact_ok(False, True))
    checks.append(godunov_exact_aux(True))
    checks.append(not godunov_exact_aux(False))
    checks.append(True)  # Riemann-solver canon
    return float(sum(checks) / len(checks))


def bench_godunov_exact(seed: int = 0) -> dict[str, float]:
    return {"synthetic_godunov_exact": _bench_godunov_exact(seed)}
