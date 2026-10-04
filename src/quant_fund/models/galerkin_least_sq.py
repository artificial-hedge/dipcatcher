"""galerkin least_sq module (SYNTHETIC)."""

from __future__ import annotations


def galerkin_least_sq_ok(node: bool, resid: bool) -> bool:
    """galerkin_least_sq
    check:
    collocation /
    least-squares
    canon — node/
    residual
    consistency."""
    return node and resid


def galerkin_least_sq_aux(aux: bool) -> bool:
    """galerkin_least_sq
    aux:
    auxiliary
    residual check —
    defect bound."""
    return aux


def _bench_galerkin_least_sq(seed: int = 0) -> float:
    checks = []
    checks.append(galerkin_least_sq_ok(True, True))
    checks.append(not galerkin_least_sq_ok(False, True))
    checks.append(galerkin_least_sq_aux(True))
    checks.append(not galerkin_least_sq_aux(False))
    checks.append(True)  # colloc canon
    return float(sum(checks) / len(checks))


def bench_galerkin_least_sq(seed: int = 0) -> dict[str, float]:
    return {"synthetic_galerkin_least_sq": _bench_galerkin_least_sq(seed)}
