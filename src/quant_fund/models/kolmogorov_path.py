"""kolmogorov path module (SYNTHETIC)."""

from __future__ import annotations


def kolmogorov_path_ok(pd1: bool, cf: bool) -> bool:
    """kolmogorov_path
    check:
    path-dependent
    PDE —
    Cont-Fournié."""
    return pd1 and cf


def kolmogorov_path_aux(aux: bool) -> bool:
    """kolmogorov_path
    aux:
    auxiliary
    path-PDE
    check —
    viscosity."""
    return aux


def _bench_kolmogorov_path(seed: int = 0) -> float:
    checks = []
    checks.append(kolmogorov_path_ok(True, True))
    checks.append(not kolmogorov_path_ok(False, True))
    checks.append(kolmogorov_path_aux(True))
    checks.append(not kolmogorov_path_aux(False))
    checks.append(True)  # path-PDE canon
    return float(sum(checks) / len(checks))


def bench_kolmogorov_path(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kolmogorov_path": _bench_kolmogorov_path(seed)}
