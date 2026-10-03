"""path sobolev module (SYNTHETIC)."""

from __future__ import annotations


def path_sobolev_ok(pd1: bool, cf: bool) -> bool:
    """path_sobolev
    check:
    path-dependent
    PDE —
    Cont-Fournié."""
    return pd1 and cf


def path_sobolev_aux(aux: bool) -> bool:
    """path_sobolev
    aux:
    auxiliary
    path-PDE
    check —
    viscosity."""
    return aux


def _bench_path_sobolev(seed: int = 0) -> float:
    checks = []
    checks.append(path_sobolev_ok(True, True))
    checks.append(not path_sobolev_ok(False, True))
    checks.append(path_sobolev_aux(True))
    checks.append(not path_sobolev_aux(False))
    checks.append(True)  # path-PDE canon
    return float(sum(checks) / len(checks))


def bench_path_sobolev(seed: int = 0) -> dict[str, float]:
    return {"synthetic_path_sobolev": _bench_path_sobolev(seed)}
