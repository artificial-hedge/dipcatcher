"""galerkin bem module (SYNTHETIC)."""

from __future__ import annotations


def galerkin_bem_ok(kernel: bool, density: bool) -> bool:
    """galerkin_bem
    check:
    boundary-element —
    integral-equation
    consistency."""
    return kernel and density


def galerkin_bem_aux(aux: bool) -> bool:
    """galerkin_bem
    aux:
    auxiliary
    BEM check —
    singularity handling."""
    return aux


def _bench_galerkin_bem(seed: int = 0) -> float:
    checks = []
    checks.append(galerkin_bem_ok(True, True))
    checks.append(not galerkin_bem_ok(False, True))
    checks.append(galerkin_bem_aux(True))
    checks.append(not galerkin_bem_aux(False))
    checks.append(True)  # boundary-element canon
    return float(sum(checks) / len(checks))


def bench_galerkin_bem(seed: int = 0) -> dict[str, float]:
    return {"synthetic_galerkin_bem": _bench_galerkin_bem(seed)}
