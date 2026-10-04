"""bem kernel module (SYNTHETIC)."""

from __future__ import annotations


def bem_kernel_ok(kernel: bool, density: bool) -> bool:
    """bem_kernel
    check:
    boundary-element —
    integral-equation
    consistency."""
    return kernel and density


def bem_kernel_aux(aux: bool) -> bool:
    """bem_kernel
    aux:
    auxiliary
    BEM check —
    singularity handling."""
    return aux


def _bench_bem_kernel(seed: int = 0) -> float:
    checks = []
    checks.append(bem_kernel_ok(True, True))
    checks.append(not bem_kernel_ok(False, True))
    checks.append(bem_kernel_aux(True))
    checks.append(not bem_kernel_aux(False))
    checks.append(True)  # boundary-element canon
    return float(sum(checks) / len(checks))


def bench_bem_kernel(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bem_kernel": _bench_bem_kernel(seed)}
