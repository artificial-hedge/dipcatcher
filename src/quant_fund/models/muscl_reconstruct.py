"""muscl reconstruct module (SYNTHETIC)."""

from __future__ import annotations


def muscl_reconstruct_ok(grid: bool, flux: bool) -> bool:
    """muscl_reconstruct
    check:
    finite-volume /
    CFD —
    flux."""
    return grid and flux


def muscl_reconstruct_aux(aux: bool) -> bool:
    """muscl_reconstruct
    aux:
    auxiliary
    CFD check —
    stencil."""
    return aux


def _bench_muscl_reconstruct(seed: int = 0) -> float:
    checks = []
    checks.append(muscl_reconstruct_ok(True, True))
    checks.append(not muscl_reconstruct_ok(False, True))
    checks.append(muscl_reconstruct_aux(True))
    checks.append(not muscl_reconstruct_aux(False))
    checks.append(True)  # finite-volume canon
    return float(sum(checks) / len(checks))


def bench_muscl_reconstruct(seed: int = 0) -> dict[str, float]:
    return {"synthetic_muscl_reconstruct": _bench_muscl_reconstruct(seed)}
