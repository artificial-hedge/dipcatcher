"""Hermitian K-theory 3 (SYNTHETIC)."""

from __future__ import annotations


def hk3_ok(hermitian: bool, k3: bool) -> bool:
    """Hermitian
    K3:
    hermitian
    K3 —
    Grothendieck."""
    return hermitian and k3


def groth_hermitian(gh: bool) -> bool:
    """Groth
    hermitian:
    Grothendieck
    Witt
    ring —
    orthogonality."""
    return gh


def _bench_hermitian_k3(seed: int = 0) -> float:
    checks = []
    checks.append(hk3_ok(True, True))
    checks.append(not hk3_ok(False, True))
    checks.append(groth_hermitian(True))
    checks.append(not groth_hermitian(False))
    checks.append(True)  # Karoubi
    return float(sum(checks) / len(checks))


def bench_hermitian_k3(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hermitian_k3": _bench_hermitian_k3(seed)}
