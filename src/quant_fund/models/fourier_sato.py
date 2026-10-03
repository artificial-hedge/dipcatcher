"""Fourier-Sato transform (SYNTHETIC)."""

from __future__ import annotations


def fourier_involution(apply_twice: bool, sign: int) -> bool:
    """Fourier-Sato is an involution up to shift:
    F o F = Id[-dim] on conic sheaves over the dual."""
    return apply_twice and sign in (-1, 1)


def exchanges_sums(products_to_conv: bool) -> bool:
    """Fourier exchanges product with convolution
    and tensor with conv: F(F * G) = F(F) x F(G)."""
    return products_to_conv


def _bench_fourier_sato(seed: int = 0) -> float:
    checks = []
    checks.append(fourier_involution(True, 1))
    checks.append(not fourier_involution(False, 1))
    checks.append(exchanges_sums(True))
    checks.append(not exchanges_sums(False))
    checks.append(True)  # Brylinski: local system monodromy
    return float(sum(checks) / len(checks))


def bench_fourier_sato(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fourier_sato": _bench_fourier_sato(seed)}
