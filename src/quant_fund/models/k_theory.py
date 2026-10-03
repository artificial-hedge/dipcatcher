"""Topological K-theory on spheres (SYNTHETIC)."""

from __future__ import annotations


def k0(sphere_dim: int) -> int:
    """K^0(S^n): Z for n even via Bott periodicity... reduced K^0(S^n)
    = Z for n even, 0 for n odd; unreduced K^0(S^n) = Z + reduced."""
    reduced = 1 if sphere_dim % 2 == 0 else 0
    return 1 + reduced


def _bench_k_theory(seed: int = 0) -> float:
    checks = []
    # K^0(S^0) = Z^2 (S^0 is two points)
    checks.append(k0(0) == 2)
    # K^0(S^1) = Z (reduced K^0(S^1) = 0)
    checks.append(k0(1) == 1)
    # K^0(S^2) = Z^2 (Hopf bundle generates reduced part)
    checks.append(k0(2) == 2)
    # Bott periodicity: reduced K^0(S^{n+2}) = reduced K^0(S^n)
    checks.append((k0(4) - 1) == (k0(2) - 1))
    # odd spheres have zero reduced K
    checks.append(k0(3) == 1)
    checks.append(k0(5) == 1)
    return float(sum(checks) / len(checks))


def bench_k_theory(seed: int = 0) -> dict[str, float]:
    return {"synthetic_k_theory": _bench_k_theory(seed)}
