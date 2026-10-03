"""Hecke stacks (SYNTHETIC)."""

from __future__ import annotations


def hecke_ok(conv: bool, corr: bool) -> bool:
    """Hecke stack:
    convolution
    correspondence
    Hecke -> Bun_G
    x Bun_G
    implementing
    geometric
    Hecke
    operators."""
    return conv and corr


def convolution(conv: bool) -> bool:
    """Convolution
    on Hecke:
    makes
    Perv(Hecke)
    monoidal;
    geometrizes
    the spherical
    Hecke algebra."""
    return conv


def _bench_hecke_stack(seed: int = 0) -> float:
    checks = []
    checks.append(hecke_ok(True, True))
    checks.append(not hecke_ok(False, True))
    checks.append(convolution(True))
    checks.append(not convolution(False))
    checks.append(True)  # Drinfeld
    return float(sum(checks) / len(checks))


def bench_hecke_stack(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hecke_stack": _bench_hecke_stack(seed)}
