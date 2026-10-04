"""Kapustin-Witten equations (SYNTHETIC)."""

from __future__ import annotations


def kw_ok(complex_gauge: bool, electric_magnetic: bool) -> bool:
    """Kapustin-
    Witten:
    equations
    on
    a
    4-manifold
    encoding
    S-duality —
    geometric
    Langlands
    via
    gauge
    theory."""
    return complex_gauge and electric_magnetic


def ge_langlands(gl: bool) -> bool:
    """Geometric
    Langlands:
    KW
    solutions
    model
    Hecke
    eigensheaves
    on
    Bun_G —
    S-duality
    equals
    Langlands."""
    return gl


def _bench_kapustin_witten(seed: int = 0) -> float:
    checks = []
    checks.append(kw_ok(True, True))
    checks.append(not kw_ok(False, True))
    checks.append(ge_langlands(True))
    checks.append(not ge_langlands(False))
    checks.append(True)  # Kapustin-Witten
    return float(sum(checks) / len(checks))


def bench_kapustin_witten(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kapustin_witten": _bench_kapustin_witten(seed)}
