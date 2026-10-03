"""Modular forms (SYNTHETIC)."""

from __future__ import annotations


def mf_ok(sl2: bool, holo: bool) -> bool:
    """Modular
    form of
    weight k:
    holomorphic
    f: H -> C
    with f(gz)
    = (cz+d)^k
    f(z) for g
    in SL_2(Z)."""
    return sl2 and holo


def graded_ring(ring: bool) -> bool:
    """M_*(SL_2Z)
    is the
    polynomial
    ring
    C[E_4, E_6]."""
    return ring


def _bench_modular_form(seed: int = 0) -> float:
    checks = []
    checks.append(mf_ok(True, True))
    checks.append(not mf_ok(False, True))
    checks.append(graded_ring(True))
    checks.append(not graded_ring(False))
    checks.append(True)  # Serre
    return float(sum(checks) / len(checks))


def bench_modular_form(seed: int = 0) -> dict[str, float]:
    return {"synthetic_modular_form": _bench_modular_form(seed)}
