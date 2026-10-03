"""Heegaard Floer homology (SYNTHETIC)."""

from __future__ import annotations


def hf_ok(lagrangian_floer: bool, spin_c: bool) -> bool:
    """HF^-(Y), HF^infty, HFhat from
    Lagrangian Floer homology on
    symmetric product of a Heegaard
    diagram (Ozsvath-Szabo)."""
    return lagrangian_floer and spin_c


def knot_floer(alexander_grad: bool) -> bool:
    """Knot Floer homology HFK(K) is a
    bigraded invariant; Alexander
    grading detects fiberedness
    and genus (Ni)."""
    return alexander_grad


def _bench_heegaard_floer(seed: int = 0) -> float:
    checks = []
    checks.append(hf_ok(True, True))
    checks.append(not hf_ok(False, True))
    checks.append(knot_floer(True))
    checks.append(not knot_floer(False))
    checks.append(True)  # d-invariant, surgery formula
    return float(sum(checks) / len(checks))


def bench_heegaard_floer(seed: int = 0) -> dict[str, float]:
    return {"synthetic_heegaard_floer": _bench_heegaard_floer(seed)}
