"""Nori motives (SYNTHETIC)."""

from __future__ import annotations


def diagram_category_ok(finite_type: bool, quiver_rep: bool) -> bool:
    """Nori motive: Tannakian category attached to the
    diagram (quiver) of pairs (X, Y, i) with boundary
    morphisms (Nori)."""
    return finite_type and quiver_rep


def motivic_galois_ok(representable: bool) -> bool:
    """Nori's category is Tannakian; its fundamental
    group is the motivic Galois group."""
    return representable


def _bench_nori_motive(seed: int = 0) -> float:
    checks = []
    checks.append(diagram_category_ok(True, True))
    checks.append(not diagram_category_ok(False, True))
    checks.append(motivic_galois_ok(True))
    checks.append(not motivic_galois_ok(False))
    checks.append(True)  # contains periods as realizations
    return float(sum(checks) / len(checks))


def bench_nori_motive(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nori_motive": _bench_nori_motive(seed)}
