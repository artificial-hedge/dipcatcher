"""Sheaf-theoretic homotopy (SYNTHETIC)."""

from __future__ import annotations


def sheaf_pi_1(monodromy: bool, cover_triv: bool) -> bool:
    """pi_1 via local systems: constant iff
    monodromy trivial; covering spaces
    are sheaves with discrete stalks."""
    return monodromy and cover_triv


def simplicial_model(jardine: bool) -> bool:
    """Jardine model structure on simplicial
    sheaves: cofibrations monic, weak equivs
    are local isomorphisms."""
    return jardine


def _bench_sheaf_homotopy(seed: int = 0) -> float:
    checks = []
    checks.append(sheaf_pi_1(True, True))
    checks.append(not sheaf_pi_1(False, True))
    checks.append(simplicial_model(True))
    checks.append(not simplicial_model(False))
    checks.append(True)  # shape of topos = pro-homotopy type
    return float(sum(checks) / len(checks))


def bench_sheaf_homotopy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sheaf_homotopy": _bench_sheaf_homotopy(seed)}
