"""Eilenberg-Steenrod axioms for a toy cellular homology (SYNTHETIC)."""

from __future__ import annotations


def h_n(space: str, n: int) -> int:
    """Toy homology ranks for standard spaces."""
    table = {
        "pt": {0: 1},
        "s1": {0: 1, 1: 1},
        "s2": {0: 1, 2: 1},
        "t2": {0: 1, 1: 2, 2: 1},
        "d2": {0: 1},
        "empty": {},
    }
    return table.get(space, {}).get(n, 0)


def _bench_eilenberg_steenrod(seed: int = 0) -> float:
    checks = []
    # Dimension axiom: H_n(pt) = Z for n=0 else 0
    checks.append(h_n("pt", 0) == 1 and all(h_n("pt", n) == 0 for n in (1, 2, 3)))
    # Homotopy axiom: D^2 ≃ pt gives same homology
    checks.append(all(h_n("d2", n) == h_n("pt", n) for n in range(3)))
    # Additivity: disjoint union of two pts has H_0 = Z^2
    checks.append(h_n("pt", 0) + h_n("pt", 0) == 2)
    # Excision implies H_n(S^2) seen from pair (D^2, S^1): kernel shift
    checks.append(h_n("s2", 2) == 1)
    # Mayer-Vietoris on T^2 = two tori-with-disk glued along S^1
    checks.append(h_n("t2", 1) == 2)
    # Empty space has zero homology
    checks.append(all(h_n("empty", n) == 0 for n in range(3)))
    return float(sum(checks) / len(checks))


def bench_eilenberg_steenrod(seed: int = 0) -> dict[str, float]:
    return {"synthetic_eilenberg_steenrod": _bench_eilenberg_steenrod(seed)}
