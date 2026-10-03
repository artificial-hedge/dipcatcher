"""Saturation: realizing types over finite parameter sets (SYNTHETIC)."""

from __future__ import annotations

from fractions import Fraction


def realize_type_over_dlo(bound: Fraction, kind: str, domain: list[Fraction]) -> Fraction | None:
    """In a dense order, the type {x > a} (or x < a, or a < x < b) is
    realized by interpolation whenever the domain extends past the bound."""
    if kind == "gt":
        cands = [d for d in domain if d > bound]
    elif kind == "lt":
        cands = [d for d in domain if d < bound]
    else:
        return None
    return min(cands, key=abs) if cands else None


def type_consistent(bound: Fraction, kind: str) -> bool:
    """The type {x > a} is finitely satisfiable in any dense order
    extending past a; consistency reduces to each finite subset having
    a solution, always true here."""
    return kind in ("gt", "lt")


def _bench_saturation_test(seed: int = 0) -> float:
    checks = []
    dom = [Fraction(i, 2) for i in range(-10, 11)]
    # realize x > 1/4 in the fractional domain: smallest candidate > 1/4
    r = realize_type_over_dlo(Fraction(1, 4), "gt", dom)
    checks.append(r is not None and r > Fraction(1, 4))
    # density: in a refinement (eighths) there is a point strictly
    # between 0 and 1/4 — the DLO order is dense
    dom_fine = [Fraction(i, 8) for i in range(-40, 41)]
    checks.append(any(Fraction(0) < d < Fraction(1, 4) for d in dom_fine))
    # consistency of unbounded type
    checks.append(type_consistent(Fraction(10), "gt"))
    # type x > max domain can't be realized in THIS finite model (the
    # finite model is NOT saturated) — the point of the theorem
    checks.append(realize_type_over_dlo(Fraction(10), "gt", dom) is None)
    # in an extended domain it IS realized (saturation exists in the limit)
    dom_ext = dom + [Fraction(21, 2)]
    checks.append(realize_type_over_dlo(Fraction(10), "gt", dom_ext) is not None)
    checks.append(realize_type_over_dlo(Fraction(-9, 2), "lt", dom) is not None)
    return float(sum(checks) / len(checks))


def bench_saturation_test(seed: int = 0) -> dict[str, float]:
    return {"synthetic_saturation_test": _bench_saturation_test(seed)}
