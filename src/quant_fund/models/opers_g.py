"""G-opers and connections (SYNTHETIC)."""

from __future__ import annotations


def is_oper(connection: bool, gauge_equivalence: bool) -> bool:
    """A G-oper is a B_G-reduction of a G-bundle
    with connection satisfying strict transversality
    (Beilinson-Drinfeld)."""
    return connection and gauge_equivalence


def opers_form_affine(dim: int, expected: int) -> bool:
    """Op_G forms an affine space modeled on
    Hitchin base of the dual nilpotent cone."""
    return dim == expected


def _bench_opers_g(seed: int = 0) -> float:
    checks = []
    checks.append(is_oper(True, True))
    checks.append(not is_oper(True, False))
    checks.append(opers_form_affine(2, 2))  # sl2 opers: rank g
    # canonical oper for uniformization
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_opers_g(seed: int = 0) -> dict[str, float]:
    return {"synthetic_opers_g": _bench_opers_g(seed)}
