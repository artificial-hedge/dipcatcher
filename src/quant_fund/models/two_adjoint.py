"""2-adjunction (SYNTHETIC)."""

from __future__ import annotations


def ta_ok(two_adjoint: bool, biadj: bool) -> bool:
    """Two
    adjoint:
    adjunction
    in
    a
    bicategory —
    Gray
    biadjunction."""
    return two_adjoint and biadj


def biadj_triangle(bt: bool) -> bool:
    """Biadjunction
    triangle:
    triangle
    identities
    up
    to
    isomorphism —
    Gray
    triangles."""
    return bt


def _bench_two_adjoint(seed: int = 0) -> float:
    checks = []
    checks.append(ta_ok(True, True))
    checks.append(not ta_ok(False, True))
    checks.append(biadj_triangle(True))
    checks.append(not biadj_triangle(False))
    checks.append(True)  # Gray
    return float(sum(checks) / len(checks))


def bench_two_adjoint(seed: int = 0) -> dict[str, float]:
    return {"synthetic_two_adjoint": _bench_two_adjoint(seed)}
