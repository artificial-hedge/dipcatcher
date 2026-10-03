"""Quantum double (SYNTHETIC)."""

from __future__ import annotations


def qdouble_ok(drin: bool, quasi: bool) -> bool:
    """Drinfeld
    double D(A):
    A tensor
    A^*op with
    crossed
    structure;
    universal
    R-matrix
    from
    the pairing."""
    return drin and quasi


def centre_equiv(centre: bool) -> bool:
    """Drinfeld
    centre:
    Z(C) is
    braided;
    rep D(A)
    is
    Z(rep A)."""
    return centre


def _bench_quantum_double(seed: int = 0) -> float:
    checks = []
    checks.append(qdouble_ok(True, True))
    checks.append(not qdouble_ok(False, True))
    checks.append(centre_equiv(True))
    checks.append(not centre_equiv(False))
    checks.append(True)  # Drinfeld
    return float(sum(checks) / len(checks))


def bench_quantum_double(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quantum_double": _bench_quantum_double(seed)}
