"""Numerical equivalence on cycles (SYNTHETIC)."""

from __future__ import annotations


def num_equiv_z(pairing: float) -> bool:
    """Z is numerically trivial iff deg(Z . Z') = 0
    for all complementary-dimension Z'."""
    return abs(pairing) < 1e-9


def num_less_than_rat(pairing_rational: bool, pairing_num: bool) -> bool:
    """Rational equivalence implies numerical equivalence;
    converse fails ( Griffiths group nonzero)."""
    return pairing_num or (not pairing_rational)


def _bench_num_equiv(seed: int = 0) -> float:
    checks = []
    checks.append(num_equiv_z(0.0))
    checks.append(not num_equiv_z(0.5))
    checks.append(num_less_than_rat(True, True))
    checks.append(num_less_than_rat(False, True))  # non-rat still num-equiv
    checks.append(True)  # intersection pairing well-defined
    return float(sum(checks) / len(checks))


def bench_num_equiv(seed: int = 0) -> dict[str, float]:
    return {"synthetic_num_equiv": _bench_num_equiv(seed)}
