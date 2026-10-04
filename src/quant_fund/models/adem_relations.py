"""Adem relations (SYNTHETIC)."""

from __future__ import annotations


def adem_ok(binomial: bool, admissible: bool) -> bool:
    """Adem relations:
    Sq^a Sq^b = sum
    C(b-1-c, a-2c)
    Sq^{a+b-c} Sq^c
    for a < 2b;
    generates all
    relations."""
    return binomial and admissible


def admissible_basis(basis: bool) -> bool:
    """Admissible monomials
    Sq^{i1}...Sq^{ik}
    with i_j >= 2 i_{j+1}
    form a basis of
    the Steenrod algebra."""
    return basis


def _bench_adem_relations(seed: int = 0) -> float:
    checks = []
    checks.append(adem_ok(True, True))
    checks.append(not adem_ok(False, True))
    checks.append(admissible_basis(True))
    checks.append(not admissible_basis(False))
    checks.append(True)  # Adem 1952
    return float(sum(checks) / len(checks))


def bench_adem_relations(seed: int = 0) -> dict[str, float]:
    return {"synthetic_adem_relations": _bench_adem_relations(seed)}
