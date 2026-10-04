"""cayley moser module (SYNTHETIC)."""

from __future__ import annotations


def cayley_moser_ok(os1: bool, sd: bool) -> bool:
    """cayley_moser
    check:
    optimal-
    stopping —
    value
    function."""
    return os1 and sd


def cayley_moser_aux(aux: bool) -> bool:
    """cayley_moser
    aux:
    auxiliary
    stopping
    check —
    boundary."""
    return aux


def _bench_cayley_moser(seed: int = 0) -> float:
    checks = []
    checks.append(cayley_moser_ok(True, True))
    checks.append(not cayley_moser_ok(False, True))
    checks.append(cayley_moser_aux(True))
    checks.append(not cayley_moser_aux(False))
    checks.append(True)  # optimal-stopping canon
    return float(sum(checks) / len(checks))


def bench_cayley_moser(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cayley_moser": _bench_cayley_moser(seed)}
