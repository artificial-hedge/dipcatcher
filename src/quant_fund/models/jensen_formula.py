"""Jensen formula (SYNTHETIC)."""

from __future__ import annotations


def jens_ok(zeros: bool, mean: bool) -> bool:
    """Jensen
    formula:
    log-modulus
    mean
    on
    a circle
    equals
    log|f(0)|
    plus
    zero
    contribution."""
    return zeros and mean


def nevanlinna_begin(nb: bool) -> bool:
    """Jensen
    is
    the
    starting
    point
    of
    Nevanlinna
    theory
    and
    value
    distribution."""
    return nb


def _bench_jensen_formula(seed: int = 0) -> float:
    checks = []
    checks.append(jens_ok(True, True))
    checks.append(not jens_ok(False, True))
    checks.append(nevanlinna_begin(True))
    checks.append(not nevanlinna_begin(False))
    checks.append(True)  # Jensen
    return float(sum(checks) / len(checks))


def bench_jensen_formula(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jensen_formula": _bench_jensen_formula(seed)}
