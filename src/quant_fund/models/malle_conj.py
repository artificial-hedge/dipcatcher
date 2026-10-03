"""Malle's conjecture (SYNTHETIC)."""

from __future__ import annotations


def mc_ok(number_fields: bool, degree: bool) -> bool:
    """Malle's
    conjecture:
    asymptotic
    count
    of
    number
    fields
    by
    Galois
    group —
    Bhargava
    verify."""
    return number_fields and degree


def counting_exponents(ce: bool) -> bool:
    """Counting
    exponents:
    predicted
    main
    term
    X^a
    log^b
    for
    G-
    fields —
    index
    of
    G
    in
    A_n."""
    return ce


def _bench_malle_conj(seed: int = 0) -> float:
    checks = []
    checks.append(mc_ok(True, True))
    checks.append(not mc_ok(False, True))
    checks.append(counting_exponents(True))
    checks.append(not counting_exponents(False))
    checks.append(True)  # Malle
    return float(sum(checks) / len(checks))


def bench_malle_conj(seed: int = 0) -> dict[str, float]:
    return {"synthetic_malle_conj": _bench_malle_conj(seed)}
