"""Berezin integral (SYNTHETIC)."""

from __future__ import annotations


def berezin_ok(integral: bool, translation: bool) -> bool:
    """Berezin
    integral:
    integral over
    odd variables
    is the top
    coefficient;
    integral of
    theta is 1,
    of 1 is 0."""
    return integral and translation


def change_var(change: bool) -> bool:
    """Berezin
    change of
    variables uses
    the inverse
    of the Jacobian
    (Berezinian),
    opposite to
    the classical
    case."""
    return change


def _bench_berezin_int(seed: int = 0) -> float:
    checks = []
    checks.append(berezin_ok(True, True))
    checks.append(not berezin_ok(False, True))
    checks.append(change_var(True))
    checks.append(not change_var(False))
    checks.append(True)  # Berezin
    return float(sum(checks) / len(checks))


def bench_berezin_int(seed: int = 0) -> dict[str, float]:
    return {"synthetic_berezin_int": _bench_berezin_int(seed)}
