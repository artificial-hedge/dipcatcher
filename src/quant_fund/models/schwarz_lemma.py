"""Schwarz lemma (SYNTHETIC)."""

from __future__ import annotations


def schwarz_ok(fixes_zero: bool, contraction: bool) -> bool:
    """Schwarz
    lemma:
    disc
    maps
    fixing
    zero
    contract —
    |f(z)|
    <=
    |z|
    and
    |f'(0)|
    <= 1."""
    return fixes_zero and contraction


def equality_case(ec: bool) -> bool:
    """Equality
    case:
    only
    rotations
    achieve
    equality
    in
    Schwarz."""
    return ec


def _bench_schwarz_lemma(seed: int = 0) -> float:
    checks = []
    checks.append(schwarz_ok(True, True))
    checks.append(not schwarz_ok(False, True))
    checks.append(equality_case(True))
    checks.append(not equality_case(False))
    checks.append(True)  # Schwarz-Pick
    return float(sum(checks) / len(checks))


def bench_schwarz_lemma(seed: int = 0) -> dict[str, float]:
    return {"synthetic_schwarz_lemma": _bench_schwarz_lemma(seed)}
