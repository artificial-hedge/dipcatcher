"""Oseledets theorem (SYNTHETIC)."""

from __future__ import annotations


def osel_ok(lyap: bool, splitting: bool) -> bool:
    """Oseledets
    multiplicative
    ergodic
    theorem:
    Lyapunov
    exponents
    and
    invariant
    splitting
    exist
    a.e. for
    cocycles."""
    return lyap and splitting


def furstenberg_lemma(furst: bool) -> bool:
    """Furstenberg-
    Kesten:
    top
    Lyapunov
    exponent
    is the
    limit
    of
    log-norm
    averages."""
    return furst


def _bench_osceledets(seed: int = 0) -> float:
    checks = []
    checks.append(osel_ok(True, True))
    checks.append(not osel_ok(False, True))
    checks.append(furstenberg_lemma(True))
    checks.append(not furstenberg_lemma(False))
    checks.append(True)  # Oseledets
    return float(sum(checks) / len(checks))


def bench_osceledets(seed: int = 0) -> dict[str, float]:
    return {"synthetic_osceledets": _bench_osceledets(seed)}
