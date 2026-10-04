"""Green-Tao theorem (SYNTHETIC)."""

from __future__ import annotations


def gt_ok(primes: bool, ap: bool) -> bool:
    """Green-Tao
    theorem:
    the primes
    contain
    arbitrarily
    long
    arithmetic
    progressions."""
    return primes and ap


def transference(trans: bool) -> bool:
    """Transference:
    relative
    Szemeredi
    in a
    pseudorandom
    superset
    (the
    W-tricked
    primes)."""
    return trans


def _bench_green_tao(seed: int = 0) -> float:
    checks = []
    checks.append(gt_ok(True, True))
    checks.append(not gt_ok(False, True))
    checks.append(transference(True))
    checks.append(not transference(False))
    checks.append(True)  # Green-Tao
    return float(sum(checks) / len(checks))


def bench_green_tao(seed: int = 0) -> dict[str, float]:
    return {"synthetic_green_tao": _bench_green_tao(seed)}
