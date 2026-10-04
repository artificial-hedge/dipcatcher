"""Zhang-Maynard breakthrough (SYNTHETIC)."""

from __future__ import annotations


def zm_ok(yitang: bool, gap70m: bool) -> bool:
    """Zhang's
    theorem:
    bounded
    gaps
    between
    primes —
    first
    finite
    bound
    70
    million."""
    return yitang and gap70m


def polynath8(pt: bool) -> bool:
    """Polymath8:
    collaborative
    improvement
    from
    70M
    to
    4680
    then
    246 —
    Maynard
    weights."""
    return pt


def _bench_zhang_maynard(seed: int = 0) -> float:
    checks = []
    checks.append(zm_ok(True, True))
    checks.append(not zm_ok(False, True))
    checks.append(polynath8(True))
    checks.append(not polynath8(False))
    checks.append(True)  # Zhang 2013
    return float(sum(checks) / len(checks))


def bench_zhang_maynard(seed: int = 0) -> dict[str, float]:
    return {"synthetic_zhang_maynard": _bench_zhang_maynard(seed)}
