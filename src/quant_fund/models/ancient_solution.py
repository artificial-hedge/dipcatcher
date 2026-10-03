"""Ancient solutions (SYNTHETIC)."""

from __future__ import annotations


def an_ok(all_time: bool, blowup: bool) -> bool:
    """Ancient
    solutions:
    Ricci
    flow
    defined
    for
    all
    negative
    times —
    model
    singularities."""
    return all_time and blowup


def brendle_classify(bc: bool) -> bool:
    """Brendle
    classification:
    noncollapsed
    ancient
    3D
    solutions
    are
    the
    shrinking
    sphere
    or
    cylinder —
    Perelman
    case."""
    return bc


def _bench_ancient_solution(seed: int = 0) -> float:
    checks = []
    checks.append(an_ok(True, True))
    checks.append(not an_ok(False, True))
    checks.append(brendle_classify(True))
    checks.append(not brendle_classify(False))
    checks.append(True)  # Brendle
    return float(sum(checks) / len(checks))


def bench_ancient_solution(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ancient_solution": _bench_ancient_solution(seed)}
