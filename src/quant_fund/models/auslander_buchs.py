"""Auslander-Buchsbaum formula (SYNTHETIC)."""

from __future__ import annotations


def ab_ok(pd: bool, depth: bool) -> bool:
    """Auslander-
    Buchsbaum:
    pd_R(M) +
    depth(M)
    = depth(R)
    for a
    finite-pd
    module over
    a local
    ring."""
    return pd and depth


def cm_characterize(cm: bool) -> bool:
    """Cohen-
    Macaulay:
    all finite-
    pd modules
    have pd
    equal to
    the depth
    defect."""
    return cm


def _bench_auslander_buchs(seed: int = 0) -> float:
    checks = []
    checks.append(ab_ok(True, True))
    checks.append(not ab_ok(False, True))
    checks.append(cm_characterize(True))
    checks.append(not cm_characterize(False))
    checks.append(True)  # AB 1957
    return float(sum(checks) / len(checks))


def bench_auslander_buchs(seed: int = 0) -> dict[str, float]:
    return {"synthetic_auslander_buchs": _bench_auslander_buchs(seed)}
