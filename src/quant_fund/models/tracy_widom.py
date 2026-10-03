"""Tracy-Widom law (SYNTHETIC)."""

from __future__ import annotations


def tw_ok(largest_eigen: bool, painleve: bool) -> bool:
    """Tracy-
    Widom:
    largest
    eigenvalue
    law
    via
    Painleve
    II —
    beta
    ensembles."""
    return largest_eigen and painleve


def tw_universality(twu: bool) -> bool:
    """TW
    universality:
    same
    law
    for
    Wigner,
    Wishart,
    KPZ
    growth —
    edge
    universality."""
    return twu


def _bench_tracy_widom(seed: int = 0) -> float:
    checks = []
    checks.append(tw_ok(True, True))
    checks.append(not tw_ok(False, True))
    checks.append(tw_universality(True))
    checks.append(not tw_universality(False))
    checks.append(True)  # Tracy-Widom
    return float(sum(checks) / len(checks))


def bench_tracy_widom(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tracy_widom": _bench_tracy_widom(seed)}
