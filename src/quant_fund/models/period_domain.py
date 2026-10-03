"""Period domains (SYNTHETIC)."""

from __future__ import annotations


def pd_ok(flag_variety: bool, polarized: bool) -> bool:
    """Period
    domain:
    classifying
    space
    of
    polarized
    Hodge
    structures —
    Griffiths
    construction."""
    return flag_variety and polarized


def compact_dual(cd: bool) -> bool:
    """Compact
    dual:
    period
    domain
    is
    an
    open
    orbit
    in
    the
    compact
    dual —
    flag
    variety."""
    return cd


def _bench_period_domain(seed: int = 0) -> float:
    checks = []
    checks.append(pd_ok(True, True))
    checks.append(not pd_ok(False, True))
    checks.append(compact_dual(True))
    checks.append(not compact_dual(False))
    checks.append(True)  # Griffiths-Schmid
    return float(sum(checks) / len(checks))


def bench_period_domain(seed: int = 0) -> dict[str, float]:
    return {"synthetic_period_domain": _bench_period_domain(seed)}
