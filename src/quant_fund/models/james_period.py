"""James periodicity (SYNTHETIC)."""

from __future__ import annotations


def jp_ok(james_map: bool, periodicity: bool) -> bool:
    """James
    periodicity:
    repeated
    James
    construction
    periodic —
    James
    periods."""
    return james_map and periodicity


def james_constr2(jc: bool) -> bool:
    """James
    construction:
    J(X)
    models
    loop
    suspension —
    James
    model."""
    return jc


def _bench_james_period(seed: int = 0) -> float:
    checks = []
    checks.append(jp_ok(True, True))
    checks.append(not jp_ok(False, True))
    checks.append(james_constr2(True))
    checks.append(not james_constr2(False))
    checks.append(True)  # James
    return float(sum(checks) / len(checks))


def bench_james_period(seed: int = 0) -> dict[str, float]:
    return {"synthetic_james_period": _bench_james_period(seed)}
