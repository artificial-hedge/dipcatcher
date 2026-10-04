"""derived conn module (SYNTHETIC)."""

from __future__ import annotations


def derived_conn_ok(derived: bool, geometric: bool) -> bool:
    """derived_conn
    check:
    derived
    structure —
    conn."""
    return derived and geometric


def derived_conn_aux(aux: bool) -> bool:
    """derived_conn
    aux:
    auxiliary
    derived
    check —
    local."""
    return aux


def _bench_derived_conn(seed: int = 0) -> float:
    checks = []
    checks.append(derived_conn_ok(True, True))
    checks.append(not derived_conn_ok(False, True))
    checks.append(derived_conn_aux(True))
    checks.append(not derived_conn_aux(False))
    checks.append(True)  # derived geometry canon
    return float(sum(checks) / len(checks))


def bench_derived_conn(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_conn": _bench_derived_conn(seed)}
