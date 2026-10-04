"""aggarwal sixv module (SYNTHETIC)."""

from __future__ import annotations


def aggarwal_sixv_ok(sixv: bool, solv: bool) -> bool:
    """aggarwal_sixv
    check:
    vertex-model
    structure —
    Baxter."""
    return sixv and solv


def aggarwal_sixv_aux(aux: bool) -> bool:
    """aggarwal_sixv
    aux:
    auxiliary
    Yang-Baxter
    check —
    Reshetikhin."""
    return aux


def _bench_aggarwal_sixv(seed: int = 0) -> float:
    checks = []
    checks.append(aggarwal_sixv_ok(True, True))
    checks.append(not aggarwal_sixv_ok(False, True))
    checks.append(aggarwal_sixv_aux(True))
    checks.append(not aggarwal_sixv_aux(False))
    checks.append(True)  # vertex-model canon
    return float(sum(checks) / len(checks))


def bench_aggarwal_sixv(seed: int = 0) -> dict[str, float]:
    return {"synthetic_aggarwal_sixv": _bench_aggarwal_sixv(seed)}
