"""stable perf module (SYNTHETIC)."""

from __future__ import annotations


def stable_perf_ok(homotopy: bool, stable: bool) -> bool:
    """stable_perf
    check:
    homotopy
    structure —
    general."""
    return homotopy and stable


def stable_perf_aux(aux: bool) -> bool:
    """stable_perf
    aux:
    auxiliary
    homotopy
    check —
    rational."""
    return aux


def _bench_stable_perf(seed: int = 0) -> float:
    checks = []
    checks.append(stable_perf_ok(True, True))
    checks.append(not stable_perf_ok(False, True))
    checks.append(stable_perf_aux(True))
    checks.append(not stable_perf_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_stable_perf(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stable_perf": _bench_stable_perf(seed)}
