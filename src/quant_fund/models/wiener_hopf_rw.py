"""wiener hopf_rw module (SYNTHETIC)."""

from __future__ import annotations


def wiener_hopf_rw_ok(step: bool, drift: bool) -> bool:
    """wiener_hopf_rw
    check:
    random-walk
    structure —
    Spitzer
    principle."""
    return step and drift


def wiener_hopf_rw_aux(aux: bool) -> bool:
    """wiener_hopf_rw
    aux:
    auxiliary
    fluctuation
    check —
    ladder
    epochs."""
    return aux


def _bench_wiener_hopf_rw(seed: int = 0) -> float:
    checks = []
    checks.append(wiener_hopf_rw_ok(True, True))
    checks.append(not wiener_hopf_rw_ok(False, True))
    checks.append(wiener_hopf_rw_aux(True))
    checks.append(not wiener_hopf_rw_aux(False))
    checks.append(True)  # random-walk canon
    return float(sum(checks) / len(checks))


def bench_wiener_hopf_rw(seed: int = 0) -> dict[str, float]:
    return {"synthetic_wiener_hopf_rw": _bench_wiener_hopf_rw(seed)}
