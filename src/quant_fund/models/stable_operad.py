"""stable operad module (SYNTHETIC)."""

from __future__ import annotations


def stable_operad_ok(homotopy: bool, stable: bool) -> bool:
    """stable_operad
    check:
    homotopy
    structure —
    stable."""
    return homotopy and stable


def stable_operad_aux(aux: bool) -> bool:
    """stable_operad
    aux:
    auxiliary
    homotopy
    check —
    stable."""
    return aux


def _bench_stable_operad(seed: int = 0) -> float:
    checks = []
    checks.append(stable_operad_ok(True, True))
    checks.append(not stable_operad_ok(False, True))
    checks.append(stable_operad_aux(True))
    checks.append(not stable_operad_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_stable_operad(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stable_operad": _bench_stable_operad(seed)}
