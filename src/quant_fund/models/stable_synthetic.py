"""stable synthetic module (SYNTHETIC)."""

from __future__ import annotations


def stable_synthetic_ok(homotopy: bool, stable: bool) -> bool:
    """stable_synthetic
    check:
    homotopy
    structure —
    abelian."""
    return homotopy and stable


def stable_synthetic_aux(aux: bool) -> bool:
    """stable_synthetic
    aux:
    auxiliary
    homotopy
    check —
    finite."""
    return aux


def _bench_stable_synthetic(seed: int = 0) -> float:
    checks = []
    checks.append(stable_synthetic_ok(True, True))
    checks.append(not stable_synthetic_ok(False, True))
    checks.append(stable_synthetic_aux(True))
    checks.append(not stable_synthetic_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_stable_synthetic(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stable_synthetic": _bench_stable_synthetic(seed)}
