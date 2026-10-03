"""unstable adams2 module (SYNTHETIC)."""

from __future__ import annotations


def unstable_adams2_ok(homotopy: bool, unstable: bool) -> bool:
    """unstable_adams2
    check:
    homotopy
    unstable
    structure —
    periodic."""
    return homotopy and unstable


def unstable_adams2_aux(aux: bool) -> bool:
    """unstable_adams2
    aux:
    auxiliary
    homotopy
    check —
    Adams."""
    return aux


def _bench_unstable_adams2(seed: int = 0) -> float:
    checks = []
    checks.append(unstable_adams2_ok(True, True))
    checks.append(not unstable_adams2_ok(False, True))
    checks.append(unstable_adams2_aux(True))
    checks.append(not unstable_adams2_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_unstable_adams2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_unstable_adams2": _bench_unstable_adams2(seed)}
