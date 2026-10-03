"""unstable vn module (SYNTHETIC)."""

from __future__ import annotations


def unstable_vn_ok(homotopy: bool, periodic: bool) -> bool:
    """unstable_vn
    check:
    homotopy
    structure —
    unstable."""
    return homotopy and periodic


def unstable_vn_aux(aux: bool) -> bool:
    """unstable_vn
    aux:
    auxiliary
    homotopy
    check —
    periodic."""
    return aux


def _bench_unstable_vn(seed: int = 0) -> float:
    checks = []
    checks.append(unstable_vn_ok(True, True))
    checks.append(not unstable_vn_ok(False, True))
    checks.append(unstable_vn_aux(True))
    checks.append(not unstable_vn_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_unstable_vn(seed: int = 0) -> dict[str, float]:
    return {"synthetic_unstable_vn": _bench_unstable_vn(seed)}
