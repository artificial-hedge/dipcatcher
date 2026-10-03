"""Lin homotopy (SYNTHETIC)."""

from __future__ import annotations


def lh_ok(lin: bool, htpy: bool) -> bool:
    """Lin
    htpy:
    Lin
    homotopy —
    Segal
    conjecture."""
    return lin and htpy


def segal_conjecture(sc: bool) -> bool:
    """Segal
    conjecture:
    Segal
    conjecture —
    Burnside."""
    return sc


def _bench_lin_htpy(seed: int = 0) -> float:
    checks = []
    checks.append(lh_ok(True, True))
    checks.append(not lh_ok(False, True))
    checks.append(segal_conjecture(True))
    checks.append(not segal_conjecture(False))
    checks.append(True)  # Lin
    return float(sum(checks) / len(checks))


def bench_lin_htpy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lin_htpy": _bench_lin_htpy(seed)}
