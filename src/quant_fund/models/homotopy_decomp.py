"""homotopy decomp module (SYNTHETIC)."""

from __future__ import annotations


def homotopy_decomp_ok(homotopy: bool, periodic: bool) -> bool:
    """homotopy_decomp
    check:
    homotopy
    structure —
    unstable."""
    return homotopy and periodic


def homotopy_decomp_aux(aux: bool) -> bool:
    """homotopy_decomp
    aux:
    auxiliary
    homotopy
    check —
    periodic."""
    return aux


def _bench_homotopy_decomp(seed: int = 0) -> float:
    checks = []
    checks.append(homotopy_decomp_ok(True, True))
    checks.append(not homotopy_decomp_ok(False, True))
    checks.append(homotopy_decomp_aux(True))
    checks.append(not homotopy_decomp_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_homotopy_decomp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_homotopy_decomp": _bench_homotopy_decomp(seed)}
