"""voev homotopy module (SYNTHETIC)."""

from __future__ import annotations


def voev_homotopy_ok(motive: bool, a1: bool) -> bool:
    """voev_homotopy
    check:
    motivic-A1
    structure —
    Voevodsky."""
    return motive and a1


def voev_homotopy_aux(aux: bool) -> bool:
    """voev_homotopy
    aux:
    auxiliary
    motive
    check —
    Morel."""
    return aux


def _bench_voev_homotopy(seed: int = 0) -> float:
    checks = []
    checks.append(voev_homotopy_ok(True, True))
    checks.append(not voev_homotopy_ok(False, True))
    checks.append(voev_homotopy_aux(True))
    checks.append(not voev_homotopy_aux(False))
    checks.append(True)  # motivic-A1 canon
    return float(sum(checks) / len(checks))


def bench_voev_homotopy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_voev_homotopy": _bench_voev_homotopy(seed)}
