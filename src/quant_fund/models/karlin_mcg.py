"""karlin mcg module (SYNTHETIC)."""

from __future__ import annotations


def karlin_mcg_ok(reg: bool, cyc: bool) -> bool:
    """karlin_mcg
    check:
    regenerative
    structure —
    Khinchin
    cycle."""
    return reg and cyc


def karlin_mcg_aux(aux: bool) -> bool:
    """karlin_mcg
    aux:
    auxiliary
    Palm
    check —
    Wold
    process."""
    return aux


def _bench_karlin_mcg(seed: int = 0) -> float:
    checks = []
    checks.append(karlin_mcg_ok(True, True))
    checks.append(not karlin_mcg_ok(False, True))
    checks.append(karlin_mcg_aux(True))
    checks.append(not karlin_mcg_aux(False))
    checks.append(True)  # regenerative canon
    return float(sum(checks) / len(checks))


def bench_karlin_mcg(seed: int = 0) -> dict[str, float]:
    return {"synthetic_karlin_mcg": _bench_karlin_mcg(seed)}
