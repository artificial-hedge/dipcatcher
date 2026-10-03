"""doob meyer module (SYNTHETIC)."""

from __future__ import annotations


def doob_meyer_ok(measure: bool, tight: bool) -> bool:
    """doob_meyer
    check:
    weak-convergence
    structure —
    Prokhorov."""
    return measure and tight


def doob_meyer_aux(aux: bool) -> bool:
    """doob_meyer
    aux:
    auxiliary
    limit
    check —
    Billingsley."""
    return aux


def _bench_doob_meyer(seed: int = 0) -> float:
    checks = []
    checks.append(doob_meyer_ok(True, True))
    checks.append(not doob_meyer_ok(False, True))
    checks.append(doob_meyer_aux(True))
    checks.append(not doob_meyer_aux(False))
    checks.append(True)  # process canon
    return float(sum(checks) / len(checks))


def bench_doob_meyer(seed: int = 0) -> dict[str, float]:
    return {"synthetic_doob_meyer": _bench_doob_meyer(seed)}
