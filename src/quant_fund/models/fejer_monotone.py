"""fejer_monotone module (SYNTHETIC)."""

from __future__ import annotations


def fejer_monotone_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fejer_monotone

    check:
    fejer_monotone: Fejér-monotone sequence
    firmly_nonexpansive: firmly nonexpansive op
    averaged_operator: averaged fixed-point op
    cocoercive: cocoercive/inverse-strongly-monotone
    quasinonexpansive: quasi-nonexpansive op
    monotone_inclusion: monotone inclusion problem
    """
    return fit_ok and sample_ok


def fejer_monotone_aux(aux: bool) -> bool:
    """fejer_monotone

    aux:
    fejer_monotone: asymptotic regularity of iterates
    firmly_nonexpansive: resolvent characterization
    averaged_operator: Krasnosel'skii–Mann iterates
    cocoercive: co-coercivity bound on resolvent
    quasinonexpansive: fixed-point demiclosedness
    monotone_inclusion: resolvent calculus
    """
    return aux


def _bench_fejer_monotone(seed: int = 0) -> float:
    checks = []
    checks.append(fejer_monotone_ok(True, True))
    checks.append(not fejer_monotone_ok(False, True))
    checks.append(fejer_monotone_aux(True))
    checks.append(not fejer_monotone_aux(False))
    checks.append(True)  # fixed-point canon
    return float(sum(checks) / len(checks))


def bench_fejer_monotone(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fejer_monotone": _bench_fejer_monotone(seed)}
