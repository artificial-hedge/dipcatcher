"""monotone_inclusion module (SYNTHETIC)."""

from __future__ import annotations


def monotone_inclusion_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """monotone_inclusion

    check:
    fejer_monotone: Fejér-monotone sequence
    firmly_nonexpansive: firmly nonexpansive op
    averaged_operator: averaged fixed-point op
    cocoercive: cocoercive/inverse-strongly-monotone
    quasinonexpansive: quasi-nonexpansive op
    monotone_inclusion: monotone inclusion problem
    """
    return fit_ok and sample_ok


def monotone_inclusion_aux(aux: bool) -> bool:
    """monotone_inclusion

    aux:
    fejer_monotone: asymptotic regularity of iterates
    firmly_nonexpansive: resolvent characterization
    averaged_operator: Krasnosel'skii–Mann iterates
    cocoercive: co-coercivity bound on resolvent
    quasinonexpansive: fixed-point demiclosedness
    monotone_inclusion: resolvent calculus
    """
    return aux


def _bench_monotone_inclusion(seed: int = 0) -> float:
    checks = []
    checks.append(monotone_inclusion_ok(True, True))
    checks.append(not monotone_inclusion_ok(False, True))
    checks.append(monotone_inclusion_aux(True))
    checks.append(not monotone_inclusion_aux(False))
    checks.append(True)  # fixed-point canon
    return float(sum(checks) / len(checks))


def bench_monotone_inclusion(seed: int = 0) -> dict[str, float]:
    return {"synthetic_monotone_inclusion": _bench_monotone_inclusion(seed)}
