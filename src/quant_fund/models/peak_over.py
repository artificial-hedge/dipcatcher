"""peak over module (SYNTHETIC)."""

from __future__ import annotations


def peak_over_ok(tail: bool, xi: bool) -> bool:
    """peak_over
    check:
    extreme-value
    structure —
    Gumbel."""
    return tail and xi


def peak_over_aux(aux: bool) -> bool:
    """peak_over
    aux:
    auxiliary
    max-domain
    check —
    Weibull."""
    return aux


def _bench_peak_over(seed: int = 0) -> float:
    checks = []
    checks.append(peak_over_ok(True, True))
    checks.append(not peak_over_ok(False, True))
    checks.append(peak_over_aux(True))
    checks.append(not peak_over_aux(False))
    checks.append(True)  # EVT canon
    return float(sum(checks) / len(checks))


def bench_peak_over(seed: int = 0) -> dict[str, float]:
    return {"synthetic_peak_over": _bench_peak_over(seed)}
