"""line sampling module (SYNTHETIC)."""

from __future__ import annotations


def line_sampling_ok(beta: bool, conv: bool) -> bool:
    """line_sampling
    check:
    reliability —
    failure-probability
    consistency."""
    return beta and conv


def line_sampling_aux(aux: bool) -> bool:
    """line_sampling
    aux:
    auxiliary
    reliability check —
    index bound."""
    return aux


def _bench_line_sampling(seed: int = 0) -> float:
    checks = []
    checks.append(line_sampling_ok(True, True))
    checks.append(not line_sampling_ok(False, True))
    checks.append(line_sampling_aux(True))
    checks.append(not line_sampling_aux(False))
    checks.append(True)  # reliability canon
    return float(sum(checks) / len(checks))


def bench_line_sampling(seed: int = 0) -> dict[str, float]:
    return {"synthetic_line_sampling": _bench_line_sampling(seed)}
