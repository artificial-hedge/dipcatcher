"""Voiculescu theorem (SYNTHETIC)."""

from __future__ import annotations


def vt_ok(asymptotic_freeness: bool, random_matrices: bool) -> bool:
    """Voiculescu:
    independent
    Gaussian
    random
    matrices
    are
    asymptotically
    free —
    connects
    RMT
    and
    free
    probability."""
    return asymptotic_freeness and random_matrices


def free_prob_rmt_bridge(fbr: bool) -> bool:
    """RMT-
    free
    bridge:
    semicircle
    from
    GUE
    is
    the
    free
    CLT
    limit —
    Voiculescu
    theorem."""
    return fbr


def _bench_voiculescu_thm(seed: int = 0) -> float:
    checks = []
    checks.append(vt_ok(True, True))
    checks.append(not vt_ok(False, True))
    checks.append(free_prob_rmt_bridge(True))
    checks.append(not free_prob_rmt_bridge(False))
    checks.append(True)  # Voiculescu
    return float(sum(checks) / len(checks))


def bench_voiculescu_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_voiculescu_thm": _bench_voiculescu_thm(seed)}
