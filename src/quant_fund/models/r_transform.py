"""R-transform (SYNTHETIC)."""

from __future__ import annotations


def rt_ok(free_cumulants: bool, additive: bool) -> bool:
    """R-
    transform:
    generating
    function
    of
    free
    cumulants —
    linearizes
    free
    additive
    convolution."""
    return free_cumulants and additive


def cauchy_transform(ct: bool) -> bool:
    """Cauchy
    transform:
    inverse
    relationship
    to
    R-
    transform —
    Voiculescu's
    analytic
    machinery."""
    return ct


def _bench_r_transform(seed: int = 0) -> float:
    checks = []
    checks.append(rt_ok(True, True))
    checks.append(not rt_ok(False, True))
    checks.append(cauchy_transform(True))
    checks.append(not cauchy_transform(False))
    checks.append(True)  # Voiculescu
    return float(sum(checks) / len(checks))


def bench_r_transform(seed: int = 0) -> dict[str, float]:
    return {"synthetic_r_transform": _bench_r_transform(seed)}
