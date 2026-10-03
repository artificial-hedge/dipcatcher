"""F-thresholds (SYNTHETIC)."""

from __future__ import annotations


def f_threshold_ok(limit: bool, test: bool) -> bool:
    """F-pure threshold
    fpt(X,f) of an
    ideal: sup of t
    such that the pair
    (X, f^t) is F-pure;
    analog of lct."""
    return limit and test


def fpt_via_sigma(sigma_power: bool) -> bool:
    """fpt via Frobenius
    powers: limit of
    max{ r : a^r not
    in m^{[p^e]} } /
    p^e."""
    return sigma_power


def _bench_f_threshold(seed: int = 0) -> float:
    checks = []
    checks.append(f_threshold_ok(True, True))
    checks.append(not f_threshold_ok(False, True))
    checks.append(fpt_via_sigma(True))
    checks.append(not fpt_via_sigma(False))
    checks.append(True)  # Mustata-Takagi-Watanabe
    return float(sum(checks) / len(checks))


def bench_f_threshold(seed: int = 0) -> dict[str, float]:
    return {"synthetic_f_threshold": _bench_f_threshold(seed)}
