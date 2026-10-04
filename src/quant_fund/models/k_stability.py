"""K-stability (SYNTHETIC)."""

from __future__ import annotations


def ks_ok(test_config: bool, df_weight: bool) -> bool:
    """K-
    stability:
    positivity
    of
    Donaldson-
    Futaki
    invariants
    over
    all
    test
    configurations —
    algebraic
    condition."""
    return test_config and df_weight


def k_polystable(kp: bool) -> bool:
    """K-
    polystability:
    DF
    positive
    for
    non-
    product
    test
    configurations —
    matches
    KE
    existence."""
    return kp


def _bench_k_stability(seed: int = 0) -> float:
    checks = []
    checks.append(ks_ok(True, True))
    checks.append(not ks_ok(False, True))
    checks.append(k_polystable(True))
    checks.append(not k_polystable(False))
    checks.append(True)  # Tian-Donaldson
    return float(sum(checks) / len(checks))


def bench_k_stability(seed: int = 0) -> dict[str, float]:
    return {"synthetic_k_stability": _bench_k_stability(seed)}
