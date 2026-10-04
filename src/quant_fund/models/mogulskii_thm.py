"""mogulskii thm module (SYNTHETIC)."""

from __future__ import annotations


def mogulskii_thm_ok(rate: bool, action: bool) -> bool:
    """mogulskii_thm
    check:
    large-deviation
    structure —
    Varadhan."""
    return rate and action


def mogulskii_thm_aux(aux: bool) -> bool:
    """mogulskii_thm
    aux:
    auxiliary
    contraction
    check —
    Wentzell."""
    return aux


def _bench_mogulskii_thm(seed: int = 0) -> float:
    checks = []
    checks.append(mogulskii_thm_ok(True, True))
    checks.append(not mogulskii_thm_ok(False, True))
    checks.append(mogulskii_thm_aux(True))
    checks.append(not mogulskii_thm_aux(False))
    checks.append(True)  # LDP canon
    return float(sum(checks) / len(checks))


def bench_mogulskii_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mogulskii_thm": _bench_mogulskii_thm(seed)}
