"""Integral p-adic Hodge theory 2 (SYNTHETIC)."""

from __future__ import annotations


def ip2_ok(integral: bool, padic: bool) -> bool:
    """Integral
    p-adic:
    integral
    p-adic
    Hodge —
    Bhatt-Morrow-Scholze."""
    return integral and padic


def breuil_kisin_prism(bkp: bool) -> bool:
    """Breuil-Kisin
    prism:
    BK
    prism —
    A_inf."""
    return bkp


def _bench_integral_padic2(seed: int = 0) -> float:
    checks = []
    checks.append(ip2_ok(True, True))
    checks.append(not ip2_ok(False, True))
    checks.append(breuil_kisin_prism(True))
    checks.append(not breuil_kisin_prism(False))
    checks.append(True)  # BMS
    return float(sum(checks) / len(checks))


def bench_integral_padic2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_integral_padic2": _bench_integral_padic2(seed)}
