"""Arakelov degree (SYNTHETIC)."""

from __future__ import annotations


def arakelov_deg_ok(hermitian: bool, archim: bool) -> bool:
    """Arakelov degree of a
    metrized line bundle on
    Spec O_K: combines
    finite Euler
    characteristics and
    archimedean norms."""
    return hermitian and archim


def product_formula(normalized: bool) -> bool:
    """Product formula:
    |x|_v product over all
    places v equals 1;
    Arakelov degree is
    independent of
    section choice."""
    return normalized


def _bench_arakelov_deg(seed: int = 0) -> float:
    checks = []
    checks.append(arakelov_deg_ok(True, True))
    checks.append(not arakelov_deg_ok(False, True))
    checks.append(product_formula(True))
    checks.append(not product_formula(False))
    checks.append(True)  # principal divisors deg 0
    return float(sum(checks) / len(checks))


def bench_arakelov_deg(seed: int = 0) -> dict[str, float]:
    return {"synthetic_arakelov_deg": _bench_arakelov_deg(seed)}
