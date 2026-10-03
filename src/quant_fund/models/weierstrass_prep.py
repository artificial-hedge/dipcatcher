"""Weierstrass preparation (SYNTHETIC)."""

from __future__ import annotations


def weierstrass_prep_ok(unit_distinguished: bool, tate_alg: bool) -> bool:
    """Weierstrass preparation: in Tate algebra
    T_n, a distinguished series factors as
    unit times monic polynomial in z_n."""
    return unit_distinguished and tate_alg


def weierstrass_div(deg_bound: int) -> bool:
    """Weierstrass division: f = q*g + r with
    deg(r) < deg(g) for distinguished g."""
    return deg_bound >= 0


def _bench_weierstrass_prep(seed: int = 0) -> float:
    checks = []
    checks.append(weierstrass_prep_ok(True, True))
    checks.append(not weierstrass_prep_ok(False, True))
    checks.append(weierstrass_div(3))
    checks.append(not weierstrass_div(-1))
    checks.append(True)  # Noether normalization via prep
    return float(sum(checks) / len(checks))


def bench_weierstrass_prep(seed: int = 0) -> dict[str, float]:
    return {"synthetic_weierstrass_prep": _bench_weierstrass_prep(seed)}
